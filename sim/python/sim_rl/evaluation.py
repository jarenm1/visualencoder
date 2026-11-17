"""Held-out evaluation: first completed episode per environment, one readback after the horizon."""
from __future__ import annotations

import torch

from .curriculum import BUCKETS, UNIFORM_WEIGHTS
from .ppo import HOVER_ACTION, gpu_range
from .stats import wilson

HEURISTICS = ("hover", "random")
FIELDS = ("completed", "success", "collision", "out_of_bounds", "timeout", "success_steps",
          "return", "return_squared", "steps")


@torch.no_grad()
def evaluate(env, features, policy=None, heuristic=None, seed=0):
    """Deterministic policy (clamped mean) or blind heuristic over one reset of the native finite horizon.

    Curriculum is uniform over visibility buckets. Outcomes are exclusive
    (success > collision > out_of_bounds > timeout) and recorded only for each
    environment's first completed episode.
    """
    if (policy is None) == (heuristic is None):
        raise ValueError("provide exactly one trained policy or blind heuristic")
    if heuristic is not None and heuristic not in HEURISTICS:
        raise ValueError(f"unknown blind heuristic {heuristic}")
    if policy is not None and policy.mode != ("visual" if features.visual else features.spec):
        raise ValueError("policy mode differs from the frozen feature arm")
    n, device = env.n, env.device
    env.write_curriculum(torch.tensor(UNIFORM_WEIGHTS, device=device, dtype=torch.float32))
    env.reset(seed)
    goals = features.goal_table(env.goal_images) if features.visual else None
    start_landed = env.diagnostics[:, 19].clamp(0, 1).clone()
    active = torch.ones(n, device=device)
    completed = torch.zeros(n, device=device)
    success = torch.zeros(n, device=device)
    collision = torch.zeros(n, device=device)
    out_of_bounds = torch.zeros(n, device=device)
    bucket = torch.zeros(n, device=device)
    steps = torch.zeros(n, device=device)
    returns = torch.zeros(n, device=device)
    starts = torch.ones(n, device=device)
    actions = torch.empty((n, env.action_dim), device=device)
    hidden = torch.zeros((n, policy.hidden_size), device=device) if policy is not None else None
    hover = torch.tensor(HOVER_ACTION, device=device).expand(n, -1)
    generator = torch.Generator(device=device).manual_seed(seed)
    if policy is not None:
        policy.eval()
    with gpu_range("sim.evaluate"):
        for _ in range(env.horizon):
            if policy is not None:
                cls, tokens = features.observe(env) if features.visual else (None, None)
                goal = goals[env.design.long()] if features.visual else None
                inputs = policy.actor_inputs(cls, tokens, goal, env.proprio, env.privileged)
                mean, hidden = policy.step(inputs, hidden, starts)
                actions.copy_(mean.clamp(-1, 1))
            elif heuristic == "hover":
                actions.copy_(hover)
            else:
                actions.copy_(torch.rand((n, env.action_dim), device=device, generator=generator) * 2 - 1)
            env.step(actions)
            finished = active * env.done
            returns.add_(active * env.reward)
            steps.add_(active)
            completed.add_(finished)
            success.add_(finished * env.success)
            collision.add_(finished * env.collision * (1 - env.success))
            out_of_bounds.add_(finished * env.out_of_bounds * (1 - env.success) * (1 - env.collision))
            bucket.add_(finished * env.ended_bucket)
            active.mul_(1 - env.done)
            starts.copy_(env.done)
    timeout = completed - success - collision - out_of_bounds
    rows = torch.stack((completed, success, collision, out_of_bounds, timeout,
                        steps * success, returns, returns.square(), steps))
    # Reduce per-environment first-episode records on the GPU into
    # [field, bucket, start-landed] tables; one readback follows the horizon.
    bucket_onehot = (bucket.unsqueeze(-1) == torch.arange(3, device=device, dtype=torch.float32)).float()
    bucket_onehot = bucket_onehot * completed.unsqueeze(-1)
    landed = start_landed
    landed_onehot = torch.stack((1 - landed, landed), dim=-1)
    table = torch.einsum("kn,nb,ns->kbs", rows, bucket_onehot, landed_onehot)
    unfinished = active.sum()
    env.stream.synchronize()  # Explicit evaluation end-batch boundary.
    values = table.sum((1, 2)).cpu().tolist()
    if unfinished.item() != 0 or round(values[0]) != n:
        raise RuntimeError(f"native finite horizon violated: {n - round(values[0])} environments unfinished")
    cells = table.cpu().tolist()
    return summarize(n, cells, policy=heuristic or "checkpoint", horizon=env.horizon)


def _aggregate(cells, buckets, landed):
    """Sum table cells over the selected buckets and start-landed indices."""
    return [sum(cells[k][b][s] for b in buckets for s in landed) for k in range(len(FIELDS))]


def _rates(values):
    completed, success, collision, out_of_bounds, timeout, success_steps, _, _, steps = values
    if completed == 0:
        return {"episodes": 0}
    return {"episodes": round(completed), "success": success / completed,
            "success_wilson95": wilson(success, round(completed)),
            "collision": collision / completed, "out_of_bounds": out_of_bounds / completed,
            "timeout": timeout / completed, "mean_steps": steps / completed,
            "mean_success_steps": success_steps / success if success else None}


def summarize(n, cells, policy, horizon):
    overall = _aggregate(cells, range(3), range(2))
    result = {"policy": policy, "horizon_terminal": horizon, **_rates(overall)}
    returns, squares = overall[6], overall[7]
    mean_return = returns / n
    result["mean_return"] = mean_return
    result["return_standard_error"] = (max(0.0, squares / n - mean_return * mean_return) / n) ** 0.5
    result["buckets"] = {name: _rates(_aggregate(cells, [b], range(2))) | {"bucket": name}
                         for b, name in enumerate(BUCKETS)}
    result["start_landed"] = {"flying": _rates(_aggregate(cells, range(3), [0])),
                              "landed": _rates(_aggregate(cells, range(3), [1]))}
    return result
