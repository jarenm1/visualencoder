"""Held-out evaluation: first completed episode per environment, one readback after the horizon."""
from __future__ import annotations

import torch

from .curriculum import BUCKETS, UNIFORM_WEIGHTS, fixed_lanes
from .features import RefreshedGoals
from .memory import SpatialMemory, observe as observe_memory
from .model import EXPLORE_COMMAND, GOAL_COMMAND, deterministic_action
from .native import HOVER_ACTIONS
from .ppo import gpu_range
from .stats import wilson

GPS_SPEED_GAIN = 0.5             # 1/s, proportional horizontal and vertical gain on GPS error (metres)
GPS_MAX_HORIZONTAL = 4.0         # m/s
GPS_MAX_VERTICAL = 2.0           # m/s
GPS_MAX_YAW_RATE = 1.5           # rad/s
# proprio lanes 14..16 hold the target hint (right, up, -forward) in metres / 32; lane 18 is hint available.
GPS_HINT_SCALE = 32.0
GPS_LANE = 18
FIELDS = ("completed", "success", "collision", "out_of_bounds", "timeout", "success_steps",
          "return", "return_squared", "steps")


def heuristics_for(control_mode):
    """Blind hover and random baselines in both modes; the GPS pursuit needs velocity commands."""
    return ("hover", "random", "gps") if control_mode == "velocity" else ("hover", "random")


def gps_pursuit(proprio, action_scales):
    """Velocity-mode GPS pursuit: float32 [n, 4] actions from the proprio GPS hint.

    Commands are proportional to the GPS error and capped; yaw turns toward the target
    bearing. Environments without GPS receive a zero (hover) command. Commands are divided by
    the native action scales, so actions are in [-1, 1] before clamping.
    """
    right = proprio[:, 14] * GPS_HINT_SCALE
    up = proprio[:, 15] * GPS_HINT_SCALE
    forward = -proprio[:, 16] * GPS_HINT_SCALE
    horizontal = torch.stack((forward, right), dim=-1) * GPS_SPEED_GAIN
    speed = horizontal.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    horizontal = horizontal * (speed.clamp(max=GPS_MAX_HORIZONTAL) / speed)
    vertical = (up * GPS_SPEED_GAIN).clamp(-GPS_MAX_VERTICAL, GPS_MAX_VERTICAL)
    bearing = torch.atan2(right, forward)
    yaw_rate = (-bearing).clamp(-GPS_MAX_YAW_RATE, GPS_MAX_YAW_RATE)
    command = torch.stack((horizontal[:, 0], horizontal[:, 1], vertical, yaw_rate), dim=-1)
    scale = torch.tensor(action_scales, device=proprio.device, dtype=torch.float32)
    actions = (command / scale).clamp(-1, 1)
    available = proprio[:, GPS_LANE:GPS_LANE + 1]
    return actions * available


@torch.no_grad()
def evaluate(env, features, policy=None, heuristic=None, seed=0, no_gps=0.0, appearance=1.0,
             goal_refresh_steps=0, bucket_weights=UNIFORM_WEIGHTS, coarse=0.0, wrong=0.0, entries=None):
    """Deterministic policy (tanh of the mean) or blind heuristic over one reset of the native finite horizon.

    Curriculum is uniform over visibility buckets with p_no_gps = ``no_gps`` (episodes without a
    target hint), hinted episodes coarse / wrong with probabilities ``coarse`` / ``wrong`` (precise
    otherwise), and the given appearance strength. ``entries`` (int32 [n] catalog indices, -1 to
    sample) fixes each environment's first episode (the hard set). Outcomes are exclusive
    (success > collision > out_of_bounds > timeout) and recorded only for each environment's first
    completed episode.
    """
    if (policy is None) == (heuristic is None):
        raise ValueError("provide exactly one trained policy or blind heuristic")
    if heuristic is not None and heuristic not in heuristics_for(env.control_mode):
        raise ValueError(f"blind heuristic {heuristic} is unavailable in {env.control_mode} mode")
    if policy is not None and policy.mode != ("visual" if features.visual else features.spec):
        raise ValueError("policy mode differs from the frozen feature arm")
    n, device = env.n, env.device
    if len(bucket_weights) != 3 or min(bucket_weights) < 0 or sum(bucket_weights) <= 0:
        raise ValueError("bucket_weights must contain three nonnegative weights with positive sum")
    env.write_curriculum(fixed_lanes(device, tuple(bucket_weights), no_gps, appearance, coarse, wrong))
    goals = features.catalog_goals(env, appearance) if policy is not None and features.visual else None
    if entries is not None:
        env.next_catalog.copy_(entries)
    env.reset(seed)
    goal_source = RefreshedGoals(env, features, goals, goal_refresh_steps) if goals is not None else None
    start_landed = env.diagnostics[:, 19].clamp(0, 1).clone()
    start_class = env.diagnostics[:, 14].clone()
    initial_coverage = env.episode_diagnostics[:, 0].clone()
    initial_distance = env.diagnostics[:, 20].clone()
    observed_tiles = torch.zeros(n, device=device)
    stationary_away = torch.zeros(n, device=device)
    arrived = torch.zeros(n, device=device)
    first_arrival_steps = torch.zeros(n, device=device)
    post_arrival_steps = torch.zeros(n, device=device)
    post_arrival_holding = torch.zeros(n, device=device)
    lost_hold = torch.zeros(n, device=device)
    hold_steps = env.metadata["success"]["hold_steps"]
    arrival_steps = env.metadata["success"]["arrival_steps"]
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
    spatial = (SpatialMemory(n, env.height // 16, env.width // 16, env.metadata["camera"], device)
               if policy is not None and policy.spatial else None)
    hover = torch.tensor(HOVER_ACTIONS[env.control_mode], device=device).expand(n, -1)
    goal_command = torch.full((n,), GOAL_COMMAND, dtype=torch.long, device=device)
    generator = torch.Generator(device=device).manual_seed(seed)
    if policy is not None:
        policy.eval()
    with gpu_range("sim.evaluate"):
        for _ in range(env.horizon):
            speed = env.diagnostics[:, 10:13].square().sum(-1).sqrt()
            stationary_away.add_(active * ((speed < 0.2) & (env.privileged[:, 3] * 32 > 6)).float())
            if policy is not None:
                cls, tokens = features.observe(env) if features.visual else (None, None)
                goal = goal_source.step(starts) if goal_source is not None else None
                memory = (observe_memory(policy, spatial, cls, tokens, goal, env.proprio, starts)
                          if spatial is not None else None)
                inputs, _ = policy.actor_inputs_and_features(cls, tokens, goal, env.proprio, env.privileged, memory,
                                                             goal_command)
                mean, hidden = policy.step(inputs, hidden, starts)
                actions.copy_(deterministic_action(mean))
            elif heuristic == "hover":
                actions.copy_(hover)
            elif heuristic == "gps":
                actions.copy_(gps_pursuit(env.proprio, env.action_scales))
            else:
                actions.copy_(torch.rand((n, env.action_dim), device=device, generator=generator) * 2 - 1)
            env.step(actions)
            finished = active * env.done
            returns.add_(active * env.reward)
            steps.add_(active)
            # Terminal hold survives native autoreset; include arrivals on the final action.
            hold = torch.where(env.done.bool(), env.episode_diagnostics[:, 2],
                               env.privileged[:, 24] * hold_steps)
            holding = (hold > 0).float()
            reached = active * (hold >= arrival_steps - 0.5).float()  # Normalized integer counter.
            first_arrival_steps.add_(reached * (1 - arrived) * steps)
            torch.maximum(arrived, reached, out=arrived)
            post_arrival_steps.add_(active * arrived)
            post_arrival_holding.add_(active * arrived * holding)
            torch.maximum(lost_hold, active * arrived * (1 - holding), out=lost_hold)
            completed.add_(finished)
            success.add_(finished * env.success)
            collision.add_(finished * env.collision * (1 - env.success))
            out_of_bounds.add_(finished * env.out_of_bounds * (1 - env.success) * (1 - env.collision))
            bucket.add_(finished * env.ended_bucket)
            observed_tiles.add_(finished * env.episode_diagnostics[:, 1])
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
    layout = torch.tensor([env.layout_classes.index(name) for name in env.target_classes], device=device,
                          dtype=torch.float32)
    class_table = torch.stack((completed, success)) @ (start_class.unsqueeze(-1) == layout).float()
    unfinished = active.sum()
    env.stream.synchronize()  # Explicit evaluation end-batch boundary.
    values = table.sum((1, 2)).cpu().tolist()
    if unfinished.item() != 0 or round(values[0]) != n:
        raise RuntimeError(f"native finite horizon violated: {n - round(values[0])} environments unfinished")
    cells = table.cpu().tolist()
    result = summarize(n, cells, policy=heuristic or "checkpoint", horizon=env.horizon, no_gps=no_gps)
    result["appearance"] = appearance
    result["goal_refresh_steps"] = goal_refresh_steps if goals is not None else 0
    result["goal_box_source"] = "simulator" if goals is not None and goal_refresh_steps else None
    result["bucket_weights"] = list(bucket_weights)
    result["start_box_distance_m"] = {"min": float(initial_distance.min()), "max": float(initial_distance.max()),
                                    "mean": float(initial_distance.mean())}
    result["mean_observed_tiles"] = float(observed_tiles.mean())
    result["mean_new_observed_tiles"] = float((observed_tiles - initial_coverage).mean())
    result["stationary_away_fraction"] = float(stationary_away.sum() / steps.sum().clamp_min(1))
    arrival_count = float(arrived.sum())
    result["arrival_rate"] = arrival_count / n
    result["mean_first_arrival_seconds"] = (float(first_arrival_steps.sum()) / arrival_count / env.control_hz
                                            if arrival_count else None)
    result["post_arrival_hold_fraction"] = (float(post_arrival_holding.sum() / post_arrival_steps.sum())
                                            if arrival_count else None)
    result["hold_loss_after_arrival_rate"] = float(lost_hold.sum()) / arrival_count if arrival_count else None
    result["arrival_to_success_rate"] = float(success.sum()) / arrival_count if arrival_count else None
    result["classes"] = {name: {"episodes": round(count), "success": hits / count if count else None}
                         for name, count, hits in zip(env.target_classes, *class_table.cpu().tolist())}
    return result


@torch.no_grad()
def evaluate_explore(env, features, policy, seed=0, appearance=1.0, no_gps=1.0):
    """Deterministic policy under the explore command (no goal) for one reset of the horizon: surface
    tiles observed by each environment's first episode (native coverage), and how it ended."""
    n, device = env.n, env.device
    env.write_curriculum(fixed_lanes(device, no_gps=no_gps, appearance=appearance))
    env.reset(seed)
    spatial = (SpatialMemory(n, env.height // 16, env.width // 16, env.metadata["camera"], device)
               if policy.spatial else None)
    command = torch.full((n,), EXPLORE_COMMAND, dtype=torch.long, device=device)
    goal = torch.zeros((n, features.goal_dim), device=device) if features.visual else None
    no_goal = torch.zeros(n, device=device)
    starts, active = torch.ones(n, device=device), torch.ones(n, device=device)
    hidden = torch.zeros((n, policy.hidden_size), device=device)
    tiles, crashed, left, steps = (torch.zeros(n, device=device) for _ in range(4))
    policy.eval()
    with gpu_range("sim.evaluate_explore"):
        for _ in range(env.horizon):
            cls, tokens = features.observe(env) if features.visual else (None, None)
            memory = (observe_memory(policy, spatial, cls, tokens, goal, env.proprio, starts, no_goal)
                      if spatial is not None else None)
            inputs, _ = policy.actor_inputs_and_features(cls, tokens, goal, env.proprio, env.privileged, memory, command)
            mean, hidden = policy.step(inputs, hidden, starts)
            env.step(deterministic_action(mean).contiguous())
            finished = active * env.done
            steps.add_(active)
            tiles.add_(finished * env.episode_diagnostics[:, 1])
            crashed.add_(finished * env.collision)
            left.add_(finished * env.out_of_bounds * (1 - env.collision))
            active.mul_(1 - env.done)
            starts.copy_(env.done)
    env.stream.synchronize()  # Explicit evaluation end-batch boundary.
    return {"episodes": n, "mean_observed_tiles": float(tiles.mean()), "collision": float(crashed.mean()),
            "out_of_bounds": float(left.mean()), "mean_steps": float(steps.mean()), "no_gps_fraction": no_gps,
            "appearance": appearance}


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


def summarize(n, cells, policy, horizon, no_gps):
    overall = _aggregate(cells, range(3), range(2))
    result = {"policy": policy, "horizon_terminal": horizon, "no_gps_fraction": no_gps, **_rates(overall)}
    returns, squares = overall[6], overall[7]
    mean_return = returns / n
    result["mean_return"] = mean_return
    result["return_standard_error"] = (max(0.0, squares / n - mean_return * mean_return) / n) ** 0.5
    result["buckets"] = {name: _rates(_aggregate(cells, [b], range(2))) | {"bucket": name}
                         for b, name in enumerate(BUCKETS)}
    result["start_landed"] = {"flying": _rates(_aggregate(cells, range(3), [0])),
                              "landed": _rates(_aggregate(cells, range(3), [1]))}
    result["bucket_start"] = {name: {"flying": _rates(_aggregate(cells, [b], [0])),
                                     "landed": _rates(_aggregate(cells, [b], [1]))}
                              for b, name in enumerate(BUCKETS)}
    return result
