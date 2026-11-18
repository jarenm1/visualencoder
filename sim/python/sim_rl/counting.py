"""Count benchmark: does the onboard perception + memory count the cars a flight has seen?

Each environment flies exploratory velocity commands (the same Ornstein-Uhlenbeck flights the
perception heads were trained on, so this measures perception and memory, not search) for
``--steps`` control steps. Every step, the frozen encoder and perception head classify and range
each patch; patches called "vehicle" go to ``tracks.ObjectTracks`` at their projected ground points.

Ground truth: the distinct layout vehicles that were the labelled object of at least ``TRUE_PATCHES``
patches in at least ``CONFIRM_HITS`` frames (what was actually seen well enough to count). Reports
the per-flight count error, exact-count rate and the estimate/true ratio, overall and by true count.
"""
from __future__ import annotations

import json

import torch

from .curriculum import UNIFORM_WEIGHTS
from .memory import SpatialMemory
from .native import PATCH, CudaEnv
from .perception import CLASSES, OU_BIAS, OU_SIGMA, OU_THETA, ground_prior, load_heads, object_classes
from .tracks import CLASS_THRESHOLD, CONFIRM_HITS, ObjectTracks

TRUE_PATCHES = 2
VEHICLE = CLASSES.index("vehicle")


@torch.no_grad()
def count_flights(env, features, heads, steps, seed, appearance):
    """(estimated counts [n], true counts [n]) after one exploratory flight per environment."""
    n, device = env.n, env.device
    lookup = object_classes(env)
    env.write_curriculum(torch.tensor(UNIFORM_WEIGHTS + (0.0, appearance), device=device, dtype=torch.float32))
    env.reset(seed)
    memory = SpatialMemory(n, env.height // PATCH, env.width // PATCH, env.metadata["camera"], device)
    tracks = ObjectTracks(n, device)
    starts = torch.ones(n, device=device)
    seen_frames = torch.zeros((n, lookup.shape[0]), device=device)
    alive = torch.ones(n, dtype=torch.bool, device=device)
    generator = torch.Generator(device=device).manual_seed(seed)
    bias = torch.tensor(OU_BIAS, device=device)
    actions = bias.expand(n, -1).clone()
    for _ in range(steps):
        memory.reset(starts)
        _, tokens = features.observe(env)
        depth, probs = heads.predict(tokens, ground_prior(env.proprio, memory.rays))
        memory.update(env.proprio, depth, torch.zeros_like(depth))
        members = (probs[..., VEHICLE] >= CLASS_THRESHOLD) & (depth > 0) & alive[:, None]
        tracks.update(memory.points, (memory.points - memory.position[:, None]).norm(dim=-1), members)
        objects = env.patch_object.long()
        vehicle = (objects >= 0) & (lookup[objects.clamp_min(0)] == VEHICLE)
        hits = torch.zeros_like(seen_frames).scatter_add_(1, objects.clamp_min(0), vehicle.float())
        seen_frames.add_(((hits >= TRUE_PATCHES) & alive[:, None]).float())
        noise = torch.randn((n, 4), device=device, generator=generator)
        actions.add_(OU_THETA * (bias - actions) + OU_SIGMA * noise).clamp_(-1, 1)
        starts.zero_()
        env.step(actions)
        # A flight ends at its first crash or exit; the count covers that flight only.
        alive &= env.done < 0.5
    return tracks.count(), (seen_frames >= CONFIRM_HITS).sum(-1)


def count_eval_command(args, started):
    from .cli import TEST_SEED, TEST_STYLE, emit
    from .features import FrozenFeatures
    device = torch.device("cuda:0")
    features = FrozenFeatures(args.encoder, device, args.encoder_precision)
    heads = load_heads(args.heads, features, device)
    with CudaEnv(args.library, args.num_envs, TEST_SEED, args.width, args.height, TEST_STYLE, "velocity", args.city,
                 start_distance=tuple(args.start_distance)) as env, env.execution():
        rows = [count_flights(env, features, heads, args.steps, TEST_SEED + r, args.appearance) for r in range(args.rounds)]
    estimate = torch.cat([r[0] for r in rows]).float().cpu()
    truth = torch.cat([r[1] for r in rows]).float().cpu()
    error = estimate - truth
    by_truth = {}
    for low, high, label in ((0, 0, "0"), (1, 2, "1-2"), (3, 5, "3-5"), (6, 1000, "6+")):
        rows_in = (truth >= low) & (truth <= high)
        if rows_in.any():
            by_truth[label] = {"flights": int(rows_in.sum()), "mean_true": float(truth[rows_in].mean()),
                               "mean_estimate": float(estimate[rows_in].mean()),
                               "mean_abs_error": float(error[rows_in].abs().mean()),
                               "exact": float((error[rows_in] == 0).float().mean())}
    result = {"encoder": features.spec, "heads": str(args.heads), "steps": args.steps,
              "flights": int(truth.numel()), "true_mean": float(truth.mean()), "estimate_mean": float(estimate.mean()),
              "mean_abs_error": float(error.abs().mean()), "mean_signed_error": float(error.mean()),
              "exact": float((error == 0).float().mean()), "within_one": float((error.abs() <= 1).float().mean()),
              "by_true_count": by_truth}
    print(json.dumps(result), flush=True)
    emit(result, args.out)
