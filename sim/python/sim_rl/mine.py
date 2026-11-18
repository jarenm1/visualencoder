"""Mine perception failures of a frozen visual policy for encoder post-training.

The policy flies fixed catalog goals (no oracle refresh). Every ``stride`` control steps each
environment's observation is labelled from simulator ground truth:

  normal       the policy's saliency agrees with ground truth (kept at ``normal_rate``)
  wrong_peak   target visible, policy confident it is in view (P > 0.5), peak patch off the target
  false_view   target not visible, policy confident it is in view (it locked onto something else)

Each split directory holds uint8 frames, privileged labels (box, visibility), diagnostics, the
frame's catalog index, its kind and the policy's saliency summary, plus the goal-crop image of
every catalog entry used (``goal_images`` indexed by ``goal_index``). Goals are stored as images,
not embeddings, so an updated encoder re-encodes them. Mining is an offline boundary: host
copies happen once per sampled step.
"""
from __future__ import annotations

import json

import numpy as np
import torch

from .cli import (TEST_SEED, TEST_STYLE, TRAIN_STYLE, VALIDATION_SEED, VALIDATION_STYLE, build_policy,
                  emit, load_checkpoint, sha256, validate_checkpoint)
from .curriculum import fixed_lanes
from .features import checkpoint_features, crop_goals
from .model import deterministic_action, saliency_targets
from .native import CudaEnv

KINDS = ("normal", "wrong_peak", "false_view")
SPLITS = {"train": (TRAIN_STYLE, 40_001), "validation": (VALIDATION_STYLE, VALIDATION_SEED + 7),
          "test": (TEST_STYLE, TEST_SEED + 7)}


@torch.no_grad()
def goal_images(env, catalog):
    """uint8 [U,3,H,W] goal-crop images for sorted unique catalog indices. Each render_goals call
    renders n consecutive entries, so one call covers every wanted index inside its window."""
    wanted = sorted(set(catalog))
    images = {}
    position = 0
    while position < len(wanted):
        first = wanted[position]
        env.render_goals(first)
        crops = crop_goals(env.goal_views, env.goal_crops)
        while position < len(wanted) and wanted[position] < first + env.n:
            index = wanted[position]
            images[index] = (crops[index - first].clamp(0, 1) * 255).round().to(torch.uint8).cpu()
            position += 1
    return wanted, torch.stack([images[index] for index in wanted]).numpy()


@torch.no_grad()
def mine_split(args, saved, features, split):
    style, seed = SPLITS[split]
    metadata = saved["metadata"]
    out = args.out / split
    out.mkdir(parents=True, exist_ok=False)
    rows = {name: [] for name in ("images", "privileged", "diagnostics", "catalog", "kind", "in_view", "peak",
                                  "step")}
    counts = {kind: 0 for kind in KINDS}
    sampled = 0
    with CudaEnv(args.library, args.num_envs, seed, metadata["width"], metadata["height"], style,
                 metadata["control_mode"], args.city) as env, env.execution():
        validate_checkpoint(saved, env, features, for_task_transfer=args.allow_task_transfer)
        policy = build_policy(env, features, saved)
        policy.eval()
        grid_h, grid_w = env.height // 16, env.width // 16
        env.write_curriculum(fixed_lanes(env.device, appearance=args.appearance))
        table = features.catalog_goals(env, args.appearance)
        generator = torch.Generator(device=env.device).manual_seed(seed)
        for round_index in range(args.rounds):
            env.reset(seed + round_index)
            hidden = torch.zeros((env.n, policy.hidden_size), device=env.device)
            starts = torch.ones(env.n, device=env.device)
            for step in range(env.horizon * args.horizons):
                cls, tokens = features.observe(env)
                goal = table[env.catalog_index.long()].float()
                inputs, logits = policy.actor_inputs_and_features(cls, tokens, goal, env.proprio, env.privileged)
                if step % args.stride == args.stride - 1:
                    probs = logits.softmax(-1)
                    in_view = 1 - probs[:, -1]
                    peak = probs[:, :-1].argmax(-1)
                    _, visible, on_target = saliency_targets(env.privileged, grid_h, grid_w)
                    hit = on_target.gather(1, peak.unsqueeze(-1)).squeeze(-1)
                    confident = in_view > 0.5
                    kind = torch.zeros(env.n, dtype=torch.long, device=env.device)
                    kind[visible & confident & ~hit] = 1
                    kind[~visible & confident] = 2
                    keep = (kind > 0) | (torch.rand(env.n, device=env.device, generator=generator) < args.normal_rate)
                    ids = keep.nonzero().squeeze(-1)
                    sampled += env.n
                    if len(ids):
                        rows["images"].append((env.obs[ids] * 255).round().to(torch.uint8).cpu())
                        rows["privileged"].append(env.privileged[ids].cpu())
                        rows["diagnostics"].append(env.diagnostics[ids].cpu())
                        rows["catalog"].append(env.catalog_index[ids].long().cpu())
                        rows["kind"].append(kind[ids].cpu())
                        rows["in_view"].append(in_view[ids].cpu())
                        rows["peak"].append(peak[ids].cpu())
                        rows["step"].append(torch.full((len(ids),), round_index * 10_000 + step))
                mean, hidden = policy.step(inputs, hidden, starts)
                env.step(deterministic_action(mean).contiguous())
                starts.copy_(env.done)
            print(json.dumps({"split": split, "round": round_index,
                              "kept": sum(len(chunk) for chunk in rows["kind"])}), flush=True)
        arrays = {name: torch.cat(chunks).numpy() for name, chunks in rows.items()}
        wanted, goals = goal_images(env, arrays["catalog"].tolist())
    position = {index: slot for slot, index in enumerate(wanted)}
    arrays["goal_index"] = np.asarray([position[index] for index in arrays["catalog"]], dtype=np.int64)
    arrays["goal_images"] = goals
    for name, value in arrays.items():
        np.save(out / f"{name}.npy", value)
    for kind, name in enumerate(KINDS):
        counts[name] = int((arrays["kind"] == kind).sum())
    return {"split": split, "style": style, "seed": seed, "sampled_observations": sampled,
            "kept": len(arrays["kind"]), "counts": counts, "unique_goals": len(wanted)}


def run(args):
    saved = load_checkpoint(args.checkpoint, for_task_transfer=args.allow_task_transfer)
    metadata = saved["metadata"]
    features = checkpoint_features(metadata, torch.device("cuda:0"))
    args.out.mkdir(parents=True, exist_ok=True)
    splits = [mine_split(args, saved, features, split) for split in args.splits]
    payload = {"checkpoint": str(args.checkpoint), "checkpoint_sha256": sha256(args.checkpoint),
               "checkpoint_task": metadata["task"], "encoder": metadata["encoder"],
               "goal_source": "fixed catalog goal crop (no refresh)", "stride": args.stride,
               "normal_rate": args.normal_rate, "horizons": args.horizons, "rounds": args.rounds,
               "num_envs": args.num_envs, "appearance": args.appearance, "kinds": list(KINDS),
               "splits": splits}
    emit(payload, args.out / "mining.json")
    print(json.dumps(payload, indent=2), flush=True)
