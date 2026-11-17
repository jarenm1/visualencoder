"""Capture policy-consumed frames, poses, and encoder maps on GPU, then render replay clips offline."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch

from .cli import (DEFAULT_LIBRARY, ROOT, TEST_SEED, TEST_STYLE, TRAIN_STYLE, VALIDATION_SEED,
                  VALIDATION_STYLE, build_policy, load_checkpoint, sha256, validate_checkpoint)
from .evaluation import UNIFORM_WEIGHTS
from .features import FrozenFeatures
from .native import DEFAULT_CITY, CudaEnv

# Diagnostics indices (ABI v3, width 24) recorded per frame, pre-action.
STATE_INDEX = (0, 1, 2, 6, 7, 8, 9, 3, 4, 5, 14, 17, 18)
STATE_FIELDS = ["world_x", "world_y", "world_z", "quat_x", "quat_y", "quat_z", "quat_w",
                "target_x", "target_y", "target_z", "design", "clock", "landed"]
TRANSITION_FIELDS = ["throttle", "roll", "pitch", "yaw", "reward", "done", "success", "collision", "out_of_bounds"]
START_LANDED_INDEX = 19
PCA_SAMPLE = 65536
TORQUE_NOTE = "Goal-facing visualization proxy; the task has no rotor/aerodynamic attitude state."


@torch.no_grad()
def encoder_maps(tokens, valid):
    """tokens bf16 [T,N,P,D] -> uint8 [T,N,h,w,3].

    One shared 3-component PCA basis fit on valid (pre-terminal) frames, scaled by the
    1st/99th percentile of the valid projections. Patch grid h x w is read from the tokens' P axis.
    """
    device = tokens.device
    generator = torch.Generator(device=device).manual_seed(0)
    rows = tokens[valid].reshape(-1, tokens.shape[-1]).float()
    if rows.shape[0] > PCA_SAMPLE:
        rows = rows[torch.randperm(rows.shape[0], device=device, generator=generator)[:PCA_SAMPLE]]
    mean = rows.mean(0)
    _, _, principal = torch.linalg.svd(rows - mean, full_matrices=False)
    axes = principal[:3].T
    projected = torch.stack([(tokens[t].float() - mean) @ axes for t in range(tokens.shape[0])])
    sample = projected[valid]
    if sample.shape[0] > PCA_SAMPLE:
        sample = sample[torch.randperm(sample.shape[0], device=device, generator=generator)[:PCA_SAMPLE]]
    low, high = torch.quantile(sample, 0.01, dim=0), torch.quantile(sample, 0.99, dim=0)
    scaled = ((projected - low) / (high - low).clamp_min(1e-6)).clamp(0, 1)
    return (scaled * 255).round().to(torch.uint8)


@torch.no_grad()
def capture(args, saved, features, seed, style):
    """Roll out the deterministic policy (clamped mean) over the full horizon with uniform curriculum."""
    metadata = saved["metadata"]
    n = args.num_envs
    with CudaEnv(args.library, n, seed, metadata["width"], metadata["height"], style, args.city) as env:
        with env.execution():
            validate_checkpoint(saved, env, features)
            policy = build_policy(env, features)
            policy.load_state_dict(saved["model"])
            policy.eval()
            device, horizon = env.device, env.horizon
            h, w = env.height, env.width
            patch_h, patch_w = h // 16, w // 16
            goals = features.goal_table(env.goal_images) if features.visual else None
            env.write_curriculum(torch.tensor(UNIFORM_WEIGHTS, device=device, dtype=torch.float32))
            env.reset(seed)
            start_landed = env.diagnostics[:, START_LANDED_INDEX].clone()
            index = torch.tensor(STATE_INDEX, device=device)
            images = torch.empty((horizon, n, h, w, 3), dtype=torch.uint8, device=device)
            states = torch.empty((horizon, n, len(STATE_INDEX)), device=device)
            transitions = torch.empty((horizon, n, len(TRANSITION_FIELDS)), device=device)
            ended = torch.empty((horizon, n), device=device)
            tokens = (torch.empty((horizon, n, patch_h * patch_w, features.dim), dtype=torch.bfloat16, device=device)
                      if features.visual else None)
            starts = torch.ones(n, device=device)
            hidden = torch.zeros((n, policy.hidden_size), device=device)
            for t in range(horizon):
                images[t].copy_((env.obs.clamp(0, 1) * 255).round().to(torch.uint8).permute(0, 2, 3, 1))
                states[t].copy_(env.diagnostics.index_select(1, index))
                if features.visual:
                    cls, patch_tokens = features.observe(env)
                    tokens[t].copy_(patch_tokens)
                    goal = goals[env.design.long()]
                else:
                    cls, patch_tokens, goal = None, None, None
                inputs = policy.actor_inputs(cls, patch_tokens, goal, env.proprio, env.privileged)
                mean, hidden = policy.step(inputs, hidden, starts)
                applied = mean.clamp(-1, 1).contiguous()
                env.step(applied)
                transitions[t, :, :4].copy_(applied)
                transitions[t, :, 4].copy_(env.reward)
                transitions[t, :, 5].copy_(env.done)
                transitions[t, :, 6].copy_(env.success)
                transitions[t, :, 7].copy_(env.collision)
                transitions[t, :, 8].copy_(env.out_of_bounds)
                ended[t].copy_(env.ended_bucket)
                starts.copy_(env.done)
            done = transitions[:, :, 5]
            # Frame t is pre-terminal when no episode ended strictly before it.
            valid = (torch.cumsum(done, 0) - done) == 0
            maps = encoder_maps(tokens, valid).cpu() if features.visual else None
            env.stream.synchronize()  # Explicit export boundary: one readback of the recording.
            return {"images": images.cpu().numpy(), "states": states.cpu().numpy(),
                    "transitions": transitions.cpu().numpy(), "ended": ended.cpu().numpy(),
                    "start_landed": start_landed.cpu().numpy(), "maps": None if maps is None else maps.numpy(),
                    "native": {"control_hz": env.control_hz, "city_path": str(env.city_path),
                               "city_fingerprint": env.metadata["city_fingerprint"]}}


def record(args):
    if args.num_envs < 1:
        raise ValueError("--num-envs must be positive")
    renderer = args.renderer.resolve(strict=True)
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required; use the sim Nix development shell")
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite replay: {args.out}")
    saved = load_checkpoint(args.checkpoint)
    metadata = saved["metadata"]
    features = FrozenFeatures(metadata["encoder"], torch.device("cuda:0"))
    styles = {"train": TRAIN_STYLE, "validation": VALIDATION_STYLE, "test": TEST_STYLE}
    seeds = {"train": 0, "validation": VALIDATION_SEED, "test": TEST_SEED}
    seed = seeds[args.split] if args.seed is None else args.seed
    if not 0 <= seed < 2**64:
        raise ValueError("seed must fit unsigned 64 bits")
    recording = capture(args, saved, features, seed, styles[args.split])
    torch.cuda.empty_cache()  # Release this exporter's unused memory before the renderer process.
    states, transitions = recording["states"], recording["transitions"]
    if not np.isfinite(states).all() or not np.isfinite(transitions).all():
        raise RuntimeError("native recording contains non-finite state or transitions")
    args.out.mkdir(parents=True)
    environments = []
    for index in range(args.num_envs):
        terminal = np.flatnonzero(transitions[:, index, 5])
        if not len(terminal):
            raise RuntimeError(f"environment {index} did not terminate within its finite horizon")
        length = int(terminal[0]) + 1
        episode = states[:length, index]
        if not np.all(episode[:, 10] == episode[0, 10]) or not np.array_equal(episode[:, 11], np.arange(length)):
            raise RuntimeError(f"environment {index} recording crosses an autoreset boundary")
        final = transitions[length - 1, index]
        outcome = ("success" if final[6] else "collision" if final[7]
                   else "out_of_bounds" if final[8] else "timeout")
        basename = f"env-{index:03d}"
        with (args.out / f"{basename}.rgb").open("wb") as stream:
            for frame in recording["images"][:length, index]:
                stream.write(memoryview(np.ascontiguousarray(frame)))
        entry = {
            "id": index, "frame_count": length, "duration_seconds": length / recording["native"]["control_hz"],
            "outcome": outcome, "bucket": int(recording["ended"][length - 1, index]),
            "start_landed": bool(recording["start_landed"][index] > 0.5), "design": int(episode[0, 10]),
            "video": f"{basename}.mp4", "states": episode.tolist(),
            "transitions": transitions[:length, index].tolist(),
        }
        if recording["maps"] is not None:
            map_path = args.out / f"{basename}.encoder"
            np.ascontiguousarray(recording["maps"][:length, index]).tofile(map_path)
            entry["encoder_map"] = map_path.name
        environments.append(entry)
    width, height = metadata["width"], metadata["height"]
    manifest = {
        "format": "sim-city-replay-v2", "title": "City drone goal replay", "style": styles[args.split],
        "split": args.split, "control_hz": recording["native"]["control_hz"],
        "frame_width": width, "frame_height": height, "video_width": width * 2, "video_height": height,
        "views": {"first_person": [0, 0, width, height], "third_person": [width, 0, width, height]},
        "encoder_map": None if recording["maps"] is None else {
            "width": width // 16, "height": height // 16, "channels": 3, "dtype": "uint8",
            "method": "Unpooled patch tokens of the frozen encoder, same pass the policy consumed; shared "
                      "3-component PCA over all pre-terminal frames, 1st/99th percentile scaled."},
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": sha256(args.checkpoint),
        "native_sha256": metadata["native_library_sha256"], "encoder": metadata["encoder"],
        "actor_inputs": metadata["actor_inputs"], "selected_steps": metadata["steps"],
        "seed": seed, "city_fingerprint": recording["native"]["city_fingerprint"],
        "city_path": recording["native"]["city_path"],
        "state_fields": STATE_FIELDS, "transition_fields": TRANSITION_FIELDS,
        "state_semantics": "Pre-action poses and diagnostics. Terminal flag belongs to the action applied in the same "
                           "record; no autoreset pose is appended.",
        "third_person_attitude": TORQUE_NOTE,
        "environments": environments,
    }
    staging = args.out / "recording.json"
    staging.write_text(json.dumps(manifest, allow_nan=False) + "\n")
    subprocess.run([str(renderer), "--recording", str(staging.resolve())], check=True)
    for env_entry in environments:
        video = args.out / env_entry["video"]
        if not video.is_file() or video.stat().st_size == 0:
            raise RuntimeError(f"renderer did not produce {video}")
    staging.replace(args.out / "manifest.json")
    for env_entry in environments:
        (args.out / env_entry["video"]).with_suffix(".rgb").unlink()
    print(json.dumps({"replay": str(args.out.resolve()), "environments": len(environments),
                      "outcomes": {kind: sum(e["outcome"] == kind for e in environments)
                                   for kind in ("success", "collision", "out_of_bounds", "timeout")}}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--renderer", type=Path, default=ROOT / ".sim-target/training/sim-replay-render")
    parser.add_argument("--city", type=Path, default=DEFAULT_CITY)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/city-drone-replay")
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    record(parser.parse_args())
