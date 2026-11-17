"""Capture actual GPU policy observations and poses, then render offline replay clips."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch

from .cli import ROOT, load_checkpoint, seed_all, sha256, validate_checkpoint
from .features import FrozenFeatures
from .native import DEFAULT_CITY, CudaEnv
from .ppo import Policy


@torch.no_grad()
def capture(args, saved, seed, style):
    metadata = saved["metadata"]
    width, height, n = metadata["width"], metadata["height"], args.num_envs
    with CudaEnv(args.library, n, seed, width, height, style, args.city) as env, env.execution():
        features = FrozenFeatures(metadata["encoder"], env.device)
        validate_checkpoint(saved, env, features)
        policy = Policy(features.dim, env.action_count, metadata["ppo"]["hidden_size"]).to(env.device).eval()
        policy.load_state_dict(saved["policy"], strict=True)
        seed_all(seed)
        env.reset(seed)
        hidden = torch.zeros((n, policy.hidden_size), device=env.device)
        starts = torch.ones(n, device=env.device)
        actions = torch.empty(n, dtype=torch.int64, device=env.device)
        images = torch.empty((env.horizon, n, height, width, 3), dtype=torch.uint8, device=env.device)
        states = torch.empty((env.horizon, n, 8), device=env.device)
        transitions = torch.empty((env.horizon, n, 5), device=env.device)
        for step in range(env.horizon):
            states[step].copy_(env.diagnostics)
            images[step].copy_((env.obs.permute(0, 2, 3, 1) * 255).round().clamp(0, 255).to(torch.uint8))
            logits, _, hidden = policy(features(env), env.proprio, hidden, starts)
            actions.copy_(logits.argmax(-1))
            env.step(actions)
            transitions[step].copy_(torch.stack(
                (actions.float(), env.reward, env.done, env.success, env.collision), -1))
            starts.copy_(env.done)
        # Replay export intentionally reads back once after collection. It does
        # not alter the simulator's GPU-only training/rollout transport contract.
        env.stream.synchronize()
        return images.cpu().numpy(), states.cpu().numpy(), transitions.cpu().numpy(), dict(env.metadata)


def record(args):
    if args.num_envs < 1:
        raise ValueError("--num-envs must be positive")
    renderer = args.renderer.resolve(strict=True)
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required; use the sim Nix development shell")
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite replay: {args.out}")
    saved = load_checkpoint(args.checkpoint, args)
    metadata = saved["metadata"]
    styles = {"train": 0, "validation": metadata["validation_style"], "test": metadata["final_test_style"]}
    seeds = {"train": metadata["seed"], "validation": metadata["validation_seed"], "test": metadata["final_test_seed"]}
    seed = seeds[args.split] if args.seed is None else args.seed
    if not 0 <= seed < 2**64:
        raise ValueError("seed must fit unsigned 64 bits")
    images, states, transitions, native = capture(args, saved, seed, styles[args.split])
    torch.cuda.empty_cache()  # Release this exporter's unused memory before the renderer process.
    if not np.isfinite(states).all() or not np.isfinite(transitions).all():
        raise RuntimeError("native recording contains non-finite state or transitions")
    args.out.mkdir(parents=True)
    environments = []
    for index in range(args.num_envs):
        terminal = np.flatnonzero(transitions[:, index, 2])
        if not len(terminal):
            raise RuntimeError(f"environment {index} did not terminate within its finite horizon")
        length = int(terminal[0]) + 1
        episode = states[:length, index]
        if not np.all(episode[:, 6] == episode[0, 6]) or not np.array_equal(episode[:, 7], np.arange(length)):
            raise RuntimeError(f"environment {index} recording crosses an autoreset boundary")
        final = transitions[length - 1, index]
        outcome = "success" if final[3] else "collision" if final[4] else "timeout"
        basename = f"env-{index:03d}"
        with (args.out / f"{basename}.rgb").open("wb") as stream:
            for frame in images[:length, index]:
                stream.write(memoryview(frame))
        environments.append({
            "id": index, "route_id": int(episode[0, 6]), "frame_count": length,
            "duration_seconds": length / native["control_hz"], "outcome": outcome,
            "video": f"{basename}.mp4", "states": episode.tolist(),
            "transitions": transitions[:length, index].tolist(),
        })
    width, height = metadata["width"], metadata["height"]
    manifest = {
        "format": "sim-city-replay-v1", "title": "City PPO replay",
        "control_hz": native["control_hz"], "frame_width": width, "frame_height": height,
        "video_width": width * 2, "video_height": height,
        "views": {"first_person": [0, 0, width, height], "third_person": [width, 0, width, height]},
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": sha256(args.checkpoint),
        "native_sha256": metadata["native_sha256"], "encoder": metadata["encoder"],
        "actor_inputs": metadata["actor_inputs"], "selected_steps": saved["steps"],
        "seed": seed, "split": args.split, "city_fingerprint": native["city_fingerprint"],
        "city_path": native["city_path"], "actions": native["actions"],
        "state_fields": ["world_x", "world_y", "world_z", "goal_x", "goal_y", "goal_z", "route_id", "clock"],
        "transition_fields": ["action", "reward", "done", "success", "collision"],
        "state_semantics": "Pre-action policy/sensor poses. Terminal flag belongs to following action; no autoreset pose is appended. Last recorded pose precedes termination by one control tick.",
        "third_person_attitude": "Goal-facing visualization proxy; task has no rotor/aerodynamic attitude state.",
        "environments": environments,
    }
    staging = args.out / "recording.json"
    staging.write_text(json.dumps(manifest, allow_nan=False) + "\n")
    subprocess.run([str(renderer), "--recording", str(staging.resolve())], check=True)
    for env in environments:
        video = args.out / env["video"]
        if not video.is_file() or video.stat().st_size == 0:
            raise RuntimeError(f"renderer did not produce {video}")
    staging.replace(args.out / "manifest.json")
    for env in environments:
        (args.out / env["video"]).with_suffix(".rgb").unlink()
    print(json.dumps({"replay": str(args.out.resolve()), "environments": len(environments),
                      "outcomes": {kind: sum(e["outcome"] == kind for e in environments)
                                   for kind in ("success", "collision", "timeout")}}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "runs/city-mean-320-seed0/best.pt")
    parser.add_argument("--library", type=Path, default=ROOT / ".sim-target/training/libsim_env.so")
    parser.add_argument("--renderer", type=Path, default=ROOT / ".sim-target/training/sim-replay-render")
    parser.add_argument("--city", type=Path, default=DEFAULT_CITY)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/city-replay")
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    record(parser.parse_args())
