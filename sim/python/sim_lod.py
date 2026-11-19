"""Level-of-detail sweep: throughput and fidelity against exact rendering.

Each budget sets SIM_LOD_PIXELS (read at environment creation), then
  * profiles the step at --envs (sim_profile.profile: throughput, stage times, triangles), and
  * replays one fixed 64-env trajectory (physics does not depend on rendering, so states match
    exactly) and compares observations with exact rendering (budget 0): pixel error, patch
    depth / object labels, and the frozen encoder's CLS and patch-token cosine similarity.

    python sim/python/sim_lod.py --budgets 0.5 1 2 4 --out reports/sim-lod.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from sim_profile import profile
from sim_rl.cli import DEFAULT_LIBRARY
from sim_rl.features import FrozenFeatures
from sim_rl.native import CudaEnv

FIDELITY_ENVS, FIDELITY_STEPS, SHOT_EVERY = 64, 90, 15


def apply(lod):
    os.environ["SIM_LOD_PIXELS"] = str(lod)


def trajectory(width, height, encoder):
    """Observations, patch labels and encoder features along a fixed trajectory."""
    started = time.perf_counter()
    with CudaEnv(DEFAULT_LIBRARY, FIDELITY_ENVS, 11, width, height, 0, "velocity",
                 start_distance=(8.0, 60.0)) as env, env.execution():
        created = time.perf_counter() - started
        generator = torch.Generator(device=env.device).manual_seed(5)
        actions = torch.empty(FIDELITY_ENVS, env.action_dim, device=env.device)
        shots = []
        for step in range(FIDELITY_STEPS):
            actions.uniform_(-0.6, 0.6, generator=generator)
            env.step(actions)
            if step % SHOT_EVERY == SHOT_EVERY - 1:
                cls, tokens = encoder.encode(env.obs)
                shots.append({"obs": env.obs.clone(), "depth": env.patch_depth.clone(),
                              "object": env.patch_object.clone(), "state": env.diagnostics.clone(),
                              "cls": cls.float(), "tokens": tokens.float()})
    return shots, created


def fidelity(shots, reference):
    rows = []
    for shot, exact in zip(shots, reference):
        if not torch.equal(shot["state"], exact["state"]):
            raise RuntimeError("trajectories diverged: physics must not depend on rendering")
        error = (shot["obs"] - exact["obs"]).abs().amax(1)          # [n, H, W] worst channel
        rows.append({
            "pixel_mae": error.mean().item(),
            "pixels_over_0.1": (error > 0.1).float().mean().item(),
            "patch_depth_mae_m": (shot["depth"] - exact["depth"]).abs().mean().item(),
            "patch_object_agreement": (shot["object"] == exact["object"]).float().mean().item(),
            "cls_cosine": F.cosine_similarity(shot["cls"], exact["cls"], dim=-1).mean().item(),
            "patch_cosine": F.cosine_similarity(shot["tokens"], exact["tokens"], dim=-1).mean().item(),
            "patch_cosine_p01": torch.quantile(F.cosine_similarity(shot["tokens"], exact["tokens"], dim=-1)
                                               .flatten(), 0.01).item(),
        })
    return {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--budgets", type=float, nargs="+", required=True,
                        help="level-of-detail budgets in pixels; 0 (exact) is always measured first")
    parser.add_argument("--envs", type=int, default=1024)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--encoder", default="mean")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    encoder = FrozenFeatures(args.encoder, torch.device("cuda:0"), "fp16", "patch")
    budgets = [0.0] + [budget for budget in args.budgets if budget != 0.0]
    rows, reference = [], None
    for lod in budgets:
        apply(lod)
        shots, created = trajectory(args.width, args.height, encoder)
        reference = reference or shots
        row = {"lod_pixels": lod, "startup_s": round(created, 2),
               **profile(args.envs, args.width, args.height, args.steps, 0), "fidelity": fidelity(shots, reference)}
        rows.append(row)
        f, r = row["fidelity"], row["render"]
        print(f"lod {lod:g}px: {row['steps_per_second']} steps/s | cameras "
              f"{row['stages_ms']['vk_camera_passes_ms']:.1f} ms gpu {row['stages_ms']['gpu_step_ms']:.1f} ms | "
              f"visible tris {r['visible_triangles_per_camera']:.0f} draws {r['visible_draws_per_camera']:.0f} | "
              f"pixel MAE {f['pixel_mae']:.4f} >0.1 {f['pixels_over_0.1']:.4f} | depth {f['patch_depth_mae_m']:.3f} m "
              f"objects {f['patch_object_agreement']:.4f} | cls cos {f['cls_cosine']:.4f} patch cos "
              f"{f['patch_cosine']:.4f} (p01 {f['patch_cosine_p01']:.3f}) | startup {created:.1f} s", flush=True)
        torch.cuda.empty_cache()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"device": torch.cuda.get_device_name(0), "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
