"""Native simulator step profile: per-stage GPU time, unprofiled throughput and culling statistics.

For each (env count, resolution): steps with random velocity commands, measures plain throughput
(no profiler, one sync at the end), then the per-stage breakdown from the native profiler (which
waits after every step), then the culling statistics of the last render. Writes JSON; prints a table.

    python sim/python/sim_profile.py --envs 64 256 1024 --sizes 320x240 128x96 --out reports/sim-profile.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from sim_rl.cli import DEFAULT_LIBRARY
from sim_rl.native import CudaEnv


def size(text):
    width, height = (int(part) for part in text.lower().split("x"))
    return width, height


def profile(n, width, height, steps, seed):
    with CudaEnv(DEFAULT_LIBRARY, n, seed, width, height, 0, "velocity", start_distance=(8.0, 60.0)) as env, \
            env.execution():
        generator = torch.Generator(device=env.device).manual_seed(seed)
        actions = torch.empty(n, env.action_dim, device=env.device)

        def step():
            actions.uniform_(-0.6, 0.6, generator=generator)
            env.step(actions)

        for _ in range(10):
            step()
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(steps):
            step()
        torch.cuda.synchronize()
        throughput = n * steps / (time.perf_counter() - started)
        env.set_profiling(True)
        for _ in range(steps):
            step()
        stages = env.profile()
        env.set_profiling(False)
        stats = env.render_stats()
    return {"envs": n, "width": width, "height": height, "steps_per_second": round(throughput),
            "stages_ms": {key: round(value, 4) for key, value in stages.items() if key != "steps"},
            "profiled_steps": int(stages["steps"]), "render": stats}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--envs", type=int, nargs="+", default=[64, 256, 1024])
    parser.add_argument("--sizes", type=size, nargs="+", default=[(320, 240), (128, 96)])
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for width, height in args.sizes:
        for n in args.envs:
            row = profile(n, width, height, args.steps, args.seed)
            rows.append(row)
            s, r = row["stages_ms"], row["render"]
            print(f"{width}x{height} n={n}: {row['steps_per_second']} steps/s | gpu {s['gpu_step_ms']:.2f} ms = "
                  f"control {s['control_ms']:.2f} finish {s['finish_ms']:.2f} observe {s['observe_ms']:.2f} "
                  f"render+handoff {s['render_handoff_ms']:.2f} (cull {s['vk_cull_ms']:.2f} cameras "
                  f"{s['vk_camera_passes_ms']:.2f} pack {s['vk_pack_ms']:.2f} gaps {s['vk_gaps_ms']:.2f} "
                  f"handoff {s['handoff_ms']:.2f}) appearance {s['appearance_ms']:.2f} labels {s['labels_ms']:.2f} "
                  f"coverage {s['coverage_ms']:.2f} | host enqueue {s['host_enqueue_ms']:.2f} wall {s['wall_ms']:.2f} | "
                  f"draws {r['draws_per_camera']:.0f} visible {r['visible_draws_per_camera']:.1f}, triangles "
                  f"{r['scene_triangles']:.0f} visible {r['visible_triangles_per_camera']:.0f}", flush=True)
            torch.cuda.empty_cache()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"device": torch.cuda.get_device_name(0), "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
