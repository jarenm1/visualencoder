"""Generate paper tables from completed native GPU navigation experiments only."""

import argparse
import json
import math
from pathlib import Path


def wilson(successes: float, count: int) -> tuple[float, float]:
    z = 1.959963984540054
    proportion = successes / count
    denominator = 1 + z * z / count
    center = (proportion + z * z / (2 * count)) / denominator
    half = z * math.sqrt(proportion * (1 - proportion) / count + z * z / (4 * count * count)) / denominator
    return center - half, center + half


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("report/navigation.typ"))
    args = parser.parse_args()
    rows = []
    measurements = []
    for run in args.runs:
        result = json.loads((run / "summary.json").read_text())
        metadata = json.loads((run / "provenance.json").read_text())
        if result["actual_steps"] < 1 or result["final_test"] is None:
            raise ValueError(f"{run}: no completed trained-policy evaluation")
        if metadata["native_sha256"] == "" or metadata["encoder_sha256"] == "":
            raise ValueError(f"{run}: missing measured-model provenance")
        label = {"lingbot:small": "Off-the-shelf ViT-S", "run:runs/baseline": "Standard-distilled ViT-S",
                 "run:runs/mean": "Paired-consistency ViT-S", "state": "Privileged state",
                 "proprio": "Proprioception only"}.get(metadata["encoder"], metadata["encoder"])
        key = "visual" if metadata["actor_inputs"].startswith("pooled visual") else metadata["encoder"]
        score = result["final_test"][key]
        count = score["episodes"]
        if count < 128:
            raise ValueError(f"{run}: insufficient independent final episodes")
        interval = wilson(score["success"] * count, count)
        elapsed = result["elapsed_seconds_including_initialization_training_evaluation"]
        goal_steps = score["mean_success_steps"]
        rows.append([label, f"{result['actual_steps']:,}", f"{elapsed / 60:.2f}",
                     f"{score['success']:.3f} [{interval[0]:.3f}, {interval[1]:.3f}]",
                     f"{score['collision']:.3f}",
                     "—" if goal_steps is None else f"{goal_steps * 0.05:.2f}"])
        measurements.append({"run": str(run), "encoder": metadata["encoder"], "seed": metadata["seed"],
                             "actual_steps": result["actual_steps"], "selected_steps": result["selected_steps"],
                             "elapsed_seconds": elapsed, "width": metadata["width"], "height": metadata["height"],
                             "success_wilson95": interval, "gpu": metadata["gpu"], "result": result})
    first = measurements[0]
    resolutions = {(r["width"], r["height"]) for r in measurements}
    seeds = {r["seed"] for r in measurements}
    text = "== GPU-owned closed-loop navigation\n"
    text += "A separate Rust workspace tests frozen encoders in a first-person random-gap flight task. "
    text += "Hardware: " + ", ".join(sorted({r["gpu"] for r in measurements})) + ". "
    text += "Vulkan/wgpu-hal rasterization shares device-local buffers with NVIDIA Rust CUDA through external-memory FDs and timeline semaphores. "
    text += "CUDA owns dynamics, swept collisions, reward, reset and camera state; a recurrent PPO actor reads frozen visual tokens and proprioception. "
    text += "Frozen encoder inference is bfloat16; the actor and critic use float32. "
    text += "Visual actors see no privileged gap coordinates; the state-only ceiling does. "
    text += "The policy steers laterally with three actions; forward speed and altitude are stabilized. "
    text += "Physics runs at 120 Hz and control at 20 Hz; the observed finite horizon is 150 steps.\n\n"
    if len(resolutions) == 1:
        width, height = next(iter(resolutions))
        text += f"All compared runs use {width}×{height} RGB and the same episode protocol. "
    text += "Selection uses validation seeds and appearance only; final tests use a separate seed domain and palette. "
    text += "Each policy is tested on the first completed episode of 128 environments; the table reports actual executed transitions and end-to-end process time, including initialization and evaluation but excluding compilation.\n\n"
    text += "#figure(table(\n  columns: (auto, auto, auto, auto, auto, auto),\n  align: (left, center, center, center, center, center),\n  stroke: none, table.hline(),\n"
    text += "  [*Actor features*], [*Transitions*], [*Minutes*], [*Success [95% CI]* ↑], [*Collision* ↓], [*Goal (s)*],\n  table.hline(stroke: 0.5pt),\n"
    text += "\n".join("  " + ", ".join(f"[{value}]" for value in row) + "," for row in rows)
    text += "\n  table.hline(),\n), caption: [Independent closed-loop experiment. Goal time is conditional on success (dash: none); forward speed is fixed. Wall times were observed with concurrent distillation, not a controlled encoder-speed comparison. "
    text += f"Intervals are Wilson intervals over test episodes, not training-seed uncertainty. {len(seeds)} training seed(s) shown; no claim of convergence or real-world flight transfer.])\n\n"
    blind = first["result"]["final_test"]
    text += "Blind controls on the same test episodes achieve success "
    text += ", ".join(f"{name.replace('_', ' ')} {blind[name]['success']:.3f}" for name in ("center", "goal_centering", "random")) + ". "
    text += "A separate profiler capture of actual visual collection and PPO updates finds zero host transfers or CUDA host waits inside those ranges. "
    text += "Initialization, post-rollout logging and checkpoint/image export are explicit boundaries outside that claim. "
    text += "This is an initial control-relevance test on simple geometry, not proof of transfer to the detailed city or a physical drone.\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    args.out.with_suffix(".json").write_text(json.dumps(measurements, indent=2) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
