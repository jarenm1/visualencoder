"""Generate city navigation tables from completed ABI-v2 GPU experiments only."""

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
    parser.add_argument("--out", type=Path, default=Path("report/city-navigation.typ"))
    args = parser.parse_args()
    rows = []
    measurements = []
    for run in args.runs:
        result = json.loads((run / "summary.json").read_text())
        metadata = json.loads((run / "provenance.json").read_text())
        contract = metadata.get("task_contract", {})
        if metadata.get("task") != "city-block-navigation-v1" or contract.get("abi_version") != 2:
            raise ValueError(f"{run}: historical gap results are not city navigation evidence")
        if measurements and contract != measurements[0]["task_contract"]:
            raise ValueError(f"{run}: scene/control/split/reward contract differs between compared city runs")
        if result["actual_steps"] < 1 or result["final_test"] is None:
            raise ValueError(f"{run}: no completed trained-policy evaluation")
        if metadata["native_sha256"] == "" or metadata["encoder_sha256"] == "":
            raise ValueError(f"{run}: missing measured-model provenance")
        label = {"lingbot:small": "Off-the-shelf ViT-S", "run:runs/baseline": "Standard-distilled ViT-S",
                 "run:runs/mean": "Paired-consistency ViT-S", "state": "Privileged waypoint state",
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
                     "—" if goal_steps is None else f"{goal_steps / contract['control_hz']:.2f}"])
        measurements.append({"run": str(run), "encoder": metadata["encoder"], "seed": metadata["seed"],
                             "actual_steps": result["actual_steps"], "selected_steps": result["selected_steps"],
                             "elapsed_seconds": elapsed, "width": metadata["width"], "height": metadata["height"],
                             "task_contract": contract,
                             "success_wilson95": interval, "gpu": metadata["gpu"], "result": result})
    first = measurements[0]
    resolutions = {(r["width"], r["height"]) for r in measurements}
    seeds = {r["seed"] for r in measurements}
    contract = first["task_contract"]
    text = "== GPU-owned city block navigation\n"
    text += "A separate Rust workspace tests frozen encoders on the actual imported city geometry at stabilized altitude. "
    text += "Hardware: " + ", ".join(sorted({r["gpu"] for r in measurements})) + ". "
    text += "Vulkan/wgpu-hal rasterization shares device-local buffers with NVIDIA Rust CUDA through external-memory FDs and timeline semaphores. "
    text += "CUDA owns kinematic agent state, swept collisions, reward, reset and camera state; a recurrent PPO actor reads frozen visual tokens and proprioception. "
    text += "Frozen encoder inference is bfloat16; the actor and critic use float32. "
    text += "Visual actors receive only images and final-goal proprioception, never route waypoints or collision geometry; the explicit state ceiling sees next-waypoint state. "
    text += f"The {contract['action_count']}-action controller commands hover or eight unit-normalized horizontal velocity headings in the goal-facing camera frame, with automatic goal yaw. "
    text += f"Kinematic/collision substeps run at {contract['physics_hz']} Hz and control at {contract['control_hz']} Hz; the finite terminal horizon is {contract['horizon']} steps. "
    text += "Collision uses exact swept-sphere tests against the same transformed city triangles used for rendering. City RGB uses palette base colours, not the legacy phototexture/normal-map/shadow pipeline. "
    text += "Dense reward measures progress along collision-free route waypoints, with time cost and terminal success/collision terms. "
    text += f"Scene content fingerprint: #raw({json.dumps(contract['city_fingerprint'])}).\n\n"
    if len(resolutions) == 1:
        width, height = next(iter(resolutions))
        text += f"All compared runs use {width}×{height} RGB and the same episode protocol. "
    text += "Selection uses a validation route district; train, validation and final-test routes occupy spatially disjoint city districts with fixed independent catalogs. "
    text += "Each policy is tested on the first completed episode of at least 128 environments; the table reports actual executed transitions and end-to-end process time, including initialization and evaluation but excluding compilation.\n\n"
    text += "#figure(table(\n  columns: (auto, auto, auto, auto, auto, auto),\n  align: (left, center, center, center, center, center),\n  stroke: none, table.hline(),\n"
    text += "  [*Actor features*], [*Transitions*], [*Minutes*], [*Success [95% CI]* ↑], [*Collision* ↓], [*Goal (s)*],\n  table.hline(stroke: 0.5pt),\n"
    text += "\n".join("  " + ", ".join(f"[{value}]" for value in row) + "," for row in rows)
    text += "\n  table.hline(),\n), caption: [Independent city navigation experiment. Goal time is conditional on success (dash: none); commanded horizontal speed and altitude are stabilized. Process times include the observed concurrent workload, not a controlled encoder-speed comparison. "
    text += f"Intervals are Wilson intervals over test episodes, not training-seed uncertainty. {len(seeds)} training seed(s) shown; no claim of convergence or real-world flight transfer.])\n\n"
    blind = first["result"]["final_test"]
    text += "Blind controls on the same test episodes achieve success "
    text += ", ".join(f"{name.replace('_', ' ')} {blind[name]['success']:.3f}" for name in ("hover", "goal_centering", "random")) + ". "
    text += "The profiling command separately verifies zero host transfers or CUDA host waits inside actual visual collection and PPO update ranges. "
    text += "Initialization, post-rollout logging and checkpoint/image export are explicit boundaries outside that claim. "
    text += "This is a rendered city navigation experiment with goal-facing stabilized control, not full rotor flight or proof of transfer to a physical drone.\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    args.out.with_suffix(".json").write_text(json.dumps(measurements, indent=2) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
