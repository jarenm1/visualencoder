"""Generate city-drone visual-goal tables from completed ABI-v3 training runs only."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "sim/python")]
from sim_rl.cli import MIN_EPISODES  # noqa: E402
from sim_rl.features import ARMS  # noqa: E402
from sim_rl.native import ABI_VERSION, TASK  # noqa: E402

LABELS = {"pretrained": "Off-the-shelf ViT-S", "baseline": "Standard-distilled ViT-S",
          "mean": "Paired-consistency ViT-S", "state": "Privileged state", "hint": "Proprioception only"}
BUCKETS = ("visible", "partial", "hidden")
ARM_OF = {spec: arm for arm, spec in ARMS.items()}
SHARED = ("control_hz", "width", "height", "city_fingerprint")


def load_run(run):
    summary = json.loads((run / "summary.json").read_text())
    final = json.loads((run / "final-test.json").read_text())
    if summary.get("task") != TASK or summary.get("abi_version") != ABI_VERSION:
        raise ValueError(f"{run}: not a {TASK} ABI {ABI_VERSION} result")
    if final["policy"]["episodes"] < MIN_EPISODES:
        raise ValueError(f"{run}: fewer than {MIN_EPISODES} independent final episodes")
    return summary, final


def typst_table(headers, rows, caption):
    columns = ", ".join(["auto"] * len(headers))
    align = ", ".join(["left"] + ["center"] * (len(headers) - 1))
    head = ", ".join(f"[*{header}*]" for header in headers)
    body = "\n".join("  " + ", ".join(f"[{cell}]" for cell in row) + "," for row in rows)
    return (f"#figure(table(\n  columns: ({columns}),\n  align: ({align}),\n  stroke: none, table.hline(),\n"
            f"  {head},\n  table.hline(stroke: 0.5pt),\n{body}\n  table.hline(),\n), caption: [{caption}])\n\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("report/city-drone-goal.typ"))
    args = parser.parse_args()
    runs = [load_run(run) for run in args.runs]
    first = runs[0][0]
    for summary, _ in runs[1:]:
        for key in SHARED:
            if summary[key] != first[key]:
                raise ValueError(f"{key} differs between compared runs")
    rows, bucket_rows, measurements = [], [], []
    for run, (summary, final) in zip(args.runs, runs):
        policy = final["policy"]
        low, high = policy["success_wilson95"]
        goal_steps = policy["mean_success_steps"]
        arm = ARM_OF.get(summary["encoder"], summary["encoder"])
        elapsed = summary["elapsed_seconds_including_initialization_training_evaluation"]
        rows.append([LABELS.get(arm, arm), f"{summary['steps']:,}", f"{elapsed / 60:.2f}",
                     f"{policy['success']:.3f} [{low:.3f}, {high:.3f}]", f"{policy['collision']:.3f}",
                     "—" if goal_steps is None else f"{goal_steps / summary['control_hz']:.2f}"])
        for name in BUCKETS:
            bucket = policy["buckets"][name]
            bucket_rows.append([LABELS.get(arm, arm), name, str(bucket["episodes"]),
                                "—" if bucket.get("success") is None else f"{bucket['success']:.3f}"])
        measurements.append({"run": str(run), "encoder": summary["encoder"], "seed": summary["seed"],
                             "summary": summary, "final_test": final})
    fingerprint = json.dumps(first["city_fingerprint"])
    blind = runs[0][1]["heuristics"]
    text = "== City drone visual-goal navigation\n"
    text += ("A quadrotor in the imported city must reach a goal shown only in an image. "
             "Frozen encoder tokens feed a goal-attention actor with proprioception; a GRU PPO learner "
             "trains on the native ABI-v3 CUDA environment. "
             "Starts are 50% landed and 50% flying; curriculum buckets are visible, partial and hidden. ")
    text += (f"Control runs at {first['control_hz']} Hz; frames are {first['width']}×{first['height']} RGB. "
             f"City content fingerprint: #raw({fingerprint}).\n\n")
    text += ("Each policy is tested on the first completed episode of at least "
             f"{MIN_EPISODES} environments with uniform bucket weights. Intervals are Wilson 95% over test episodes. "
             "Process times include initialization, training and evaluation but exclude compilation.\n\n")
    text += typst_table(["Actor features", "Transitions", "Minutes", "Success [95% CI] ↑", "Collision ↓",
                         "Goal (s)"], rows,
                        "Independent visual-goal experiment. Goal time is conditional on success "
                        "(dash: none). No convergence or real-flight transfer claim.")
    text += typst_table(["Actor features", "Bucket", "Episodes", "Success"], bucket_rows,
                        "Per-visibility-bucket success on the same test episodes.")
    text += ("Blind controls on the same test episodes: "
             + ", ".join(f"{name} {blind[name]['success']:.3f}" for name in ("hover", "random")) + ".\n")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    args.out.with_suffix(".json").write_text(json.dumps(measurements, indent=2) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
