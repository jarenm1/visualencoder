"""Build independent paper tables from completed, provenance-bearing experiments.

Run inside nix develop: python scripts/show_experiments.py
Writes report/experiments.typ and reports/experiments/showcase.json. No missing
experiment or checkpoint is silently replaced with a plan, smoke result or zero.
Navigation is added only when a GPU-native benchmark has measured results.
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports/experiments"


def read_json(path):
    return json.loads(path.read_text())


def values_by_spec(path):
    payload = read_json(path)
    rows = payload if isinstance(payload, list) else payload["results"]
    return {row["spec"]: row for row in rows}


def mean_seed(rows, specs, value):
    numbers = np.asarray([value(rows[spec]) for spec in specs], dtype=float)
    if not np.isfinite(numbers).all():
        raise ValueError(f"nonfinite results for {specs}")
    return {"mean": float(numbers.mean()), "min": float(numbers.min()), "max": float(numbers.max()),
            "seeds": len(numbers)}


def table(headers, rows, caption):
    body = ["#figure(table(", f"  columns: ({', '.join(['auto'] * len(headers))}),",
            "  align: (left, " + ", ".join(["center"] * (len(headers) - 1)) + "),",
            "  stroke: none, table.hline(),", "  " + ", ".join(f"[*{header}*]" for header in headers) + ",",
            "  table.hline(stroke: 0.5pt),"]
    body += ["  " + ", ".join(f"[{cell}]" for cell in row) + "," for row in rows]
    body += ["  table.hline(),", f"), caption: [{caption}])", ""]
    return "\n".join(body)


def main():
    geometry_protocol = read_json(REPORTS / "geometry/experiment.json")
    mixed_protocol = read_json(REPORTS / "mixed/experiment.json")
    for name, protocol in (("geometry", geometry_protocol), ("mixed", mixed_protocol)):
        if protocol["steps"] != 10000 or len(protocol["seeds"]) < 2:
            raise ValueError(f"{name}: paper tables require the declared full-budget, multiple-seed experiment")
    geometry = {arm: values_by_spec(REPORTS / f"geometry/{arm}/report.json") for arm in ("fixed", "multi")}
    mixed_real = values_by_spec(REPORTS / "mixed/real/report.json")
    mixed_sim = values_by_spec(REPORTS / "mixed/sim/report.json")
    geometry_summary: list[dict[str, object]] = []
    mixed_summary: list[dict[str, object]] = []
    summary = {"geometry": geometry_summary, "mixed": mixed_summary,
               "protocols": {"geometry": geometry_protocol, "mixed": mixed_protocol}}
    geometry_cells = []
    for arm, source in (("fixed", "controls"), ("multi", "treatments")):
        for target in ("own", "mean"):
            specs = geometry_protocol[source][target]
            # Probes and test images must be identical across training-geometry arms.
            payload = read_json(REPORTS / f"geometry/{arm}/report.json")
            if payload["protocol"]["arm_semantics"] != "encoder training geometry label only":
                raise ValueError("geometry reports did not keep readout data fixed")
            metrics = {}
            for regime in ("familiar/novel", "heldout/canonical", "heldout/novel"):
                for metric in ("semantic_miou", "depth_absrel"):
                    metrics[f"{regime}/{metric}"] = mean_seed(
                        geometry[arm], specs, lambda row, r=regime, m=metric: row["summary"][r][m])
            label = f"{arm}, {'standard' if target == 'own' else 'paired'}"
            geometry_cells.append([label, *(f"{v['mean']:.3f}" for v in metrics.values())])
            geometry_summary.append({"arm": arm, "target": target, "specs": specs, "metrics": metrics})
    mixed_cells = []
    for condition, source in (("sim only", "controls"), ("sim + real", "treatments")):
        for target in ("own", "mean"):
            specs = mixed_protocol[source][target]
            metrics = {
                "sim_to_real_miou": mean_seed(mixed_real, specs, lambda row: row["sim_to_real"]["miou"]),
                "car_ap": mean_seed(mixed_real, specs, lambda row: row["sim_to_real"]["vehicle_ap"]),
                "real_to_real_miou": mean_seed(mixed_real, specs, lambda row: row["real_to_real"]["miou"]),
                "sim_novel_miou": mean_seed(mixed_sim, specs, lambda row: np.mean([
                    value["semantic_miou"] for view, value in row["views"].items() if view.startswith("novel")])),
            }
            label = f"{condition}, {'standard' if target == 'own' else 'paired'}"
            mixed_cells.append([label, *(f"{v['mean']:.3f}" for v in metrics.values())])
            mixed_summary.append({"condition": condition, "target": target, "specs": specs, "metrics": metrics})
    text = "== Geometry versus appearance\n"
    text += "Training geometry and consistency are crossed at a fixed 20,000-pose, 10,000-step budget. "
    text += "All encoders use the same canonical probe-fitting poses and the same test images; held-out layouts never enter probe fitting. "
    text += "City extent and camera support are matched. Each entry averages two training seeds.\n\n"
    text += table(["Training", "Appearance IoU ↑", "Depth ↓", "Geometry IoU ↑", "Depth ↓", "Both IoU ↑", "Depth ↓"],
                  geometry_cells,
                  "Independent geometry experiment. Appearance: original city, novel look. Geometry: held-out layouts, canonical look. Both: held-out layouts and novel looks. Depth is AbsRel; per-scene and seed-level results accompany the code.")
    text += "== Unlabelled-real adaptation\n"
    text += "Only UAVid training RGB is added; labels and validation frames are excluded from distillation. "
    text += "Both conditions use 64 images/step: 32 paired simulator poses versus 24 paired poses plus 16 real frames. "
    text += "This matches compute and optimizer steps, not simulator exposure; it is real-data-assisted adaptation, not sim-only transfer.\n\n"
    text += table(["Training", "Sim→real IoU ↑", "Car AP ↑", "Real→real IoU ↑", "Sim novel IoU ↑"], mixed_cells,
                  "Independent real-data experiment, two training seeds per condition. Car AP uses ≥10% car versus car-free patches. Whole-video sequence bootstrap intervals and paired differences are in the saved real-image evaluation; these means do not imply seed-level significance.")
    output = ROOT / "report/experiments.typ"
    output.write_text(text)
    (REPORTS / "showcase.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
