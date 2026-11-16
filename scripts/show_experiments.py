"""Build independent paper tables from completed, provenance-bearing experiments.

Run inside nix develop: python scripts/show_experiments.py
The default writes report/experiments.typ and reports/experiments/showcase.json.
Use --study geometry or --study mixed to showcase one completed study independently.
Geometry also emits report/geometry-extra.typ with global and real-transfer readouts.
Missing experiments are errors, never replaced with plans, smoke results or zeros.
"""

import argparse
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


def geometry_results(protocol):
    geometry = {arm: values_by_spec(REPORTS / f"geometry/{arm}/report.json")
                for arm in ("fixed", "multi")}
    real = values_by_spec(REPORTS / "geometry/real/report.json")
    summary: list[dict[str, object]] = []
    cells = []
    global_cells = []
    real_means: dict[tuple[str, str], list[float]] = {}
    for arm, source in (("fixed", "controls"), ("multi", "treatments")):
        payload = read_json(REPORTS / f"geometry/{arm}/report.json")
        if payload["protocol"]["arm_semantics"] != "encoder training geometry label only":
            raise ValueError("geometry reports did not keep readout data fixed")
        for target in ("own", "mean"):
            specs = protocol[source][target]
            metrics = {}
            for regime in ("familiar/novel", "heldout/canonical", "heldout/novel"):
                for metric in ("semantic_miou", "depth_absrel"):
                    metrics[f"{regime}/{metric}"] = mean_seed(
                        geometry[arm], specs, lambda row, r=regime, m=metric: row["summary"][r][m])
            label = f"{arm}, {'standard' if target == 'own' else 'paired'}"
            cells.append([label, *(f"{v['mean']:.3f}" for v in metrics.values())])
            global_metrics = {}
            for regime in ("familiar/novel", "heldout/canonical", "heldout/novel"):
                global_metrics[f"{regime}/pose_r2"] = mean_seed(
                    geometry[arm], specs, lambda row, r=regime: row["summary"][r]["pose_r2"])
            for regime in ("familiar/novel", "heldout/novel"):
                global_metrics[f"{regime}/retrieval_at_1"] = mean_seed(
                    geometry[arm], specs, lambda row, r=regime: row["summary"][r]["pairs"]["retrieval_at_1"])
            global_cells.append([label, *(f"{v['mean']:.3f}" for v in global_metrics.values())])
            real_metrics = {
                "sim_to_real_miou": mean_seed(real, specs, lambda row: row["sim_to_real"]["miou"]),
                "car_auroc": mean_seed(real, specs, lambda row: row["sim_to_real"]["vehicle_auroc"]),
                "car_ap": mean_seed(real, specs, lambda row: row["sim_to_real"]["vehicle_ap"]),
                "real_to_real_miou": mean_seed(real, specs, lambda row: row["real_to_real"]["miou"]),
            }
            real_means[arm, target] = [float(v["mean"]) for v in real_metrics.values()]
            summary.append({"arm": arm, "target": target, "specs": specs, "metrics": metrics,
                            "global_metrics": global_metrics, "real_metrics": real_metrics})
    text = "== Geometry versus appearance\n"
    text += "Training geometry and consistency are crossed at a fixed 20,000-pose, 10,000-step budget. "
    text += "Canonical probes fit 750 original-city poses; all encoders share 750 familiar and 1,500 held-out test poses. "
    text += "City extent and camera support are matched. Each entry averages two training seeds.\n\n"
    text += table(["Training", "App. IoU ↑", "Depth ↓", "Geo. IoU ↑", "Depth ↓", "Both IoU ↑", "Depth ↓"],
                  cells,
                  "Independent geometry experiment. Appearance: original city, novel look. Geometry: held-out layouts, canonical look. Both: held-out layouts and novel looks. Depth is AbsRel; per-scene and seed-level results accompany the code.")
    extra = "== Additional geometry readouts\n"
    extra += table(["Training", "App. pose R²", "Geo. pose R²", "Both pose R²", "App. retrieval", "Both retrieval"],
                   global_cells,
                   "Same geometry probe protocol as the main table. Retrieval candidates: 750 within each layout. App., Geo., and Both denote the appearance-only, geometry-only, and combined shifts.")
    deltas = []
    for target in ("own", "mean"):
        values = [new - old for new, old in zip(real_means["multi", target], real_means["fixed", target])]
        deltas.append(["standard" if target == "own" else "paired", *(f"{value:+.4f}" for value in values)])
    extra += table(["Target", "Δ real IoU", "Δ car AUROC", "Δ car AP", "Δ real-fit IoU"],
                   deltas,
                   "Geometry intervention on UAVid: three-layout minus fixed-layout training, averaged over matched seeds. These descriptive differences are not equivalence tests; real-image scores retain the same simulator-fit and real-fit probe protocols.")
    (ROOT / "report/geometry-extra.typ").write_text(extra)
    return text, summary


def mixed_results(protocol):
    real = values_by_spec(REPORTS / "mixed/real/report.json")
    sim = values_by_spec(REPORTS / "mixed/sim/report.json")
    summary: list[dict[str, object]] = []
    cells = []
    for condition, source in (("sim only", "controls"), ("sim + real", "treatments")):
        for target in ("own", "mean"):
            specs = protocol[source][target]
            metrics = {
                "sim_to_real_miou": mean_seed(real, specs, lambda row: row["sim_to_real"]["miou"]),
                "car_auroc": mean_seed(real, specs, lambda row: row["sim_to_real"]["vehicle_auroc"]),
                "car_ap": mean_seed(real, specs, lambda row: row["sim_to_real"]["vehicle_ap"]),
                "real_to_real_miou": mean_seed(real, specs, lambda row: row["real_to_real"]["miou"]),
                "sim_novel_miou": mean_seed(sim, specs, lambda row: np.mean([
                    value["semantic_miou"] for view, value in row["views"].items() if view.startswith("novel")])),
            }
            label = f"{condition}, {'standard' if target == 'own' else 'paired'}"
            cells.append([label, *(f"{v['mean']:.3f}" for v in metrics.values())])
            summary.append({"condition": condition, "target": target, "specs": specs, "metrics": metrics})
    text = "== Unlabelled-real adaptation\n"
    text += "Only UAVid training RGB is added; labels and validation frames are excluded from distillation. "
    text += "Both conditions use 64 images/step: 32 paired simulator poses versus 24 paired poses plus 16 real frames. "
    text += "This matches compute and optimizer steps, not simulator exposure; it is real-data-assisted adaptation, not sim-only transfer.\n\n"
    text += table(["Training", "Sim→real IoU ↑", "Car AUROC ↑", "Car AP ↑", "Real→real IoU ↑", "Sim novel IoU ↑"], cells,
                  "Independent real-data experiment, two training seeds per condition. Car AP uses ≥10% car versus car-free patches. Whole-video bootstrap intervals and paired differences are in the saved real-image evaluation; these means do not imply seed-level significance.")
    return text, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=("all", "geometry", "mixed"), default="all")
    args = parser.parse_args()
    studies = ("geometry", "mixed") if args.study == "all" else (args.study,)
    protocols: dict[str, object] = {}
    summary: dict[str, object] = {"protocols": protocols}
    sections = []
    for study in studies:
        protocol = read_json(REPORTS / study / "experiment.json")
        if protocol["steps"] != 10000 or len(protocol["seeds"]) < 2:
            raise ValueError(f"{study}: paper tables require full-budget, multiple-seed results")
        builder = geometry_results if study == "geometry" else mixed_results
        text, rows = builder(protocol)
        sections.append(text)
        summary[study], protocols[study] = rows, protocol
    name = "experiments" if args.study == "all" else args.study
    output = ROOT / "report" / f"{name}.typ"
    output.write_text("\n".join(sections))
    summary_path = REPORTS / "showcase.json" if args.study == "all" else REPORTS / args.study / "showcase.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
