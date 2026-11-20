"""Encoder gate: accept a candidate encoder only if it fixes the target failures without losing
anything else, judged with paired bootstrap confidence intervals against a baseline encoder.

Three suites, each scored on held-out data for both encoders (results are cached in --out):
  sim       `visenc eval` on --sim-data: frozen linear probes for pose, depth and semantics on
            appearance-shifted views and a spatially held-out region, plus same-pose retrieval
            and patch cosine. Paired bootstrap over poses.
  failures  confusion probe: a fresh goal-saliency head per encoder, trained on --mined frames and
            scored on held-out --failures frames. Paired bootstrap over goals (frames of one goal
            are correlated).
  real      `visenc real-eval` on UAVid: sim->real segmentation and car patch ranking, and the
            real->real readout. Its own paired bootstrap over whole video sequences.

Verdict: accept iff every TARGET metric improves (95% CI of candidate - baseline above 0) and
every GUARD metric is non-inferior (95% CI lower bound of the improvement above -margin).
Margins are fixed below and recorded in the verdict.

    python sim/python/encoder_gate.py --baseline run:runs/mean --candidate run:runs/posttrain-mean-b4 \
        --out reports/gate/posttrain-mean-b4
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from visenc.city import OBSERVABLE_STATE, STATE_NAMES
from visenc.evaluate import sample_name

ROOT = Path(__file__).resolve().parents[2]
VISENC = str(Path(sys.executable).with_name("visenc"))
RESAMPLES = 1000

# name: (suite, higher is better, non-inferiority margin; None marks a target metric)
METRICS = {
    "failures/patch_head_score": ("failures", True, None),
    "failures/global_head_score": ("failures", True, 0.01),
    "sim/pose_r2_shifted": ("sim", True, 0.02),
    "sim/depth_absrel_shifted": ("sim", False, 0.01),
    "sim/semantic_miou_shifted": ("sim", True, 0.01),
    "sim/pose_r2_region": ("sim", True, 0.02),
    "sim/depth_absrel_region": ("sim", False, 0.01),
    "sim/semantic_miou_region": ("sim", True, 0.01),
    "sim/retrieval_at_1": ("sim", True, 0.02),
    "sim/patch_cosine": ("sim", True, 0.01),
    "real/sim_to_real_miou": ("real", True, 0.01),
    "real/sim_to_real_vehicle_auroc": ("real", True, 0.01),
    "real/sim_to_real_vehicle_ap": ("real", True, 0.02),
    "real/real_to_real_miou": ("real", True, 0.01),
}


def run(command, log):
    print("+", " ".join(command), flush=True)
    with open(log, "a") as handle:
        subprocess.run(command, check=True, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)


def suites(args, specs):
    """Run (or reuse) every suite for both encoders."""
    out = args.out
    sim, failures, real = out / "sim", out / "failures", out / "real"
    if not all((sim / "samples" / f"{sample_name(s)}.npz").exists() for s in specs):
        run([VISENC, "eval", "--data", str(args.sim_data), "--encoders", *specs, "--heldout-region",
             *map(str, args.heldout_region), "--out", str(sim), "--samples", str(sim / "samples")], out / "sim.log")
    if not all((failures / "samples" / f"{sample_name(s)}-{h}.npz").exists() for s in specs
               for h in ("patch", "global")):
        run([sys.executable, "sim/python/train.py", "confusion-probe", "--encoders", *specs, "--mined",
             str(args.mined), "--held-out", str(args.failures), "--heads", "patch", "global",
             "--out", str(failures / "report.json"), "--samples", str(failures / "samples")], out / "failures.log")
    if not (real / "report.json").exists():
        run([VISENC, "real-eval", "--encoders", *specs, "--baseline", args.baseline, "--real", str(args.real),
             "--out", str(real)], out / "real.log")
    return sim, failures, real


def draws(clusters, resamples, seed):
    """Index arrays of cluster-bootstrap resamples: whole clusters drawn with replacement."""
    rng = np.random.default_rng(seed)
    unique, inverse = np.unique(clusters, return_inverse=True)
    members = [np.flatnonzero(inverse == k) for k in range(len(unique))]
    for _ in range(resamples):
        yield np.concatenate([members[k] for k in rng.integers(0, len(unique), len(unique))])


def sim_metrics(samples, rows, region_rows, fit_view):
    """Gate metrics of one encoder's sim samples on test pose rows `rows` and held-out region pose
    rows `region_rows` (None without a region)."""
    views = sorted({key.split("/")[0] for key in samples if key.endswith("/state_pred") and "/" in key
                    and not key.startswith("region/")})
    shifted = [v for v in views if v != fit_view]
    observable = [STATE_NAMES.index(name) for name in OBSERVABLE_STATE]

    def pose_r2(prefix, view, take):
        pred, true = samples[f"{prefix}{view}/state_pred"][take], samples[f"{prefix}{view}/state_true"][take]
        sse = ((pred - true) ** 2).sum(0)
        sst = ((true - true.mean(0)) ** 2).sum(0)
        return float(np.mean(1 - sse[observable] / np.maximum(sst[observable], 1e-12)))

    def absrel(prefix, view, take):
        return float(samples[f"{prefix}{view}/depth_absrel_sum"][take].sum()
                     / max(samples[f"{prefix}{view}/depth_count"][take].sum(), 1))

    def miou(prefix, view, take):
        inter = samples[f"{prefix}{view}/class_intersection"][take].sum(0)
        union = samples[f"{prefix}{view}/class_union"][take].sum(0)
        support = samples[f"{prefix}{view}/class_support"][take].sum(0) > 0
        return float(np.mean(inter[support] / np.maximum(union[support], 1)))

    out = {
        "sim/pose_r2_shifted": np.mean([pose_r2("", v, rows) for v in shifted]),
        "sim/depth_absrel_shifted": np.mean([absrel("", v, rows) for v in shifted]),
        "sim/semantic_miou_shifted": np.mean([miou("", v, rows) for v in shifted]),
        "sim/retrieval_at_1": np.mean([samples[f"pairs/{v}/retrieval_hit"][rows].mean() for v in shifted]),
        "sim/patch_cosine": np.mean([samples[f"pairs/{v}/patch_cosine"][rows].mean() for v in shifted]),
    }
    if region_rows is not None:
        out.update({
            "sim/pose_r2_region": np.mean([pose_r2("region/", v, region_rows) for v in shifted]),
            "sim/depth_absrel_region": np.mean([absrel("region/", v, region_rows) for v in shifted]),
            "sim/semantic_miou_region": np.mean([miou("region/", v, region_rows) for v in shifted]),
        })
    return out


def failure_score(outcomes, take):
    """Probe score: mean of peak-on-target (target visible) and absent-correct (target hidden)."""
    visible, hit, absent_ok = (outcomes[take, k] for k in range(3))
    return 0.5 * (hit[visible].mean() + absent_ok[~visible].mean())


def paired(base, cand, resampled_base, resampled_cand):
    differences = np.asarray(resampled_cand) - np.asarray(resampled_base)
    low, high = np.percentile(differences, [2.5, 97.5])
    return {"baseline": float(base), "candidate": float(cand), "difference": float(cand - base),
            "ci95": [float(low), float(high)]}


def compare(args, sim, failures, real):
    results = {}
    # Sim: paired pose bootstrap; test and region poses are resampled independently.
    base_s = dict(np.load(sim / "samples" / f"{sample_name(args.baseline)}.npz"))
    cand_s = dict(np.load(sim / "samples" / f"{sample_name(args.candidate)}.npz"))
    poses = np.arange(len(base_s["test_index"]))
    region_key = f"region/{args.fit_view}/depth_count"
    region = np.arange(len(base_s[region_key])) if region_key in base_s else None
    point_b = sim_metrics(base_s, poses, region, args.fit_view)
    point_c = sim_metrics(cand_s, poses, region, args.fit_view)
    boot_b, boot_c = {k: [] for k in point_b}, {k: [] for k in point_b}
    region_draws = draws(region, RESAMPLES, args.seed + 1) if region is not None else iter(lambda: None, 0)
    for take, region_take in zip(draws(poses, RESAMPLES, args.seed), region_draws):
        for store, samples in ((boot_b, base_s), (boot_c, cand_s)):
            for key, value in sim_metrics(samples, take, region_take, args.fit_view).items():
                store[key].append(value)
    for key in point_b:
        results[key] = paired(point_b[key], point_c[key], boot_b[key], boot_c[key])
    # Failures: paired goal-cluster bootstrap per head.
    for head in ("patch", "global"):
        b = np.load(failures / "samples" / f"{sample_name(args.baseline)}-{head}.npz")
        c = np.load(failures / "samples" / f"{sample_name(args.candidate)}-{head}.npz")
        frames = np.arange(len(b["outcomes"]))
        takes = list(draws(b["goal_index"], RESAMPLES, args.seed))
        results[f"failures/{head}_head_score"] = paired(
            failure_score(b["outcomes"], frames), failure_score(c["outcomes"], frames),
            [failure_score(b["outcomes"], t) for t in takes], [failure_score(c["outcomes"], t) for t in takes])
    # Real: real-eval's own paired sequence bootstrap.
    rows = {row["spec"]: row for row in json.loads((real / "report.json").read_text())}
    pair = rows[args.candidate]["paired_vs_baseline"]
    for direction in ("sim_to_real", "real_to_real"):
        for metric in ("miou", "vehicle_auroc", "vehicle_ap"):
            key = f"real/{direction}_{metric}"
            if key not in METRICS:
                continue
            p = pair[direction][metric]
            results[key] = {"baseline": rows[args.baseline][direction][metric],
                            "candidate": rows[args.candidate][direction][metric],
                            "difference": p["difference"], "ci95": [p["ci95"]["low"], p["ci95"]["high"]]}
    return results


def verdict(results):
    checks = {}
    for key, (_, higher, margin) in METRICS.items():
        if key not in results:
            continue
        r = results[key]
        # Improvement interval in the metric's good direction.
        low, high = r["ci95"] if higher else (-r["ci95"][1], -r["ci95"][0])
        role = "target" if margin is None else "guard"
        passed = low > 0 if margin is None else low > -margin
        checks[key] = {**r, "role": role, "margin": margin, "improvement_ci95": [low, high], "pass": bool(passed)}
    accepted = all(c["pass"] for c in checks.values())
    return {"accepted": accepted, "failed": [k for k, c in checks.items() if not c["pass"]], "checks": checks}


def markdown(args, decision):
    lines = [f"# Encoder gate: {args.candidate} vs {args.baseline}", "",
             f"**{'ACCEPT' if decision['accepted'] else 'REJECT'}**"
             + ("" if decision["accepted"] else f" (failed: {', '.join(decision['failed'])})"), "",
             "| metric | role | baseline | candidate | improvement [95% CI] | margin | pass |", "|---|---|---|---|---|---|---|"]
    for key, c in decision["checks"].items():
        low, high = c["improvement_ci95"]
        sign = 1 if METRICS[key][1] else -1
        lines.append(f"| {key} | {c['role']} | {c['baseline']:.4f} | {c['candidate']:.4f} | "
                     f"{sign * c['difference']:+.4f} [{low:+.4f}, {high:+.4f}] | "
                     f"{'-' if c['margin'] is None else c['margin']} | {'✓' if c['pass'] else '✗'} |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--baseline", required=True, help="visenc encoder spec, e.g. run:runs/mean")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--sim-data", type=Path, default=Path("data/eval-v2"))
    parser.add_argument("--heldout-region", type=float, nargs=4, default=[0, 0, 245, 245],
                        help="world rectangle excluded from distillation (reported as the region suite)")
    parser.add_argument("--fit-view", default="canonical")
    # Encoder-centric mined flights (drone13): probes fit on train/, scored on the held-out test/ district.
    parser.add_argument("--mined", type=Path, default=Path("runs/mined-drone13"),
                        help="mine output holding train/ (probe fitting)")
    parser.add_argument("--failures", type=Path, default=Path("runs/mined-drone13"),
                        help="mine output holding test/ (held-out failure frames)")
    parser.add_argument("--real", type=Path, default=Path("data/uavid"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sim, failures, real = suites(args, [args.baseline, args.candidate])
    decision = verdict(compare(args, sim, failures, real))
    decision.update({"baseline": args.baseline, "candidate": args.candidate, "resamples": RESAMPLES,
                     "data": {"sim": str(args.sim_data), "heldout_region": args.heldout_region,
                              "mined": str(args.mined), "failures": str(args.failures), "real": str(args.real)}})
    (args.out / "verdict.json").write_text(json.dumps(decision, indent=1))
    text = markdown(args, decision)
    (args.out / "verdict.md").write_text(text)
    print(text, flush=True)
    sys.exit(0 if decision["accepted"] else 1)


if __name__ == "__main__":
    main()
