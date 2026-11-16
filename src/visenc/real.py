"""Sim-to-real evaluation on UAVid (real oblique urban drone video frames, CC BY-NC-SA 4.0).

`visenc uavid --out data/uavid` downloads the labelled splits (200 train, 70 val frames, 3840 or 4096 x 2160), centre-crops
each to 4:3 (2880x2160), resizes the RGB to 320x240, and stores per-patch class fractions computed on the full-
resolution mask: the 4:3 crop is exactly 15x20 patches of 144 px, so no label resampling is needed.

UAVid classes map onto ours; human and clutter have no counterpart in the city and are ignored:
    building -> building, road -> road, tree -> vegetation, low vegetation -> ground,
    static car / moving car -> vehicle

`visenc real-eval` fits the per-patch segmentation probe two ways and scores both on UAVid val:
    sim -> real   probe fit on simulator renders (canonical view, as in `visenc eval`), applied unchanged
    real -> real  probe fit on UAVid train: an upper bound for what the frozen features hold about real images
The gap between the two is the part of the sim-to-real gap that the features do not close.

Evaluation saves complete per-frame probe scores and provenance. Recompute metrics without models with
`python -m visenc.real rescore --reports RUN/report.json --out RESCORED --bootstrap-resamples 1000`.
Intervals resample whole validation video sequences, paired across methods; standard UAVid has only
seven validation sequences, so these intervals exclude training uncertainty and have limited support.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np

REPO = "dronefreak/UAVid-2020"
UAVID_NAMES = ["clutter", "building", "road", "static_car", "tree", "low_vegetation", "human", "moving_car"]
# Our classes: 0 sky, 1 ground, 2 road, 3 building, 4 vegetation, 5 vehicle, 6 prop; 7 = ignore.
IGNORE = 7
UAVID_TO_OURS = np.array([IGNORE, 3, 2, 5, 4, 1, IGNORE, 5], dtype=np.uint8)
EVAL_CLASSES = [1, 2, 3, 4, 5]  # classes present in both worlds
EVAL_NAMES = {1: "ground", 2: "road", 3: "building", 4: "vegetation", 5: "vehicle"}
GRID = (15, 20)
CROP = (2160, 2880)


def prepare(out: Path) -> None:
    from huggingface_hub import snapshot_download
    from PIL import Image

    root = Path(snapshot_download(REPO, repo_type="dataset", allow_patterns=[f"{k}/{s}/*" for k in ("images", "masks") for s in ("train", "val")]))
    out.mkdir(parents=True, exist_ok=True)
    meta = {"source": REPO, "license": "CC BY-NC-SA 4.0", "classes_ours": "0 sky,1 ground,2 road,3 building,4 vegetation,5 vehicle,6 prop,7 ignore"}
    for split in ("train", "val"):
        paths = sorted((root / "images" / split).glob("*.png"))
        rgb = np.zeros((len(paths), 240, 320, 3), np.uint8)
        frac = np.zeros((len(paths), GRID[0] * GRID[1], IGNORE + 1), np.float32)
        for k, p in enumerate(paths):
            im = Image.open(p).convert("RGB")
            mask = np.asarray(Image.open(root / "masks" / split / p.name))
            width = im.size[0]
            if im.size[1] != 2160 or width < CROP[1] or mask.shape != (2160, width):
                raise ValueError(f"unexpected size {im.size} / {mask.shape} for {p.name}")
            x0 = (width - CROP[1]) // 2
            rgb[k] = np.asarray(im.crop((x0, 0, x0 + CROP[1], CROP[0])).resize((320, 240), Image.LANCZOS))
            ours = UAVID_TO_OURS[np.minimum(mask[:, x0 : x0 + CROP[1]], 7)]  # 255 (unknown colour) -> ignore via clamp+map
            ours[mask[:, x0 : x0 + CROP[1]] == 255] = IGNORE
            blocks = ours.reshape(GRID[0], CROP[0] // GRID[0], GRID[1], CROP[1] // GRID[1]).transpose(0, 2, 1, 3).reshape(-1, (CROP[0] // GRID[0]) ** 2)
            frac[k] = np.stack([(blocks == c).mean(1) for c in range(IGNORE + 1)], 1)
            print(f"{split} {k + 1}/{len(paths)} {p.name}", flush=True)
        np.save(out / f"{split}_rgb.npy", rgb)
        np.save(out / f"{split}_frac.npy", frac)
        meta[split] = [p.name for p in paths]
    (out / "meta.json").write_text(json.dumps(meta, indent=1))


def patch_targets(frac: np.ndarray, max_ignore: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Per-patch label over EVAL_CLASSES (argmax of mapped fractions) and a validity mask (mostly mapped)."""
    valid = frac[..., IGNORE] < max_ignore
    label = np.asarray(EVAL_CLASSES)[frac[..., EVAL_CLASSES].argmax(-1)]
    return label, valid


METRICS = ("miou", "vehicle_auroc", "vehicle_ap")
DIRECTIONS = ("sim_to_real", "real_to_real")


def sequence_ids(frames: list[str]) -> np.ndarray:
    """Recover video identity from the UAVid filenames saved by prepare()."""
    ids = []
    for frame in frames:
        match = re.fullmatch(r"(seq\d+)_\d+\.[A-Za-z0-9]+", Path(frame).name)
        if match is None:
            raise ValueError(f"cannot recover UAVid video sequence from {frame!r}")
        ids.append(match[1])
    return np.asarray(ids)


def validate_provenance(meta: dict, n_train: int, n_val: int) -> tuple[np.ndarray, np.ndarray]:
    """Check the actual frame ordering and sequence-disjoint split before fitting any probe."""
    for split, count in (("train", n_train), ("val", n_val)):
        frames = meta[split]
        if len(frames) != count or len(set(frames)) != count or count == 0:
            raise ValueError(f"{split} metadata must contain exactly {count} unique, nonempty frame names")
    train, val = sequence_ids(meta["train"]), sequence_ids(meta["val"])
    overlap = sorted(set(train) & set(val))
    if overlap:
        raise ValueError(f"train/val video sequence leakage: {', '.join(overlap)}")
    return train, val


def sequence_resamples(ids: np.ndarray, resamples: int = 1000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return sequence multiplicities, never independently sampled frames or patches."""
    if resamples < 0:
        raise ValueError("bootstrap resamples must be nonnegative")
    unique = np.unique(ids)
    if not len(unique):
        raise ValueError("at least one validation sequence is required")
    draws = np.random.default_rng(seed).integers(len(unique), size=(resamples, len(unique)))
    weights = np.zeros((resamples, len(unique)), dtype=np.int64)
    np.add.at(weights, (np.arange(resamples)[:, None], draws), 1)
    return unique, weights


def _binary_metrics(pos: np.ndarray, neg: np.ndarray) -> dict:
    """Counts at ascending, distinct score thresholds; ties receive half credit in AUROC."""
    n_pos, n_neg = float(pos.sum()), float(neg.sum())
    auroc = float(np.sum(pos * (np.cumsum(neg) - 0.5 * neg)) / (n_pos * n_neg)) if n_pos and n_neg else float("nan")
    # Non-interpolated AP: add each tied group together, then evaluate its precision.
    tp, fp = np.cumsum(pos[::-1]), np.cumsum(neg[::-1])
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) > 0)
    ap = float(np.sum(pos[::-1] * precision) / n_pos) if n_pos else float("nan")
    return {
        "vehicle_auroc": auroc,
        "vehicle_ap": ap,
        "vehicle_positive_count": int(n_pos),
        "vehicle_negative_count": int(n_neg),
        "vehicle_prevalence": n_pos / (n_pos + n_neg) if n_pos + n_neg else float("nan"),
    }


def _blocks(pred: np.ndarray, label: np.ndarray, valid: np.ndarray, frac: np.ndarray,
            vehicle_score: np.ndarray, ids: np.ndarray) -> dict:
    """Sufficient statistics per sequence; threshold groups retain exact tied-score semantics."""
    _, inverse = np.unique(ids, return_inverse=True)
    n_seq = int(inverse.max()) + 1
    seq = np.broadcast_to(inverse.reshape((-1,) + (1,) * (pred.ndim - 1)), pred.shape)
    class_index = np.full(IGNORE + 1, -1, dtype=int)
    class_index[EVAL_CLASSES] = np.arange(len(EVAL_CLASSES))
    confusion = np.zeros((n_seq, len(EVAL_CLASSES), len(EVAL_CLASSES)), dtype=np.int64)
    np.add.at(confusion, (seq[valid], class_index[label[valid]], class_index[pred[valid]]), 1)
    pos, neg = frac[..., 5] >= 0.1, frac[..., 5] == 0
    eligible = pos | neg
    if not np.isfinite(vehicle_score[eligible]).all():
        raise ValueError("car detection scores must be finite")
    _, group = np.unique(vehicle_score[eligible], return_inverse=True)
    positive = np.zeros((n_seq, int(group.max()) + 1 if len(group) else 0), dtype=np.int64)
    negative = np.zeros_like(positive)
    np.add.at(positive, (seq[eligible], group), pos[eligible])
    np.add.at(negative, (seq[eligible], group), neg[eligible])
    return {"confusion": confusion, "positive": positive, "negative": negative}


def _block_scores(blocks: dict, weights: np.ndarray) -> dict:
    confusion = np.einsum("s,sij->ij", weights, blocks["confusion"])
    gt, predicted = confusion.sum(1), confusion.sum(0)
    intersection = confusion.diagonal()
    ious = {EVAL_NAMES[c]: float(intersection[i] / (gt[i] + predicted[i] - intersection[i]))
            for i, c in enumerate(EVAL_CLASSES) if gt[i]}
    binary = _binary_metrics(np.einsum("s,sg->g", weights, blocks["positive"]),
                             np.einsum("s,sg->g", weights, blocks["negative"]))
    return {
        "miou": float(np.mean(list(ious.values()))) if ious else float("nan"),
        "accuracy": float(intersection.sum() / confusion.sum()) if confusion.sum() else float("nan"),
        "iou": ious,
        **binary,
    }


def scores(pred: np.ndarray, label: np.ndarray, valid: np.ndarray, frac: np.ndarray, vehicle_score: np.ndarray) -> dict:
    return _block_scores(_blocks(pred, label, valid, frac, vehicle_score, np.zeros(len(pred), dtype=int)), np.ones(1, dtype=int))


def _ci(values: np.ndarray) -> dict:
    finite = values[np.isfinite(values)]
    lo, hi = np.quantile(finite, [0.025, 0.975]) if len(finite) else (None, None)
    return {"low": None if lo is None else float(lo), "high": None if hi is None else float(hi),
            "valid_resamples": int(len(finite))}


def _summarize_cache(cache: dict, weights: np.ndarray) -> tuple[dict, dict]:
    row, samples = {"spec": str(cache["spec"].item())}, {}
    for direction in DIRECTIONS:
        output = cache[f"{direction}_scores"]
        blocks = _blocks(cache[f"{direction}_pred"], cache["label"], cache["valid"], cache["frac"],
                         output[..., 5], cache["sequence_ids"])
        row[direction] = _block_scores(blocks, np.ones(weights.shape[1], dtype=int))
        values = [_block_scores(blocks, w) for w in weights]
        samples[direction] = {metric: np.asarray([v[metric] for v in values]) for metric in METRICS}
        row[direction]["ci95"] = {metric: _ci(samples[direction][metric]) for metric in METRICS}
    row["gap"] = {metric: row["real_to_real"][metric] - row["sim_to_real"][metric] for metric in METRICS}
    samples["gap"] = {metric: samples["real_to_real"][metric] - samples["sim_to_real"][metric] for metric in METRICS}
    row["gap"]["ci95"] = {metric: _ci(samples["gap"][metric]) for metric in METRICS}
    return row, samples


def _protocol(meta: dict, ids: np.ndarray, resamples: int, seed: int) -> dict:
    return {
        "schema_version": 2,
        "source": meta.get("source", REPO),
        "train_frames": meta["train"], "val_frames": meta["val"],
        "train_sequence_ids": sorted(set(sequence_ids(meta["train"]))),
        "val_sequence_ids": sorted(set(ids)),
        "sequence_disjoint_checked": True,
        "segmentation": "Shared-class argmax [ground,road,building,vegetation,vehicle]; valid iff ignore fraction <0.5; mIoU averages classes with ground-truth support.",
        "car_detection": "Positive car fraction >=0.1; negative exactly 0; ambiguous (0,0.1) excluded. Uses all eligible patches, independently of segmentation validity. Score is uncalibrated ridge vehicle output. AUROC gives ties half credit; AP groups all equal-score thresholds (non-interpolated).",
        "bootstrap": {"unit": "whole UAVid validation video sequence", "resamples": resamples, "seed": seed,
                      "interval": "95% percentile", "paired": "Identical sequence draws across encoders, probe directions and gaps; probes fixed, not refit.",
                      "undefined": "Nonfinite draws excluded per metric; valid_resamples is reported. No-negative AUROC and no-positive AUROC/AP are undefined."},
        "gap": "real_to_real minus sim_to_real, separately for mIoU, car AUROC and car AP",
        "limitation": f"Only {len(set(ids))} validation sequences (standard UAVid val has only 7): intervals are descriptive, have limited cluster support, and exclude probe-fit/training uncertainty.",
    }


def _render_report(results: list[dict], protocol: dict) -> str:
    def cell(branch: dict, metric: str) -> str:
        value, ci = branch[metric], branch["ci95"][metric]
        if ci["low"] is None:
            return f"{value:.3f} [undefined CI]"
        return f"{value:.3f} [{ci['low']:.3f}, {ci['high']:.3f}]"

    lines = ["| encoder | sim→real mIoU [95% CI] | car AUROC [95% CI] | car AP [95% CI] | real→real mIoU [95% CI] | mIoU gap [95% CI] | car + / − | prevalence |",
             "|" + "---|" * 8]
    for row in results:
        s, t = row["sim_to_real"], row["real_to_real"]
        cells = [row["spec"], cell(s, "miou"), cell(s, "vehicle_auroc"), cell(s, "vehicle_ap"),
                 cell(t, "miou"), cell(row["gap"], "miou"),
                 f"{s['vehicle_positive_count']} / {s['vehicle_negative_count']}", f"{s['vehicle_prevalence']:.4f}"]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", protocol["limitation"], "", protocol["segmentation"], "", protocol["car_detection"],
              "", f"Bootstrap: {protocol['bootstrap']['resamples']} whole-sequence resamples; seed {protocol['bootstrap']['seed']}. " + protocol["bootstrap"]["paired"],
              "", "Full real→real car metrics, gap intervals, valid-resample counts and paired baseline differences are in report.json."]
    return "\n".join(lines) + "\n"


def rescore(reports: list[Path], out: Path, resamples: int = 1000, seed: int = 0, baseline: str | None = None) -> list[dict]:
    """Recompute all metrics from saved per-frame arrays, optionally merging evaluation reports."""
    entries = []
    for report in reports:
        report = Path(report)
        for entry in json.loads(report.read_text()):
            if "predictions" not in entry:
                raise ValueError(f"{report}: no saved predictions; historical reports need one encoder/probe rerun")
            path = Path(entry["predictions"])
            entries.append((report.parent / path).resolve())
    if not entries:
        raise ValueError("no saved encoder predictions")
    # Load one cache at a time; the frame arrays, not encoder weights, suffice.
    results, samples, reference = [], [], None
    for path in entries:
        with np.load(path, allow_pickle=False) as archive:
            cache = dict(archive)
        meta = json.loads(str(cache["provenance"].item()))
        _, ids = validate_provenance(meta, len(meta["train"]), len(cache["frac"]))
        if not np.array_equal(ids, cache["sequence_ids"]):
            raise ValueError(f"{path}: cached sequence IDs disagree with metadata")
        identity = {key: cache[key] for key in ("frames", "sequence_ids", "label", "valid", "frac")}
        if reference is None:
            reference = identity
            _, weights = sequence_resamples(ids, resamples, seed)
            protocol = _protocol(meta, ids, resamples, seed)
        elif any(not np.array_equal(reference[key], identity[key]) for key in reference):
            raise ValueError("paired reports must have identical validation frame order, labels, masks and fractions")
        if list(cache["frames"]) != meta["val"]:
            raise ValueError(f"{path}: frame ordering disagrees with provenance")
        row, draws = _summarize_cache(cache, weights)
        if any(r["spec"] == row["spec"] for r in results):
            raise ValueError(f"duplicate encoder spec {row['spec']!r}; use one report per encoder")
        row["predictions"] = str(path)
        row["probe_fit"] = json.loads(str(cache["probe_fit"].item()))
        results.append(row)
        samples.append(draws)
    if baseline is not None:
        matches = [i for i, row in enumerate(results) if row["spec"] == baseline]
        if len(matches) != 1:
            raise ValueError(f"baseline {baseline!r} must exactly match one encoder spec in the input reports")
        index = matches[0]
        for row, draws in zip(results, samples):
            comparison = {"spec": baseline}
            for direction in (*DIRECTIONS, "gap"):
                comparison[direction] = {
                    metric: {"difference": row[direction][metric] - results[index][direction][metric],
                             "ci95": _ci(draws[direction][metric] - samples[index][direction][metric])}
                    for metric in METRICS
                }
            row["paired_vs_baseline"] = comparison
        protocol["baseline"] = baseline
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(results, indent=2))
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2))
    table = _render_report(results, protocol)
    (out / "report.md").write_text(table)
    print(table)
    return results


def run(args: argparse.Namespace) -> None:
    # Keep the historical fit data, region selection, row ordering and probe settings unchanged.
    import torch
    from .data import Split
    from .evaluate import Ridge
    from .explore import DenseProbes, Encoder

    real = Path(args.real)
    tr_rgb, tr_frac = np.load(real / "train_rgb.npy"), np.load(real / "train_frac.npy")
    va_rgb, va_frac = np.load(real / "val_rgb.npy"), np.load(real / "val_frac.npy")
    meta = json.loads((real / "meta.json").read_text())
    _, ids = validate_provenance(meta, len(tr_rgb), len(va_rgb))
    if len(tr_frac) != len(tr_rgb) or len(va_frac) != len(va_rgb):
        raise ValueError("RGB/fraction frame counts differ")
    resamples, seed = getattr(args, "bootstrap_resamples", 1000), getattr(args, "bootstrap_seed", 0)
    sequence_resamples(ids, resamples, seed)  # Reject invalid settings before fits.
    baseline = getattr(args, "baseline", None)
    out = Path(args.out)
    results = []
    if (out / "report.json").exists():
        if not getattr(args, "append", False):
            raise ValueError(f"{out}/report.json already exists; use --append or a new output directory")
        results = json.loads((out / "report.json").read_text())
        if any("predictions" not in row for row in results):
            raise ValueError("cannot append historical reports without cached predictions; use a new output directory")
    existing = {row["spec"] for row in results}
    if baseline is not None and baseline not in existing and args.encoders.count(baseline) != 1:
        raise ValueError("--baseline must exactly match an existing or new encoder spec")
    if len(set(args.encoders)) != len(args.encoders) or existing.intersection(args.encoders):
        raise ValueError("encoder specs must be unique across existing and new rows")
    va_label, va_valid = patch_targets(va_frac)
    sim = Split(args.sim)
    n = len(sim)
    st = np.asarray(sim.state)
    x, z = st[:, 0], st[:, 2]
    outside = ~((x >= 0) & (x <= 245) & (z >= 0) & (z <= 245))
    sim_fit = np.flatnonzero(outside[: int(0.8 * n)])[: args.sim_fit]
    out.mkdir(parents=True, exist_ok=True)
    start = len(results)
    for index, spec in enumerate(args.encoders, start=start):
        prediction_path = out / f"predictions-{index:03d}.npz"
        if prediction_path.exists():
            raise ValueError(f"refusing to overwrite saved predictions: {prediction_path}")
        enc = Encoder(spec, batch_size=16)
        va = enc(va_rgb).patches.flatten(1, 2)
        flat = va.reshape(-1, va.shape[-1])
        sim_probe = DenseProbes(enc, sim, sim_fit, view=args.sim_view).semantic
        out_sim = sim_probe(flat).reshape(*va.shape[:2], -1).cpu().numpy()
        pred_sim = np.asarray(EVAL_CLASSES)[out_sim[..., EVAL_CLASSES].argmax(-1)]
        tr = enc(tr_rgb).patches.flatten(1, 2)
        tr_valid = tr_frac[..., IGNORE] < 0.5
        y = torch.from_numpy(tr_frac[..., :IGNORE][tr_valid])
        real_probe = Ridge(tr[torch.from_numpy(tr_valid)], y, enc.device)
        out_real = real_probe(flat).reshape(*va.shape[:2], -1).cpu().numpy()
        pred_real = np.asarray(EVAL_CLASSES)[out_real[..., EVAL_CLASSES].argmax(-1)]
        enc.release()
        fit = {"sim": str(Path(args.sim).resolve()), "sim_view": args.sim_view,
               "sim_fit_requested": args.sim_fit, "sim_fit_indices": sim_fit.tolist(),
               "sim_selection": "outside [0,245]x[0,245], first 80% of split, first sim_fit eligible poses; historical real-eval protocol",
               "real": str(real.resolve()), "real_train_frames": meta["train"]}
        np.savez_compressed(prediction_path, spec=np.asarray(spec), provenance=np.asarray(json.dumps(meta)),
                            probe_fit=np.asarray(json.dumps(fit)), frames=np.asarray(meta["val"]), sequence_ids=ids,
                            label=va_label, valid=va_valid, frac=va_frac,
                            sim_to_real_pred=pred_sim, sim_to_real_scores=out_sim,
                            real_to_real_pred=pred_real, real_to_real_scores=out_real)
        row = {"spec": spec, "predictions": prediction_path.name}
        results.append(row)
        (out / "report.json").write_text(json.dumps(results, indent=2))
        print(f"{spec}: saved per-frame predictions to {prediction_path}", flush=True)
    rescore([out / "report.json"], out, resamples, seed, baseline)


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--sim", default="data/eval-v2", help="simulator split to fit the sim->real probe on (historical default)")
    p.add_argument("--sim-view", default="canonical")
    p.add_argument("--sim-fit", type=int, default=1200, help="simulator poses for the sim probe")
    p.add_argument("--real", default="data/uavid", help="output of `visenc uavid`")
    p.add_argument("--encoders", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--append", action="store_true", help="add new encoder rows to an existing cached report without overwriting prior rows")
    _add_bootstrap_args(p)


def _add_bootstrap_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--bootstrap-resamples", type=int, default=1000, help="whole-video sequence bootstrap draws; 0 disables CIs")
    p.add_argument("--bootstrap-seed", type=int, default=0)
    p.add_argument("--baseline", help="exact encoder spec for paired differences (must be present in input encoders/reports)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    add_args(commands.add_parser("eval", help="fit historical probes and save reusable per-frame predictions"))
    p = commands.add_parser("rescore", help="metrics/bootstrap only; no encoder loading or probe fitting")
    p.add_argument("--reports", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    _add_bootstrap_args(p)
    args = parser.parse_args()
    if args.command == "eval":
        run(args)
    else:
        rescore(args.reports, args.out, args.bootstrap_resamples, args.bootstrap_seed, args.baseline)


if __name__ == "__main__":
    main()
