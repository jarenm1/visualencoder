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
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .data import Split
from .evaluate import Ridge
from .explore import DenseProbes, Encoder

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


def scores(pred: np.ndarray, label: np.ndarray, valid: np.ndarray, frac: np.ndarray, vehicle_score: np.ndarray) -> dict:
    p, t = pred[valid], label[valid]
    ious = {}
    for c in EVAL_CLASSES:
        union = ((p == c) | (t == c)).sum()
        if (t == c).any():
            ious[EVAL_NAMES[c]] = float(((p == c) & (t == c)).sum() / union)
    # Vehicle detection: patches with >=10% car vs car-free patches (AUROC of the probe's vehicle score).
    pos, neg = vehicle_score[frac[..., 5] >= 0.1], vehicle_score[frac[..., 5] == 0]
    s = np.concatenate([pos, neg])
    r = np.empty(len(s))
    r[np.argsort(s, kind="stable")] = np.arange(1, len(s) + 1)
    auroc = (r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)) if len(pos) and len(neg) else float("nan")
    return {"miou": float(np.mean(list(ious.values()))), "accuracy": float((p == t).mean()), "iou": ious, "vehicle_auroc": float(auroc)}


def run(args: argparse.Namespace) -> None:
    real = Path(args.real)
    tr_rgb, tr_frac = np.load(real / "train_rgb.npy"), np.load(real / "train_frac.npy")
    va_rgb, va_frac = np.load(real / "val_rgb.npy"), np.load(real / "val_frac.npy")
    va_label, va_valid = patch_targets(va_frac)
    sim = Split(args.sim)
    n = len(sim)
    st = np.asarray(sim.state)
    x, z = st[:, 0], st[:, 2]
    outside = ~((x >= 0) & (x <= 245) & (z >= 0) & (z <= 245))
    sim_fit = np.flatnonzero(outside[: int(0.8 * n)])[: args.sim_fit]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for spec in args.encoders:
        enc = Encoder(spec, batch_size=16)
        va = enc(va_rgb).patches.flatten(1, 2)  # [70,300,D]
        flat = va.reshape(-1, va.shape[-1])
        row = {"spec": spec}

        sim_probe = DenseProbes(enc, sim, sim_fit, view=args.sim_view).semantic  # 7-way fractions, sim classes
        out_sim = sim_probe(flat).reshape(*va.shape[:2], -1).cpu().numpy()
        pred_sim = np.asarray(EVAL_CLASSES)[out_sim[..., EVAL_CLASSES].argmax(-1)]  # restrict to shared classes
        row["sim_to_real"] = scores(pred_sim, va_label, va_valid, va_frac, out_sim[..., 5])

        tr = enc(tr_rgb).patches.flatten(1, 2)
        tr_valid = tr_frac[..., IGNORE] < 0.5
        y = torch.from_numpy(tr_frac[..., :IGNORE][tr_valid])
        real_probe = Ridge(tr[torch.from_numpy(tr_valid)], y, enc.device)
        out_real = real_probe(flat).reshape(*va.shape[:2], -1).cpu().numpy()
        pred_real = np.asarray(EVAL_CLASSES)[out_real[..., EVAL_CLASSES].argmax(-1)]
        row["real_to_real"] = scores(pred_real, va_label, va_valid, va_frac, out_real[..., 5])
        enc.release()
        results.append(row)
        print(f"{spec}: sim->real mIoU {row['sim_to_real']['miou']:.3f}  real->real mIoU {row['real_to_real']['miou']:.3f}", flush=True)
        (out / "report.json").write_text(json.dumps(results, indent=2))

    names = list(EVAL_NAMES.values())
    lines = ["| encoder | sim→real mIoU | sim→real acc | " + " | ".join(f"sim→real {c}" for c in names) + " | sim→real car AUROC | real→real mIoU | gap |",
             "|" + "---|" * (6 + len(names))]  # fmt: skip
    for r in results:
        s, t = r["sim_to_real"], r["real_to_real"]
        cells = [r["spec"], s["miou"], s["accuracy"], *(s["iou"].get(c, float("nan")) for c in names), s["vehicle_auroc"], t["miou"], t["miou"] - s["miou"]]
        lines.append("| " + " | ".join(c if isinstance(c, str) else f"{c:.3f}" for c in cells) + " |")
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--sim", default="data/eval-v2", help="simulator split to fit the sim->real probe on")
    p.add_argument("--sim-view", default="canonical")
    p.add_argument("--sim-fit", type=int, default=1200, help="simulator poses for the sim probe")
    p.add_argument("--real", default="data/uavid", help="output of `visenc uavid`")
    p.add_argument("--encoders", nargs="+", required=True)
    p.add_argument("--out", required=True)
