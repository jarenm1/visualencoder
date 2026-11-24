"""Vision panels for the slide deck: real model readouts and real mined frames. CPU only, run once.

    .venv/bin/python scripts/fig_presentation_vision.py

Writes three PNGs under report/figures/. Every panel is computed from cached artifacts; nothing is drawn from truth
except where stated (truth is used only to choose which legible example to show, and printed for the caption).

patch-readouts.png  976x240   RGB | predicted depth | predicted semantic, one held-out canonical eval-v2 frame.
    Encoder run:runs/presentation-retrain-seed2/mean. Probes: visenc.explore.DenseProbes fit exactly as the gate
    fit (first 1200 canonical frames outside the [0,245]^2 region of the first 80% of eval-v2; ridge). The frame is
    drawn from the held-out last 20%, which is never in the fit set. Depth: exp(log-depth ridge), turbo colour
    (near 2 m red, beyond 400 m and sky black). Semantic: argmax of the 7 ridge class scores, PALETTE colours.
    Frame choice: the held-out frame with the most distinct true classes covering >= 10 patches.

real-car-readout.png  976x306  UAVid RGB | seed-2 mean car score | seed-2 refined car score, plus a shared colour bar.
    Scores: sim_to_real_scores[..., 5] (vehicle column) from runs/presentation-retrain-seed2/gate/real/
    predictions-000.npz (mean) and predictions-001.npz (refined). These are sim-fit ridge outputs, uncalibrated, not
    probabilities. Both panels use one scale: 1st-99th percentile pooled over both models and all 70 val frames.
    Frame choice: the val frame with the most ground-truth car patches (frac[..., 5] >= 0.1).

pilot-mined.png  648x736  three rows, each: goal crop | observation.
    Frames from runs/mined-drone13/test (policy runs/drone13-v5-seed0/best.pt, encoder run:runs/mean). Magenta outline:
    the patch the policy chose (row-major 15x20 peak). Yellow box: true target screen bounds from privileged lanes
    25-28 (NDC, y up) when the target is visible (lane 14 > 0.5). Rows: wrong_peak (visible target, peak off it),
    false_view (target absent, policy confident in view), normal (visible, peak on target).

Conventions copied from sources (not imported, to keep this CPU-only and free of sim dependencies):
    VISIBLE_LANE = 14, BOX_LANES = 25:29 (sim/python/sim_rl/model.py). Patch centre x = (col+.5)/20*2-1,
    y = 1-(row+.5)/15*2. Pixel: px_x = (x+1)/2*320, px_y = (1-y)/2*240.

CPU only: CUDA is hidden before torch loads, so the GPU-resident PPO run is untouched. Encoder autocast is therefore
disabled and features are fp32; the gate used bf16 on GPU, so probe values can differ slightly from the gate's.
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU only: the GPU is busy with PPO
os.environ.setdefault("MPLBACKEND", "Agg")

import json  # noqa: E402
import sys  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from visenc import explore as X  # noqa: E402
from visenc.data import Split  # noqa: E402

torch.set_num_threads(6)
warnings.filterwarnings("ignore", message=".*CUDA is not available.*")

OUT = ROOT / "report" / "figures"
FEATURE_MODEL = "run:runs/presentation-retrain-seed2/mean"
REAL_RUNS = [
    ("seed-2 mean", "run:runs/presentation-retrain-seed2/mean", ROOT / "runs/presentation-retrain-seed2/gate/real/predictions-000.npz"),
    ("seed-2 refined", "run:runs/presentation-retrain-seed2/refined", ROOT / "runs/presentation-retrain-seed2/gate/real/predictions-001.npz"),
]
MINED = ROOT / "runs/mined-drone13/test"
FONT = matplotlib.font_manager.findfont("DejaVu Sans")

SIZE_HW = (240, 320)  # native frame
GRID_HW = (15, 20)  # 16 px patches
PATCH = 16
VEHICLE = 5
VISIBLE_LANE, BOX_LANES = 14, slice(25, 29)
BACKGROUND = (247, 249, 252)  # deck background, used for gutters
PATCH_RGB = (255, 0, 255)  # magenta: patch the policy chose
BOX_RGB = (255, 230, 0)  # yellow: true target box
GAP = 8


def hstack(panels: list[np.ndarray]) -> np.ndarray:
    out = np.empty((panels[0].shape[0], sum(p.shape[1] for p in panels) + GAP * (len(panels) - 1), 3), np.uint8)
    out[:] = BACKGROUND
    x = 0
    for p in panels:
        out[:, x : x + p.shape[1]] = p
        x += p.shape[1] + GAP
    return out


def vstack(rows: list[np.ndarray]) -> np.ndarray:
    out = np.empty((sum(r.shape[0] for r in rows) + GAP * (len(rows) - 1), rows[0].shape[1], 3), np.uint8)
    out[:] = BACKGROUND
    y = 0
    for r in rows:
        out[y : y + r.shape[0]] = r
        y += r.shape[0] + GAP
    return out


def patch_readouts() -> dict:
    sim = Split(ROOT / "data/eval-v2")
    n = len(sim)
    state = np.asarray(sim.state)
    outside = ~((state[:, 0] >= 0) & (state[:, 0] <= 245) & (state[:, 2] >= 0) & (state[:, 2] <= 245))
    fit = np.flatnonzero(outside[: int(0.8 * n)])[:1200]
    held = np.arange(int(0.8 * n), n)
    assert not np.isin(held, fit).any()

    _, held_seg = X.patch_truth(sim, held.tolist())
    n_cls = len(sim.meta["classes"])
    counts = np.stack([(held_seg == c).sum((1, 2)) for c in range(n_cls)], 1)
    index = int(held[np.argmax((counts >= 10).sum(1))])

    enc = X.Encoder(FEATURE_MODEL, device="cpu", batch_size=16)
    probe = X.DenseProbes(enc, sim, fit, view="canonical")
    rgb = np.asarray(sim.rgb["canonical"][index])
    depth, label = probe.predict(enc(rgb[None]).patches)
    truth_depth, truth_label = X.patch_truth(sim, [index])

    up = lambda grid: X.upsample(grid, SIZE_HW)[0]  # noqa: E731
    panels = [rgb, X.colour_depth(up(depth)), up(X.colour_seg(label))]
    Image.fromarray(hstack(panels)).save(OUT / "patch-readouts.png", optimize=True)
    return {
        "index": index,
        "fit_size": int(len(fit)),
        "patch_class_accuracy": float((label == truth_label).mean()),
        "median_abs_log_depth_error": float(np.median(np.abs(np.log(depth) - np.log(truth_depth)))),
        "held_out_distinct_classes": int((counts[index - int(0.8 * n)] >= 10).sum()),
    }


def colour_bar(lo: float, hi: float, cmap, width: int) -> np.ndarray:
    bar_h = 18
    ramp = (cmap(np.linspace(0, 1, width))[:, :3] * 255).round().astype(np.uint8)
    strip = np.empty((bar_h + 40, width, 3), np.uint8)
    strip[:] = BACKGROUND
    strip[:bar_h] = ramp[None]
    img = Image.fromarray(strip)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT, 16)
    span = hi - lo
    step = next(s for s in (0.05, 0.1, 0.2, 0.25, 0.5, 1.0) if span / s <= 6)
    for tick in np.arange(np.ceil(lo / step) * step, hi, step):
        x = round((tick - lo) / span * (width - 1))
        if x < 12 or x > width - 12:
            continue
        draw.line([(x, bar_h), (x, bar_h + 5)], fill=(40, 40, 40), width=1)
        draw.text((x, bar_h + 7), f"{tick + 0.0:.2f}", fill=(40, 40, 40), font=font, anchor="ma")
    return np.asarray(img)


def real_car() -> dict:
    meta = json.loads((ROOT / "data/uavid/meta.json").read_text())
    rgb = np.load(ROOT / "data/uavid/val_rgb.npy", mmap_mode="r")
    runs, frac = [], None
    for name, spec, path in REAL_RUNS:
        z = np.load(path)
        assert str(z["spec"]) == spec, path
        assert list(z["frames"]) == meta["val"], path
        if frac is None:
            frac = z["frac"]
        else:
            assert np.array_equal(frac, z["frac"]), path
        scores = z["sim_to_real_scores"][..., VEHICLE].reshape(-1, *GRID_HW)
        runs.append((name, scores))

    positives = (frac[..., VEHICLE] >= 0.1).sum(1)
    index = int(np.argmax(positives))
    lo, hi = np.percentile(np.concatenate([s.ravel() for _, s in runs]), [1, 99])
    cmap = matplotlib.colormaps["viridis"]

    def paint(s: np.ndarray) -> np.ndarray:
        t = np.clip((s - lo) / (hi - lo), 0, 1)
        return X.upsample((cmap(t)[..., :3] * 255).round().astype(np.uint8)[None], SIZE_HW)[0]

    panels = [np.asarray(rgb[index])] + [paint(s[index]) for _, s in runs]
    bar = colour_bar(lo, hi, cmap, 3 * SIZE_HW[1] + 2 * GAP)
    Image.fromarray(vstack([hstack(panels), bar])).save(OUT / "real-car-readout.png", optimize=True)
    return {
        "val_index": index,
        "uavid_frame": meta["val"][index],
        "gt_car_patches": int(positives[index]),
        "scale": [float(lo), float(hi)],
    }


def pilot_rows(d: Path) -> list[dict]:
    kind = np.load(d / "kind.npy")
    priv = np.load(d / "privileged.npy")
    peak = np.load(d / "peak.npy")
    in_view = np.load(d / "in_view.npy")
    goal = np.load(d / "goal_index.npy")

    vis = priv[:, VISIBLE_LANE] > 0.5
    box = priv[:, BOX_LANES].astype(np.float64)
    area = np.clip(box[:, 2] - box[:, 0], 0, None) * np.clip(box[:, 3] - box[:, 1], 0, None)
    rows, cols = np.meshgrid(np.arange(GRID_HW[0]), np.arange(GRID_HW[1]), indexing="ij")
    centres = np.stack(((cols.ravel() + 0.5) / GRID_HW[1] * 2 - 1, 1 - (rows.ravel() + 0.5) / GRID_HW[0] * 2), -1)
    pc = centres[peak]
    inside = (pc[:, 0] >= box[:, 0]) & (pc[:, 0] <= box[:, 2]) & (pc[:, 1] >= box[:, 1]) & (pc[:, 1] <= box[:, 3])
    miss = np.linalg.norm(pc - (box[:, :2] + box[:, 2:]) / 2, axis=1)

    used: set[int] = set()

    def pick(mask: np.ndarray, rank: np.ndarray) -> int:
        for i in np.flatnonzero(mask)[np.argsort(-rank[mask], kind="stable")]:
            if goal[i] not in used:
                used.add(int(goal[i]))
                return int(i)
        raise RuntimeError("no candidate with an unused goal")

    wrong = pick((kind == 1) & vis & (area >= 0.05) & (area <= 0.5), miss)
    absent = pick((kind == 2) & ~vis, in_view)
    normal = pick((kind == 0) & vis & inside, -np.abs(area - area[wrong]))
    names = {wrong: "wrong_peak", absent: "false_view", normal: "normal"}
    return [
        {
            "row": names[i],
            "index": i,
            "goal_index": int(goal[i]),
            "peak_rc": [int(v) for v in divmod(int(peak[i]), GRID_HW[1])],
            "visible": bool(vis[i]),
            "box_ndc": [round(float(v), 3) for v in box[i]] if vis[i] else None,
            "box_area_ndc": round(float(area[i]), 4),
            "in_view": round(float(in_view[i]), 4),
            "peak_to_box_centre": round(float(miss[i]), 3) if vis[i] else None,
        }
        for i in (wrong, absent, normal)
    ]


def observation(frame_chw: np.ndarray, row: dict) -> np.ndarray:
    img = Image.fromarray(np.ascontiguousarray(frame_chw.transpose(1, 2, 0)))
    draw = ImageDraw.Draw(img)
    if row["visible"]:
        x0, x1, y0, y1 = None, None, None, None
        b = row["box_ndc"]
        x0 = round((b[0] + 1) / 2 * SIZE_HW[1])
        x1 = round((b[2] + 1) / 2 * SIZE_HW[1])
        y0 = round((1 - b[3]) / 2 * SIZE_HW[0])
        y1 = round((1 - b[1]) / 2 * SIZE_HW[0])
        draw.rectangle([x0, y0, x1, y1], outline=BOX_RGB, width=2)
    r, c = row["peak_rc"]
    draw.rectangle([c * PATCH, r * PATCH, c * PATCH + PATCH - 1, r * PATCH + PATCH - 1], outline=PATCH_RGB, width=2)
    return np.asarray(img)


def pilot() -> list[dict]:
    rows = pilot_rows(MINED)
    images = np.load(MINED / "images.npy", mmap_mode="r")
    goals = np.load(MINED / "goal_images.npy", mmap_mode="r")
    blocks = [
        hstack([np.asarray(goals[r["goal_index"]]).transpose(1, 2, 0), observation(np.asarray(images[r["index"]]), r)])
        for r in rows
    ]
    Image.fromarray(vstack(blocks)).save(OUT / "pilot-mined.png", optimize=True)
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    mining = json.loads((ROOT / "runs/mined-drone13/mining.json").read_text())
    summary = {
        "patch-readouts": patch_readouts(),
        "real-car-readout": real_car(),
        "pilot-mined": pilot(),
        "mining_counts": mining["splits"][1]["counts"],
        "threads": torch.get_num_threads(),
        "cuda_visible": os.environ["CUDA_VISIBLE_DEVICES"],
    }
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
