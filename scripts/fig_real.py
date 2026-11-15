"""Report figure: sim-trained segmentation probes applied to real UAVid frames.

    python scripts/fig_real.py   ->  report/figures/real.png

Each encoder's per-patch segmentation probe is fit on simulator renders only (canonical view, as in
`visenc real-eval`) and applied unchanged to UAVid val frames. Black = classes with no simulator
counterpart (human, clutter) in the ground truth.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np

from visenc import explore as X
from visenc.data import Split
from visenc.real import EVAL_CLASSES, IGNORE

FRAMES = [0, 30, 60]
ENCODERS = {
    "LingBot ViT-S": "lingbot:small",
    "LingBot ViT-L (teacher)": "lingbot:large",
    "Ours: distillation": "run:runs/baseline",
    "Ours: + consistency": "run:runs/mean",
}

rgb = np.load("data/uavid/val_rgb.npy")[FRAMES]
frac = np.load("data/uavid/val_frac.npy")[FRAMES]
truth = frac.argmax(-1).reshape(-1, 15, 20)
palette = np.vstack([X.PALETTE, [[0, 0, 0]]]).astype(np.uint8)  # IGNORE (7) drawn black

sim = Split(Path("data/eval-v2"))
state = np.asarray(sim.state)
outside = ~((state[:, 0] >= 0) & (state[:, 0] <= 245) & (state[:, 2] >= 0) & (state[:, 2] <= 245))
fit = np.flatnonzero(outside[: int(0.8 * len(sim))])[:1200]

preds = {}
for name, spec in ENCODERS.items():
    enc = X.Encoder(spec)
    probe = X.DenseProbes(enc, sim, fit).semantic
    p = enc(rgb).patches
    out = probe(p.reshape(-1, p.shape[-1])).reshape(*p.shape[:3], -1).cpu().numpy()
    preds[name] = np.asarray(EVAL_CLASSES)[out[..., EVAL_CLASSES].argmax(-1)]
    enc.release()


def up(labels: np.ndarray) -> np.ndarray:
    return palette[labels].repeat(16, 0).repeat(16, 1)


rows = [[rgb[i], up(truth[i]), *(up(preds[n][i]) for n in ENCODERS)] for i in range(len(FRAMES))]
fig = X.grid(rows, [f"frame {k}" for k in range(1, len(FRAMES) + 1)], ["real image", "ground truth", *ENCODERS], scale=1.6)
fig.savefig("report/figures/real.png", dpi=170, bbox_inches="tight")
print("wrote report/figures/real.png; ignore class id", IGNORE)
