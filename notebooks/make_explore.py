"""Generates notebooks/explore.ipynb (kept as code so the notebook diffs cleanly).

    python notebooks/make_explore.py && jupyter lab notebooks/explore.ipynb
"""

from pathlib import Path

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
    md("""# What do the encoders see?

Frozen-feature inspection on `data/eval-v2` (held-out poses; `canonical`, `random*` = seen appearance regime,
`novel*` = held-out textures and lighting). Four views into the features:

1. **PCA-RGB**: patch features projected to three colours with one basis shared across appearance views. Matching colours across views means appearance-invariant features.
2. **Patch similarity**: one query patch against every patch of every view (correspondence).
3. **Linear probe readouts**: what depth and segmentation are linearly decodable, and where they fail.
4. **Global map**: the CLS + mean-patch vector over many poses, coloured by ground-truth state.

Run from the repo root's dev shell (`jupyter lab notebooks/explore.ipynb`)."""),
    code("""import os, numpy as np, torch, matplotlib.pyplot as plt
os.chdir(os.path.dirname(os.path.abspath("__file__")) if os.path.basename(os.getcwd()) == "notebooks" else os.getcwd())
if os.path.basename(os.getcwd()) == "notebooks":
    os.chdir("..")
from visenc.data import Split
from visenc.city import STATE_NAMES
from visenc import explore as X

split = Split("data/eval-v2")
VIEWS = split.views                       # canonical, random1, random2, novel1, novel2
H, W = split.size
N = len(split)
state = np.asarray(split.state)
x, z = state[:, STATE_NAMES.index("pos_x")], state[:, STATE_NAMES.index("pos_z")]
outside = ~((x >= 0) & (x <= 245) & (z >= 0) & (z <= 245))   # same held-out region as the benchmark
fit_index = np.flatnonzero(outside[: int(0.8 * N)])[:600]   # probe fitting poses (as in eval)
test_pool = np.flatnonzero(outside & (np.arange(N) >= int(0.8 * N)))
POSES = list(np.random.default_rng(3).choice(test_pool, 4, replace=False))
ENCODERS = {
    "LingBot S (off-shelf)": "lingbot:small",
    "LingBot L (teacher)": "lingbot:large",
    "DINOv3 L": "hf:facebook/dinov3-vitl16-pretrain-lvd1689m",
    "Ours: baseline ViT-S": "run:runs/baseline",
}
print(VIEWS, "poses:", POSES)"""),
    md("## 0. The views and their ground truth"),
    code("""imgs = X.views_of(split, POSES, VIEWS)                     # [V, P, H, W, 3]
gt_depth = np.stack([split.depth[i] for i in POSES]).astype(np.float32)
gt_seg = np.stack([split.semantic[i] for i in POSES])
rows = [[imgs[v, p] for v in range(len(VIEWS))] + [X.colour_seg(gt_seg[p]), X.colour_depth(gt_depth[p])] for p in range(len(POSES))]
X.grid(rows, [f"pose {i}" for i in POSES], VIEWS + ["GT segmentation", "GT depth"]);
print("classes:", split.meta["classes"])"""),
    md("""## 1. Features for every encoder

One encoder on the GPU at a time; features for the chosen poses stay on the CPU. Each encoder also embeds 64 other
poses (canonical + `novel1`) as the reference set for a PCA basis that is shared across poses."""),
    code("""feats, basis = {}, {}
basis_poses = np.random.default_rng(1).choice(test_pool, 64, replace=False)
basis_imgs = np.concatenate([np.stack([split.rgb[v][i] for i in basis_poses]) for v in ("canonical", "novel1")])
for name, spec in ENCODERS.items():
    enc = X.Encoder(spec)
    f = enc(imgs.reshape(-1, H, W, 3))
    feats[name] = X.Features(f.cls.unflatten(0, (len(VIEWS), len(POSES))), f.patches.unflatten(0, (len(VIEWS), len(POSES))))
    basis[name] = enc(basis_imgs).patches
    enc.release()
    print(f"{name:24s} cls {tuple(f.cls.shape)}  patches {tuple(f.patches.shape)}")"""),
    md("""## 2. PCA-RGB: how each encoder groups the image

**Per-pose basis** (first figure per pose): PCA over all views of this one pose. The top components are dominated by
the near-to-far gradient (sky, horizon, ground). **Many-pose basis** (second figure): PCA fit on 64 other poses, so
colours mean the same feature direction in every image and object-level structure (roofs, roads, trees) shows up.
Read across a row: if colours stay put under seen and novel looks, the features ignore the appearance change."""),
    code("""for p_i, pose in enumerate(POSES[:2]):
    for label, use_basis in (("per-pose basis", False), ("many-pose basis", True)):
        rows, titles = [list(imgs[:, p_i])], ["image"]
        for name, f in feats.items():
            per_view = X.pca_rgb([f.patches[v, p_i : p_i + 1] for v in range(len(VIEWS))], (H, W),
                                 basis_from=basis[name] if use_basis else None)
            rows.append([r[0] for r in per_view]); titles.append(name)
        X.grid(rows, titles, VIEWS).suptitle(f"pose {pose}: PCA-RGB, {label}", y=1.01)"""),
    md("""## 3. Patch similarity: does a surface stay recognisable across looks?

Query: a building patch near the image centre of the canonical view (red box). Each map shows cosine similarity of
every patch, in every view of the same pose, to that query. A sharp, consistent blob across views means the encoder
re-identifies the same surface under new lighting and textures."""),
    code("""def pick_query(seg_patch, cls_id=3):
    h, w = seg_patch.shape
    ys, xs = np.nonzero(seg_patch == cls_id)
    if len(ys) == 0:
        return h // 2, w // 2
    k = np.argmin((ys - h / 2) ** 2 + (xs - w / 2) ** 2)
    return int(ys[k]), int(xs[k])

_, seg_patch = X.patch_truth(split, POSES)
p_i = 0
row, col = pick_query(seg_patch[p_i], cls_id=split.meta["classes"].index("building"))
rows, titles = [], []
for name, f in feats.items():
    maps = X.similarity(f.patches[0, p_i], row, col, f.patches[:, p_i], (H, W))
    rows.append(list(maps)); titles.append(name)
fig = X.grid(rows, titles, VIEWS, cmap="magma", vmin=0, vmax=1)
for ax in fig.axes[: len(VIEWS)]:
    ax.add_patch(plt.Rectangle((col * 16, row * 16), 16, 16, fill=False, ec="red", lw=1.5))
fig.suptitle(f"pose {POSES[p_i]}: similarity to query patch ({row},{col})", y=1.01);"""),
    md("""### Try your own query
Call `show(pose_i, row, col, encoder)` (patch grid is 15 x 20). In live JupyterLab you can uncomment the `interact`
line for sliders; it is off by default because widget messages stall headless execution (`nbconvert --execute`)."""),
    code("""def show(pose_i=0, row=7, col=10, encoder=list(ENCODERS)[-1]):
    f = feats[encoder]
    maps = X.similarity(f.patches[0, pose_i], row, col, f.patches[:, pose_i], (H, W))
    fig = X.grid([list(imgs[:, pose_i]), list(maps)], ["image", encoder], VIEWS, cmap="magma", vmin=0, vmax=1)
    for ax in fig.axes[: 2 * len(VIEWS) : len(VIEWS)]:
        ax.add_patch(plt.Rectangle((col * 16, row * 16), 16, 16, fill=False, ec="red", lw=1.5))
    plt.show()

show(1, 9, 5)
# from ipywidgets import interact, IntSlider
# interact(show, pose_i=IntSlider(0, 0, len(POSES) - 1), row=IntSlider(7, 0, H // 16 - 1), col=IntSlider(10, 0, W // 16 - 1), encoder=list(ENCODERS));"""),
    md("""## 4. Linear probe readouts: what is decodable, and where it breaks

Probes are fit on the canonical view of 600 training-region poses (as in `visenc eval`) and applied to every view of
the shown pose. Columns: image, predicted / true segmentation, predicted / true depth, absolute log-depth error
(bright = wrong; sky patches excluded from depth)."""),
    code("""gt_d_patch, gt_s_patch = X.patch_truth(split, POSES)
probe_out = {}
for name, spec in ENCODERS.items():
    enc = X.Encoder(spec)
    probes = X.DenseProbes(enc, split, fit_index)
    enc.release()
    f = feats[name].patches
    probe_out[name] = probes.predict(f.reshape(-1, *f.shape[2:]))   # depth, seg for every (view, pose)
    print("fitted", name)

def readout(name, p_i=0):
    depth, seg = (a.reshape(len(VIEWS), len(POSES), *a.shape[1:]) for a in probe_out[name])
    rows = []
    sky = gt_s_patch[p_i] == 0
    for v in range(len(VIEWS)):
        err = np.abs(np.log(depth[v, p_i]) - np.log(gt_d_patch[p_i]))
        err[sky] = 0
        pred_d = np.where(sky, 1e6, depth[v, p_i])   # sky has no depth target: draw it black like ground truth
        rows.append([imgs[v, p_i], X.colour_seg(X.upsample(seg[v, p_i : p_i + 1], (H, W))[0]),
                     X.colour_seg(X.upsample(gt_s_patch[p_i : p_i + 1], (H, W))[0]),
                     X.colour_depth(X.upsample(pred_d[None], (H, W))[0]),
                     X.colour_depth(X.upsample(gt_d_patch[p_i : p_i + 1], (H, W))[0]),
                     (np.clip(X.upsample(err[None], (H, W))[0] / 1.0, 0, 1) * 255).astype(np.uint8)])
    X.grid(rows, VIEWS, ["image", "pred seg", "true seg", "pred depth", "true depth", "|log depth error|"], cmap="inferno") \\
        .suptitle(f"{name}: pose {POSES[p_i]}", y=1.01)

for name in ["LingBot L (teacher)", "Ours: baseline ViT-S"]:
    readout(name, 0)"""),
    code("""# Per-view summary for the shown poses: segmentation accuracy and median |log depth error|.
import pandas as pd
rows = []
for name in ENCODERS:
    depth, seg = (a.reshape(len(VIEWS), len(POSES), *a.shape[1:]) for a in probe_out[name])
    for v, view in enumerate(VIEWS):
        solid = gt_s_patch != 0
        rows.append({"encoder": name, "view": view,
                     "seg acc": float((seg[v] == gt_s_patch).mean()),
                     "median |log depth err|": float(np.median(np.abs(np.log(depth[v][solid]) - np.log(gt_d_patch[solid]))))})
pd.DataFrame(rows).pivot(index="encoder", columns="view").round(3)"""),
    md("""## 5. Global map: does the global vector know where and how high the drone is?

CLS + mean-patch vectors for 800 poses in the canonical and `novel1` views, projected with one 2-D PCA per encoder.
Colour = drone height. Lines join the two views of the same pose: short lines mean the global vector barely moves
under a held-out appearance change."""),
    code("""many = np.random.default_rng(0).choice(np.flatnonzero(outside), 800, replace=False)
height = state[many, STATE_NAMES.index("pos_y")]
fig, axes = plt.subplots(1, len(ENCODERS), figsize=(4.2 * len(ENCODERS), 4))
for ax, (name, spec) in zip(axes, ENCODERS.items()):
    enc = X.Encoder(spec)
    g = [torch.cat([o.cls, o.patches.mean((1, 2))], 1) for o in (enc(np.stack([split.rgb[v][i] for i in many])) for v in ("canonical", "novel1"))]
    enc.release()
    allg = torch.cat(g)
    mean = allg.mean(0)
    _, _, V = torch.pca_lowrank(allg - mean, q=2, center=False)
    a, b = ((t - mean) @ V for t in g)
    for k in range(0, len(many), 4):
        ax.plot([a[k, 0], b[k, 0]], [a[k, 1], b[k, 1]], c="0.8", lw=0.5, zorder=0)
    sc = ax.scatter(a[:, 0], a[:, 1], c=height, s=6, cmap="viridis", label="canonical")
    ax.scatter(b[:, 0], b[:, 1], c=height, s=6, cmap="viridis", marker="x", label="novel1")
    shift = (a - b).norm(dim=1).mean() / a.std(0).norm()
    ax.set_title(f"{name}\\nmean view shift / spread = {shift:.2f}", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
axes[0].legend(fontsize=8)
fig.colorbar(sc, ax=axes, label="height (m)", shrink=0.8);"""),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
out = Path(__file__).with_name("explore.ipynb")
nbf.write(nb, out)
print(f"wrote {out}")
