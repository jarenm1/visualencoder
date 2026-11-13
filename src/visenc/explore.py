"""Helpers for notebooks/explore.ipynb: look inside frozen encoders on rendered views.

    PCA-RGB      patch features projected to 3 colours with one basis shared across views
    similarity   cosine map of one query patch against every patch of other views
    probe maps   linear depth / segmentation readouts drawn as images next to ground truth
    global map   2-D PCA of global features over many poses, coloured by ground-truth state

Everything takes uint8 images [N,H,W,3] straight from a data split.
"""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from .data import Split
from .evaluate import SKY, Ridge, dense_targets
from .models import PATCH, load_encoder

# Colours for the semantic classes in layout.json order (sky, ground, road, building, vegetation, vehicle, prop).
PALETTE = np.array([[135, 190, 235], [110, 160, 70], [120, 120, 120], [200, 80, 60], [30, 150, 40], [60, 80, 220], [230, 200, 40]], np.uint8)


@dataclass
class Features:
    cls: torch.Tensor  # [N,D]
    patches: torch.Tensor  # [N,h,w,D]


class Encoder:
    """One frozen encoder on the GPU; call .release() before loading a large one next."""

    def __init__(self, spec: str, device: str = "cuda", batch_size: int = 32):
        self.spec, self.device, self.batch_size = spec, torch.device(device), batch_size
        self.model = load_encoder(spec).to(self.device).eval()

    @torch.no_grad()
    def __call__(self, images: np.ndarray) -> Features:
        h, w = images.shape[1] // PATCH, images.shape[2] // PATCH
        cls, patches = [], []
        for s in range(0, len(images), self.batch_size):
            x = torch.from_numpy(np.ascontiguousarray(images[s : s + self.batch_size])).permute(0, 3, 1, 2)
            x = x.to(self.device).float().div_(255)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=getattr(self.model, "autocast", True)):
                c, p = self.model(x)
            cls.append(c.float().cpu())
            patches.append(p.float().cpu().unflatten(1, (h, w)))
        return Features(torch.cat(cls), torch.cat(patches))

    def release(self) -> None:
        del self.model
        torch.cuda.empty_cache()


def views_of(split: Split, index: np.ndarray | list[int], views: list[str]) -> np.ndarray:
    """uint8 [len(views), len(index), H, W, 3]."""
    return np.stack([np.stack([split.rgb[v][i] for i in index]) for v in views])


def pca_rgb(feature_sets: list[torch.Tensor], size: tuple[int, int], basis_from: torch.Tensor | None = None) -> list[np.ndarray]:
    """Project several [N,h,w,D] patch sets with ONE PCA basis to RGB images [N,H,W,3] in [0,1].
    The basis is fit on `basis_from` (any [..., D] patches, e.g. many poses) or else on the sets
    themselves. Shared basis: equal colours mean equal features across sets."""
    flat = torch.cat([f.reshape(-1, f.shape[-1]) for f in feature_sets])
    fit = flat if basis_from is None else basis_from.reshape(-1, basis_from.shape[-1])
    mean = fit.mean(0)
    _, _, v = torch.pca_lowrank(fit - mean, q=3, center=False)
    ref = (fit - mean) @ v
    lo, hi = ref.quantile(0.01, 0), ref.quantile(0.99, 0)
    proj = (flat - mean) @ v
    out, start = [], 0
    for f in feature_sets:
        n = f.shape[0] * f.shape[1] * f.shape[2]
        rgb = ((proj[start : start + n] - lo) / (hi - lo)).clamp(0, 1).reshape(*f.shape[:3], 3)
        rgb = F.interpolate(rgb.permute(0, 3, 1, 2), size=size, mode="nearest").permute(0, 2, 3, 1)
        out.append(rgb.numpy())
        start += n
    return out


def similarity(query: torch.Tensor, row: int, col: int, targets: torch.Tensor, size: tuple[int, int]) -> np.ndarray:
    """Cosine similarity of query[row, col] ([h,w,D] patch grid) against every patch of targets [N,h,w,D];
    returns [N,H,W] upsampled maps."""
    q = F.normalize(query[row, col], dim=-1)
    sim = torch.einsum("nhwd,d->nhw", F.normalize(targets, dim=-1), q)
    return F.interpolate(sim[:, None], size=size, mode="nearest")[:, 0].numpy()


class DenseProbes:
    """Linear per-patch depth (log metres) and segmentation probes, fit on one view of `fit_index`
    exactly as in `visenc eval` (ridge, lambda chosen on a validation slice, sky patches skipped for depth)."""

    def __init__(self, enc: Encoder, split: Split, fit_index: np.ndarray, view: str = "canonical", per_image: int = 64):
        self.n_classes = len(split.meta["classes"])
        feats = enc(np.stack([split.rgb[view][i] for i in fit_index])).patches.flatten(1, 2)  # [n,P,D]
        frac, logd = dense_targets(split, fit_index, self.n_classes)
        rng = np.random.default_rng(0)
        keep = torch.from_numpy(np.stack([rng.choice(feats.shape[1], per_image, replace=False) for _ in fit_index]))
        x = torch.gather(feats, 1, keep[..., None].expand(-1, -1, feats.shape[-1])).reshape(-1, feats.shape[-1])
        f = torch.gather(frac, 1, keep[..., None].expand(-1, -1, self.n_classes)).reshape(-1, self.n_classes)
        d = torch.gather(logd, 1, keep).reshape(-1, 1)
        solid = f[:, SKY] < 0.5
        self.depth = Ridge(x[solid], d[solid], enc.device)
        self.semantic = Ridge(x, f, enc.device)

    @torch.no_grad()
    def predict(self, patches: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """patches [N,h,w,D] -> (depth metres [N,h,w], class ids [N,h,w])."""
        n, h, w, d = patches.shape
        flat = patches.reshape(-1, d)
        depth = self.depth(flat).exp().reshape(n, h, w).cpu().numpy()
        seg = self.semantic(flat).argmax(-1).reshape(n, h, w).cpu().numpy()
        return depth, seg


def patch_truth(split: Split, index: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Ground truth at patch resolution: (mean depth metres [N,h,w], majority class [N,h,w])."""
    frac, logd = dense_targets(split, np.asarray(index), len(split.meta["classes"]))
    h, w = split.size[0] // PATCH, split.size[1] // PATCH
    return logd.exp().reshape(-1, h, w).numpy(), frac.argmax(-1).reshape(-1, h, w).numpy()


def colour_seg(seg: np.ndarray) -> np.ndarray:
    return PALETTE[np.minimum(seg, len(PALETTE) - 1)]


def colour_depth(depth: np.ndarray, near: float = 2.0, far: float = 400.0) -> np.ndarray:
    """Log-scaled depth through the turbo colour map (near = red, far = blue); beyond `far` (incl. sky) is black."""
    v = np.clip((np.log(np.maximum(depth, near)) - np.log(near)) / (np.log(far) - np.log(near)), 0, 1)
    rgb = (plt.get_cmap("turbo")(1 - v)[..., :3] * 255).astype(np.uint8)
    rgb[depth > far] = 0
    return rgb


def upsample(grid: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour [N,h,w(,C)] -> [N,H,W(,C)]."""
    ry, rx = size[0] // grid.shape[1], size[1] // grid.shape[2]
    return grid.repeat(ry, 1).repeat(rx, 2)


def grid(rows: list[list[np.ndarray]], row_titles: list[str], col_titles: list[str], scale: float = 2.2, **imshow):
    """Show a rows x cols image grid with titles."""
    fig, axes = plt.subplots(len(rows), len(rows[0]), figsize=(scale * 1.33 * len(rows[0]), scale * len(rows)), squeeze=False)
    for r, row in enumerate(rows):
        for c, img in enumerate(row):
            ax = axes[r][c]
            ax.imshow(img, **imshow)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(col_titles[c], fontsize=9)
            if c == 0:
                ax.set_ylabel(row_titles[r], fontsize=9)
    fig.tight_layout()
    return fig
