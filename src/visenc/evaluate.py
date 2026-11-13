"""Frozen-encoder evaluation against simulator ground truth.

All probes are linear (closed-form ridge, regularisation picked on a validation slice) so
scores measure what the frozen features expose, not what a probe can learn. Probes are fit
on one appearance view (default "canonical") and tested on every view of held-out poses:
the gap between the fit view and the random views is the appearance-shift penalty, our
in-simulator proxy for sim-to-real.

Metrics per encoder:
  state      R^2 of pose/clearance from the global feature (CLS ++ mean patch), per group
  depth      per-patch log-depth regression: AbsRel and RMSE(log)
  semantic   per-patch class prediction: mIoU over classes present
  pairs      same pose, different appearance (no fitting): retrieval@1 among test poses
             and mean centred cosine of global and patch features
  region     optional spatially held-out region: state/depth/semantic on poses there
  real       optional unlabelled real images: linear sim-vs-real domain accuracy
             (lower = less separable; read together with probe scores to rule out collapse)
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .city import STATE_GROUPS, STATE_NAMES
from .data import ImageFolder, Split
from .models import PATCH, load_encoder

LAMBDAS = [10.0**k for k in range(-4, 4)]
TRAIN_PATCHES_PER_IMAGE = 64
SKY = 0  # class id of background/sky in layout.json


class Ridge:
    """Standardised closed-form ridge; lambda chosen on the last `val_frac` of the fit rows."""

    def __init__(self, x: torch.Tensor, y: torch.Tensor, device: torch.device, val_frac: float = 0.2):
        x, y = x.to(device, torch.float32), y.to(device, torch.float32)
        self.mx, self.sx = x.mean(0), x.std(0).clamp_min(1e-6)
        self.my, self.sy = y.mean(0), y.std(0).clamp_min(1e-6)
        xs, ys = (x - self.mx) / self.sx, (y - self.my) / self.sy
        n_val = max(1, int(len(xs) * val_frac))
        fit_x, fit_y, val_x, val_y = xs[:-n_val], ys[:-n_val], xs[-n_val:], ys[-n_val:]
        best = min(LAMBDAS, key=lambda lam: ((val_x @ self._solve(fit_x, fit_y, lam) - val_y) ** 2).mean().item())
        self.lam = best
        self.w = self._solve(xs, ys, best)

    @staticmethod
    def _solve(x: torch.Tensor, y: torch.Tensor, lam: float) -> torch.Tensor:
        xtx = (x.T @ x).double()
        xtx.diagonal().add_(lam * len(x))
        return torch.linalg.solve(xtx, (x.T @ y).double()).float()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.w.device, torch.float32)
        return ((x - self.mx) / self.sx) @ self.w * self.sy + self.my


def r2(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    y = y.to(pred.device)
    sst = ((y - y.mean(0)) ** 2).sum(0)
    return 1 - ((pred - y) ** 2).sum(0) / sst.clamp_min(1e-12)


def miou(pred: torch.Tensor, gt: torch.Tensor, n_classes: int) -> float:
    ious = []
    for c in range(n_classes):
        union = ((pred == c) | (gt == c)).sum().item()
        if (gt == c).any():
            ious.append(((pred == c) & (gt == c)).sum().item() / union)
    return float(np.mean(ious))


class Features:
    """Global features for every sample of each view; patch features for chosen indices only."""

    def __init__(self, encoder, split: Split, device: torch.device, batch_size: int):
        self.encoder, self.split, self.device, self.batch_size = encoder, split, device, batch_size

    @torch.no_grad()
    def extract(self, images, indices: np.ndarray, patch_keep: dict[int, np.ndarray | None]):
        """images: uint8 array-like [N,H,W,3] or Dataset of [1,3,H,W]. patch_keep: index -> patch
        positions to keep (None = all); indices absent from patch_keep keep no patches."""
        globals_, patches = [], {}
        for start in range(0, len(indices), self.batch_size):
            idx = indices[start : start + self.batch_size]
            if isinstance(images, np.ndarray):
                batch = torch.from_numpy(np.stack([images[i] for i in idx])).permute(0, 3, 1, 2)
            else:
                batch = torch.cat([images[i] for i in idx])
            x = batch.to(self.device).float().div_(255)
            # Precision wrappers (fp16:, int8:) run their own numerics; everything else uses bf16 autocast.
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=getattr(self.encoder, "autocast", True)):
                cls, tok = self.encoder(x)
            cls, tok = cls.float(), tok.float()
            globals_.append(torch.cat([cls, tok.mean(1)], 1).cpu())
            for row, i in enumerate(idx):
                if int(i) in patch_keep:
                    keep = patch_keep[int(i)]
                    patches[int(i)] = (tok[row] if keep is None else tok[row, keep]).half().cpu()
        return torch.cat(globals_), patches


def dense_targets(split: Split, indices: np.ndarray, n_classes: int, chunk: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-patch class fractions [n,P,C] and mean log-depth [n,P], patch order row-major.
    Chunked: a full-resolution one-hot of every image at once does not fit in RAM at 640x480."""
    fracs, logds = [], []
    for s in range(0, len(indices), chunk):
        idx = indices[s : s + chunk]
        sem = torch.from_numpy(np.stack([split.semantic[i] for i in idx])).long()
        depth = torch.from_numpy(np.stack([split.depth[i] for i in idx]).astype(np.float32))
        fracs.append(F.avg_pool2d(F.one_hot(sem, n_classes).permute(0, 3, 1, 2).float(), PATCH).flatten(2).transpose(1, 2))
        logds.append(F.avg_pool2d(depth.clamp_min(0.1).log()[:, None], PATCH).flatten(1))
    return torch.cat(fracs), torch.cat(logds)


def in_region(state: np.ndarray, region: list[float] | None) -> np.ndarray:
    if region is None:
        return np.zeros(len(state), dtype=bool)
    x0, z0, x1, z1 = region
    x, z = state[:, STATE_NAMES.index("pos_x")], state[:, STATE_NAMES.index("pos_z")]
    return (x >= x0) & (x <= x1) & (z >= z0) & (z <= z1)


def evaluate_encoder(spec: str, split: Split, args, device: torch.device) -> dict:
    encoder = load_encoder(spec).to(device).eval()
    feats = Features(encoder, split, device, args.batch_size)
    n_classes = len(split.meta["classes"])
    state = np.asarray(split.state)
    region = in_region(state, args.heldout_region)
    n = len(split)
    n_fit = int(n * args.fit_frac)
    order = np.arange(n)
    fit_idx = order[(order < n_fit) & ~region]
    test_idx = order[(order >= n_fit) & ~region]
    region_idx = order[region]
    rng = np.random.default_rng(0)
    h, w = split.size
    n_patches = (h // PATCH) * (w // PATCH)
    fit_keep = {int(i): np.sort(rng.choice(n_patches, TRAIN_PATCHES_PER_IMAGE, replace=False)) for i in fit_idx}

    # Fit probes on the fit view.
    g_fit, p_fit = feats.extract(split.rgb[args.fit_view], fit_idx, fit_keep)
    y_state = torch.from_numpy(state[fit_idx])
    state_probe = Ridge(g_fit, y_state, device)
    frac, logd = dense_targets(split, fit_idx, n_classes)
    rows = torch.cat([p_fit[int(i)] for i in fit_idx])
    pos = torch.from_numpy(np.stack([fit_keep[int(i)] for i in fit_idx]))
    sel = lambda t: torch.gather(t, 1, pos if t.ndim == 2 else pos[..., None].expand(-1, -1, t.shape[-1]))  # noqa: E731
    # Depth is probed and scored only on patches that are mostly geometry (sky depth is the far plane).
    solid = sel(frac)[..., SKY] < 0.5
    depth_probe = Ridge(rows[solid.reshape(-1)], sel(logd)[solid].reshape(-1, 1), device)
    sem_probe = Ridge(rows, sel(frac).reshape(-1, n_classes), device)
    del rows, p_fit

    def score(view: str, idx: np.ndarray) -> tuple[dict, tuple[torch.Tensor, torch.Tensor]]:
        g, p = feats.extract(split.rgb[view], idx, {int(i): None for i in idx})
        r = r2(state_probe(g), torch.from_numpy(state[idx]))
        frac_t, logd_t = dense_targets(split, idx, n_classes)
        tok = torch.stack([p[int(i)] for i in idx])  # [n,P,D]
        pred_logd = torch.cat([depth_probe(t) for t in tok]).view(len(idx), -1).cpu()
        valid = frac_t[..., SKY] < 0.5
        pred_d, true_d = pred_logd[valid], logd_t[valid]
        pred_cls = torch.cat([sem_probe(t).argmax(-1) for t in tok]).view(len(idx), -1).cpu()
        out = {
            "state_r2": r.mean().item(),
            **{f"state_r2_{k}": r[s].mean().item() for k, s in STATE_GROUPS.items()},
            "state_r2_per_dim": dict(zip(STATE_NAMES, r.tolist())),
            "depth_absrel": ((pred_d.exp() - true_d.exp()).abs() / true_d.exp()).mean().item(),
            "depth_rmse_log": ((pred_d - true_d) ** 2).mean().sqrt().item(),
            "semantic_miou": miou(pred_cls, frac_t.argmax(-1), n_classes),
        }
        return out, (g, tok)

    result: dict = {"spec": spec, "probe_lambda": {"state": state_probe.lam, "depth": depth_probe.lam, "semantic": sem_probe.lam}}
    result["fit_view"] = args.fit_view
    result["n"] = {"fit": len(fit_idx), "test": len(test_idx), "region": len(region_idx)}
    views_out, cache = {}, {}
    for view in split.views:
        views_out[view], cache[view] = score(view, test_idx)
    result["views"] = views_out
    if len(region_idx):
        result["region"] = {view: score(view, region_idx)[0] for view in split.views}

    # Pair metrics: same pose, other appearance. Centre with the fit-view mean so the shared
    # mean direction does not inflate cosine.
    g_mean, p_mean = g_fit.mean(0), cache[args.fit_view][1].float().mean((0, 1))
    pairs = {}
    ga, ta = cache[args.fit_view]
    for view in split.views:
        if view == args.fit_view:
            continue
        gb, tb = cache[view]
        qa, qb = F.normalize(ga - g_mean, dim=1), F.normalize(gb - g_mean, dim=1)
        sim = qa @ qb.T
        pairs[view] = {
            "retrieval_at_1": (sim.argmax(1) == torch.arange(len(sim))).float().mean().item(),
            "global_cosine": sim.diagonal().mean().item(),
            "patch_cosine": F.cosine_similarity(ta.float() - p_mean, tb.float() - p_mean, dim=-1).mean().item(),
        }
    result["pairs"] = pairs

    if args.real:
        real = ImageFolder(args.real, split.size)
        k = min(len(real), len(test_idx))
        g_real, _ = feats.extract(real, np.arange(k), {})
        g_sim = ga[:k]
        x = torch.cat([g_sim, g_real])
        y = torch.cat([torch.zeros(k), torch.ones(k)])[:, None]
        perm = torch.from_numpy(np.random.default_rng(1).permutation(2 * k))
        x, y = x[perm], y[perm]
        half = k
        clf = Ridge(x[:half], y[:half], device)
        result["real_domain_accuracy"] = ((clf(x[half:]) > 0.5).float().cpu() == y[half:]).float().mean().item()
    return result


def view_group(view: str) -> str:
    """Appearance shift group of a non-fit view: 'seen' (training distribution) or 'novel' (held out)."""
    return "novel" if view.startswith("novel") else "seen"


def summary_table(results: list[dict], fit_view: str) -> str:
    others = sorted({v for r in results for v in r["views"] if v != fit_view})
    groups = [g for g in ("seen", "novel") if any(view_group(v) == g for v in others)]
    cols = ["encoder"]
    for metric in ("state R²", "depth AbsRel", "seg mIoU"):
        cols += [f"{metric} {fit_view}", *(f"{metric} {g}" for g in groups)]
    cols += [f"{m} {g}" for m in ("retrieval@1", "patch cos") for g in groups]
    if any("real_domain_accuracy" in r for r in results):
        cols.append("real domain acc")
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in results:
        base = r["views"][fit_view]

        def mean(table: dict, key: str, group: str) -> float:
            vals = [table[v][key] for v in others if view_group(v) == group and v in table]
            return float(np.mean(vals)) if vals else math.nan

        row: list = [r["spec"]]
        for key in ("state_r2", "depth_absrel", "semantic_miou"):
            row += [base[key], *(mean(r["views"], key, g) for g in groups)]
        row += [mean(r["pairs"], key, g) for key in ("retrieval_at_1", "patch_cosine") for g in groups]
        if "real domain acc" in cols:
            row.append(r.get("real_domain_accuracy", math.nan))
        lines.append("| " + " | ".join(c if isinstance(c, str) else f"{c:.3f}" for c in row) + " |")
    return "\n".join(lines)


def run(args) -> None:
    device = torch.device("cuda")
    split = Split(args.data)
    if args.fit_view not in split.views:
        raise ValueError(f"fit view {args.fit_view!r} not in {split.views}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for spec in args.encoders:
        print(f"evaluating {spec}", flush=True)
        results.append(evaluate_encoder(spec, split, args, device))
        torch.cuda.empty_cache()
        (out / "report.json").write_text(json.dumps(results, indent=2))
    table = summary_table(results, args.fit_view)
    (out / "report.md").write_text(table + "\n")
    print(table)


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data", required=True, help="evaluation split (canonical + random views)")
    p.add_argument("--encoders", nargs="+", required=True, help="specs: pixels | hf:<id> | random:<id> | run:<dir>")
    p.add_argument("--fit-view", default="canonical")
    p.add_argument("--fit-frac", type=float, default=0.8, help="leading fraction of poses used to fit probes")
    p.add_argument("--heldout-region", type=float, nargs=4, metavar=("X0", "Z0", "X1", "Z1"),
                   help="report this world rectangle separately and exclude it from probe fitting")  # fmt: skip
    p.add_argument("--real", help="folder of unlabelled real images for the domain-separability metric")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--out", required=True, help="report directory")
