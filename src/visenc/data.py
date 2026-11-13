"""Dataset generation from the single city scene, packed into memory-mapped .npy arrays.

Each sample is one camera pose. Geometry passes (semantic, depth) are rendered once; RGB
is rendered once per appearance view. Seeds derive from (seed, sample index[, view]),
so output is independent of worker count.

Split layout (N samples, H x W):
    rgb_<view>.npy  uint8   [N,H,W,3]  views: "canonical", "random1".."randomK" (seen regime),
                                       "novel1".."novelM" (eval-only: held-out textures and lighting)
    semantic.npy    uint8   [N,H,W]    class ids, see meta["classes"]
    depth.npy       float16 [N,H,W]    optical-Z metres (sky = far plane)
    state.npy       float32 [N,S]      see visenc.city.STATE_NAMES
    meta.json
"""

import json
import multiprocessing as mp
import queue
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from .city import FAR, STATE_NAMES, Appearance, CityLayout, CityRenderer, pose_state

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
NEAR_WALL_M = 2.0  # reject poses where too much of the view is closer than this
NEAR_WALL_FRACTION = 0.4


def view_names(random_views: int, novel_views: int = 0) -> list[str]:
    return ["canonical"] + [f"random{k}" for k in range(1, random_views + 1)] + [f"novel{k}" for k in range(1, novel_views + 1)]


def view_appearance(view: str, seed: int, index: int, k: int) -> Appearance:
    if view == "canonical":
        return Appearance.canonical()
    regime = "novel" if view.startswith("novel") else "seen"
    return Appearance.random(np.random.default_rng([seed, index, k]), regime)


def _in_region(p: np.ndarray, region: tuple[float, float, float, float] | None) -> bool:
    if region is None:
        return False
    x0, z0, x1, z1 = region
    return x0 <= p[0] <= x1 and z0 <= p[2] <= z1


def _worker(job: dict) -> int:
    out = Path(job["out"])
    views = job["views"]
    mm = {
        "semantic": np.load(out / "semantic.npy", mmap_mode="r+"),
        "depth": np.load(out / "depth.npy", mmap_mode="r+"),
        "state": np.load(out / "state.npy", mmap_mode="r+"),
        **{v: np.load(out / f"rgb_{v}.npy", mmap_mode="r+") for v in views},
    }
    region = job["exclude_region"]
    with CityRenderer(Path(job["scene"]), job["width"], job["height"]) as renderer:
        for i in range(job["start"], job["stop"]):
            rng = np.random.default_rng([job["seed"], i])
            while True:
                pose = renderer.layout.sample_pose(rng)
                if _in_region(pose.position, region):
                    continue
                semantic, depth = renderer.render_geometry(pose)
                if (depth < NEAR_WALL_M).mean() <= NEAR_WALL_FRACTION:
                    break
            mm["semantic"][i] = semantic
            mm["depth"][i] = depth
            mm["state"][i] = pose_state(pose, renderer.layout.clearance(pose.position), depth)
            for k, v in enumerate(views):
                look = view_appearance(v, job["seed"], i, k)
                mm[v][i] = renderer.render_rgb(pose, look)
            job["progress"].put(1)
    for a in mm.values():
        a.flush()
    return job["stop"] - job["start"]


def generate(
    scene: Path,
    out: Path,
    seed: int,
    count: int,
    width: int,
    height: int,
    random_views: int,
    novel_views: int,
    workers: int,
    exclude_region: tuple[float, float, float, float] | None,
) -> None:
    if width % 16 or height % 16:
        raise ValueError("width and height must be multiples of the ViT patch size (16)")
    out.mkdir(parents=True, exist_ok=False)
    views = view_names(random_views, novel_views)
    shapes = {
        "semantic": ((count, height, width), np.uint8),
        "depth": ((count, height, width), np.float16),
        "state": ((count, len(STATE_NAMES)), np.float32),
        **{f"rgb_{v}": ((count, height, width, 3), np.uint8) for v in views},
    }
    for name, (shape, dtype) in shapes.items():
        np.lib.format.open_memmap(out / f"{name}.npy", "w+", dtype, shape).flush()

    ctx = mp.get_context("spawn")  # each worker owns its own GL context
    manager = ctx.Manager()
    progress = manager.Queue()
    bounds = np.linspace(0, count, workers + 1).astype(int)
    jobs = [
        dict(out=str(out), scene=str(scene), seed=seed, start=int(a), stop=int(b), width=width, height=height,
             views=views, exclude_region=exclude_region, progress=progress)
        for a, b in zip(bounds[:-1], bounds[1:]) if b > a
    ]  # fmt: skip
    with ctx.Pool(len(jobs)) as pool, tqdm(total=count, desc=f"generate {out.name}", mininterval=5.0) as bar:
        result = pool.map_async(_worker, jobs)
        done = 0
        while done < count:
            if result.ready() and not result.successful():
                result.get()  # re-raise the worker exception
            try:
                done += progress.get(timeout=1.0)
                bar.update()
            except queue.Empty:
                continue
        result.get()

    layout = CityLayout(scene)
    meta = {
        "count": count,
        "width": width,
        "height": height,
        "seed": seed,
        "views": views,
        "classes": layout.classes,
        "state_names": STATE_NAMES,
        "far_m": FAR,
        "scene": str(scene),
        "exclude_region": exclude_region,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))


class Split:
    """Read-only view of a generated split; arrays are memory-mapped."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.meta = json.loads((self.root / "meta.json").read_text())
        self.views: list[str] = self.meta["views"]
        self.state = np.load(self.root / "state.npy", mmap_mode="r")
        self.depth = np.load(self.root / "depth.npy", mmap_mode="r")
        self.semantic = np.load(self.root / "semantic.npy", mmap_mode="r")
        self.rgb = {v: np.load(self.root / f"rgb_{v}.npy", mmap_mode="r") for v in self.views}

    def __len__(self) -> int:
        return self.meta["count"]

    @property
    def size(self) -> tuple[int, int]:
        return self.meta["height"], self.meta["width"]


class PairedViews(Dataset):
    """Yields uint8 [V,3,H,W]: the requested appearance views of one pose."""

    def __init__(self, split: Split, views: list[str]):
        missing = set(views) - set(split.views)
        if missing:
            raise ValueError(f"{split.root} lacks views {sorted(missing)}; has {split.views}")
        self.split, self.views = split, views

    def __len__(self) -> int:
        return len(self.split)

    def __getitem__(self, i: int) -> torch.Tensor:
        views = [torch.from_numpy(np.array(self.split.rgb[v][i])) for v in self.views]
        return torch.stack(views).permute(0, 3, 1, 2)


class ImageFolder(Dataset):
    """Unlabelled images (e.g. real camera frames), resized to cover then centre-cropped to H x W."""

    def __init__(self, root: Path, size: tuple[int, int]):
        self.paths = sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
        if not self.paths:
            raise ValueError(f"no images under {root}")
        self.size = size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> torch.Tensor:
        h, w = self.size
        im = Image.open(self.paths[i]).convert("RGB")
        scale = max(h / im.height, w / im.width)
        im = im.resize((max(w, round(im.width * scale)), max(h, round(im.height * scale))), Image.BICUBIC)
        left, top = (im.width - w) // 2, (im.height - h) // 2
        im = im.crop((left, top, left + w, top + h))
        return torch.from_numpy(np.asarray(im)).permute(2, 0, 1)[None]  # [1,3,H,W]
