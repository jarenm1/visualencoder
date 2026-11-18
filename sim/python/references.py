"""Reference library: one goal image per object type, for category goals (sim_rl/category.py).

The commander names objects the drone has not seen ("go to the truck"). With no box to crop, the
agent sets the named type's library reference as the goal: the pilot searches by type, and once
Gemma can box the instance its crop replaces the reference (success is only ever the instance).

  build: for every object type (layout asset) in the training catalogue, render the goal view of
         its first entry with a fully visible start (the catalogue's own goal reference camera)
         and save the crop as <out>/<asset>.png with an index.json of {phrase: file}.

    nix develop path:sim --command .venv/bin/python sim/python/references.py --out runs/reference-library
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "sim/python"), str(ROOT / "src")]

from gemma_goal import object_phrase  # noqa: E402

VISIBLE = 0


def build(args):
    from sim_rl.cli import DEFAULT_LIBRARY, TRAIN_STYLE
    from sim_rl.curriculum import fixed_lanes
    from sim_rl.features import crop_goals
    from sim_rl.native import CudaEnv
    from sim_rl.verifier import catalog_entries
    args.out.mkdir(parents=True, exist_ok=True)
    with CudaEnv(DEFAULT_LIBRARY, args.num_envs, args.seed, 320, 240, TRAIN_STYLE, "velocity",
                 start_distance=(8.0, 60.0)) as env, env.execution():
        rows = catalog_entries(env, lambda done, total: None)
        layout = json.loads((env.city_path / "layout.json").read_text())["objects"]
        env.write_curriculum(fixed_lanes(env.device, appearance=0.0))   # plain render: a clean reference
        objects, buckets = rows["meta"][:, 0].astype(int), rows["meta"][:, 1].astype(int)
        index = {}
        for asset in sorted({layout[o]["asset"] for o in objects}):
            entries = np.flatnonzero((np.array([layout[o]["asset"] for o in objects]) == asset) & (buckets == VISIBLE))
            if not len(entries):
                continue
            env.render_goals(int(entries[0]))
            crop = crop_goals(env.goal_views[:1], env.goal_crops[:1])[0]
            image = (crop.clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
            name = re.sub(r"[^a-z0-9]+", "-", asset.lower()).strip("-") + ".png"
            Image.fromarray(image).save(args.out / name)
            index.setdefault(object_phrase(asset), []).append({"file": name, "asset": asset, "entry": int(entries[0])})
    (args.out / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    print(json.dumps({"types": len(index), "out": str(args.out)}))


class ReferenceLibrary:
    """Label -> reference image (float32 [1, 3, H, W] in [0, 1]) by phrase match."""

    def __init__(self, folder, device):
        self.folder, self.device = Path(folder), device
        self.index = json.loads((self.folder / "index.json").read_text())

    def match(self, label):
        """The phrase sharing the most words with ``label`` (exact phrases first), or None."""
        words = set(re.findall(r"[a-z]+", label.lower()))
        best, score = None, 0.0
        for phrase in self.index:
            terms = set(phrase.split())
            overlap = len(words & terms) / len(words | terms) if words else 0.0
            if phrase == label.lower().strip():
                overlap = 2.0
            if overlap > score:
                best, score = phrase, overlap
        return best

    def image(self, phrase):
        path = self.folder / self.index[phrase][0]["file"]
        array = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255
        return torch.as_tensor(array, device=self.device).permute(2, 0, 1)[None]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/reference-library")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
