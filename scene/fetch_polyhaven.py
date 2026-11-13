"""Fetch the CC0 Poly Haven textures used by the renderer (1k diffuse + OpenGL normal).

    python scene/fetch_polyhaven.py assets/polyhaven

Writes <out>/<id>/{diff,nor}.jpg and <out>/materials.json:
    {"roles": {role: [id, ...]}, "heldout": {role: [id, ...]}, "textures": {id: {"size_m": [w, h]}}}
Roles are the texture pools the renderer draws appearance variants from. "heldout" textures are
eval-only: training data never uses them (Appearance regime "novel" draws only these).
"""

import json
import sys
import urllib.request
from pathlib import Path

ROLES = {
    "grass": ["leafy_grass", "sparse_grass", "grass_path_2", "aerial_grass_rock", "withered_grass"],
    "dirt": ["park_dirt", "dirt", "forest_ground_04", "stony_dirt_path"],
    "asphalt": ["asphalt_02", "asphalt_04", "clean_asphalt", "road_damaged_clean", "aerial_asphalt_01"],
    "pavement": ["concrete_tiles", "rectangular_paving", "hexagonal_concrete_paving", "square_brick_paving", "precast_stone_paving"],
    "wall": ["concrete_block_wall", "painted_plaster_wall", "grey_plaster", "rectangular_facade_tiles", "concrete_tile_facade", "plaster_brick_01", "brick_4"],
    "roof": ["clay_roof_tiles", "clay_roof_tiles_02", "box_profile_metal_sheet", "concrete_floor"],
    "foliage": ["forest_leaves_02", "forest_leaves_03", "forest_leaves_04"],
    "bark": ["bark_brown_01", "bark_willow", "bark_platanus"],
    "metal": ["metal_plate", "painted_metal_shutter", "blue_metal_plate"],
    "concrete": ["brushed_concrete", "chipped_concrete", "dirty_concrete"],
}
# Eval-only texture(s) per role: the last entry (two for walls, the largest pool).
HELDOUT = {role: ids[-2:] if role == "wall" else ids[-1:] for role, ids in ROLES.items()}
API = "https://api.polyhaven.com"
HEADERS = {"User-Agent": "visenc-asset-fetch"}


def _get(url: str) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=60) as r:
        return r.read()


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "assets/polyhaven")
    out.mkdir(parents=True, exist_ok=True)
    textures = {}
    for ids in ROLES.values():
        for tid in ids:
            if tid in textures:
                continue
            info = json.loads(_get(f"{API}/info/{tid}"))
            if info.get("type") != 1:
                raise ValueError(f"{tid} is not a Poly Haven texture")
            files = json.loads(_get(f"{API}/files/{tid}"))
            d = out / tid
            d.mkdir(exist_ok=True)
            for key, name in (("Diffuse", "diff.jpg"), ("nor_gl", "nor.jpg")):
                if not (d / name).exists():
                    (d / name).write_bytes(_get(files[key]["1k"]["jpg"]["url"]))
            w, h = info.get("dimensions") or [2000, 2000]
            textures[tid] = {"size_m": [w / 1000.0, h / 1000.0]}
            print(f"fetched {tid}", flush=True)
    (out / "materials.json").write_text(json.dumps({"roles": ROLES, "heldout": HELDOUT, "textures": textures}, indent=1))


if __name__ == "__main__":
    main()
