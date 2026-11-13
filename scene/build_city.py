"""Assemble one large static city from CC0 Kenney kits and export it per semantic class.

Run headless:
    blender -b --python scene/build_city.py -- --assets assets/kenney --out assets/city

Layout: a square grid of city blocks separated by roads. Districts by distance from the
centre: downtown (commercial towers on pavement), suburbs (houses, yards, trees), and
parks/woods on the outskirts; street lights and cars along roads. Placement is seeded
and deterministic.

Outputs (glTF convention: metres, +Y up, ground at y=0):
    <out>/<class>.glb   one merged mesh per semantic class
    <out>/layout.json   classes, scene bounds, and every placed object's world AABB
"""

import argparse
import json
import math
import random
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector

TILE = 10.0  # metres per Kenney city tile
BLOCK = 4  # tiles per block period: 1 road + 3 interior
CLASSES = ["sky", "ground", "road", "building", "vegetation", "vehicle", "prop"]
SOLID = {"building", "vegetation", "vehicle", "prop"}  # cameras must stay outside these
FOREST_RING = 3  # tiles of dense trees around the city
GROUND_HALF_EXTENT = 2500.0  # metres; reaches the renderer's far plane from anywhere in the city
MAX_LOOPS_PER_MESH = 60000

KITS = {
    "commercial": "city-commercial/Models/GLB format",
    "suburban": "city-suburban/Models/GLB format",
    "roads": "city-roads/Models/GLB format",
    "nature": "nature/Models/GLTF format",
    "cars": "cars/Models/GLB format",
}
# Per-kit scale to metres (city kits are 1 unit = 1 tile; others are measured).
KIT_SCALE = {"commercial": TILE, "suburban": TILE, "roads": TILE, "nature": 7.0, "cars": 1.8}

DOWNTOWN = [f"building-{c}" for c in "abcdefghijklmn"]
TOWERS = [f"building-skyscraper-{c}" for c in "abcde"]
HOUSES = [f"building-type-{c}" for c in "abcdefghijklmnopqrstu"]
PARK_TREES = ["tree_oak", "tree_default", "tree_detailed", "tree_fat", "tree_pineRoundA", "tree_pineRoundC", "tree_cone", "tree_blocks"]
BUSHES = ["plant_bushLarge", "plant_bush", "plant_bushDetailed"]
ROCKS = ["rock_largeA", "rock_largeB", "rock_largeC", "stone_largeA", "stone_largeB"]
CARS = ["sedan", "suv", "taxi", "van", "hatchback-sports", "police", "delivery", "truck", "suv-luxury", "sedan-sports"]


class Library:
    """Imports each asset once as a hidden, transform-applied, single-mesh prototype."""

    def __init__(self, root: Path):
        self.root = root
        self.protos: dict[tuple[str, str], bpy.types.Object] = {}
        self.stash = bpy.data.collections.new("prototypes")

    def get(self, kit: str, name: str) -> bpy.types.Object:
        key = (kit, name)
        if key in self.protos:
            return self.protos[key]
        path = self.root / KITS[kit] / f"{name}.glb"
        before = set(bpy.data.objects)
        bpy.ops.import_scene.gltf(filepath=str(path))
        new = [o for o in bpy.data.objects if o not in before]
        meshes = [o for o in new if o.type == "MESH"]
        if not meshes:
            raise RuntimeError(f"{path} has no meshes")
        for o in meshes:
            world = o.matrix_world.copy()
            o.parent = None
            o.matrix_world = world
        for o in new:
            if o.type != "MESH":
                bpy.data.objects.remove(o)
        bpy.ops.object.select_all(action="DESELECT")
        for o in meshes:
            o.select_set(True)
        bpy.context.view_layer.objects.active = meshes[0]
        if len(meshes) > 1:
            bpy.ops.object.join()
        proto = bpy.context.view_layer.objects.active
        s = KIT_SCALE[kit]
        proto.scale = (s * proto.scale[0], s * proto.scale[1], s * proto.scale[2])
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        proto.name = f"{kit}:{name}"
        for c in list(proto.users_collection):
            c.objects.unlink(proto)
        self.stash.objects.link(proto)
        self.protos[key] = proto
        return proto


# Surface kinds written to vertex colour G (kind / 4); the renderer picks textures from them.
KIND_DEFAULT, KIND_FOLIAGE, KIND_BARK, KIND_PAVEMENT = 0, 1, 2, 3


def _material_kind(material) -> int:
    name = (material.name if material else "").lower()
    if "leaf" in name:
        return KIND_FOLIAGE
    if "wood" in name or "bark" in name or "trunk" in name:
        return KIND_BARK
    return KIND_DEFAULT


def tag_mesh(mesh, variant: float, kind: int | None = None) -> None:
    """Vertex colour R = per-object variant in [0, 1), G = surface kind / 4 (per material if kind is None)."""
    n_polys, n_loops = len(mesh.polygons), len(mesh.loops)
    if kind is None:
        kinds = np.array([_material_kind(m) for m in mesh.materials] or [KIND_DEFAULT], dtype=np.float32)
        mat_index = np.zeros(n_polys, dtype=np.int32)
        mesh.polygons.foreach_get("material_index", mat_index)
        totals = np.zeros(n_polys, dtype=np.int32)
        mesh.polygons.foreach_get("loop_total", totals)
        loop_kind = np.repeat(kinds[np.minimum(mat_index, len(kinds) - 1)], totals)
    else:
        loop_kind = np.full(n_loops, kind, dtype=np.float32)
    colors = np.zeros((n_loops, 4), dtype=np.float32)
    colors[:, 0] = variant
    colors[:, 1] = loop_kind / 4.0
    colors[:, 3] = 1.0
    attr = mesh.color_attributes.new("tag", "FLOAT_COLOR", "CORNER")
    attr.data.foreach_set("color", colors.ravel())
    mesh.color_attributes.active_color = attr


class City:
    def __init__(self, lib: Library, rng: random.Random):
        self.lib, self.rng = lib, rng
        self.tag_rng = random.Random(rng.random())  # separate stream: tags never perturb the layout
        self.collections = {}
        for cls in CLASSES[1:]:
            col = bpy.data.collections.new(cls)
            bpy.context.scene.collection.children.link(col)
            self.collections[cls] = col
        self.layout: list[dict] = []

    def place(self, cls: str, kit: str, name: str, x: float, y: float, yaw_deg: float = 0.0, scale: float = 1.0) -> None:
        """x, y in Blender ground-plane metres; yaw about +Z."""
        proto = self.lib.get(kit, name)
        inst = proto.copy()
        inst.data = proto.data.copy()
        tag_mesh(inst.data, self.tag_rng.random(), KIND_PAVEMENT if name.startswith("tile-") else None)
        inst.matrix_world = Matrix.Translation((x, y, 0.0)) @ Matrix.Rotation(math.radians(yaw_deg), 4, "Z") @ Matrix.Scale(scale, 4)
        self.collections[cls].objects.link(inst)
        corners = [inst.matrix_world @ Vector(c) for c in proto.bound_box]
        lo = [min(c[i] for c in corners) for i in range(3)]
        hi = [max(c[i] for c in corners) for i in range(3)]
        # Blender (x, y, z-up) -> glTF (x, y-up, z = -y)
        self.layout.append(
            {"class": cls, "asset": f"{kit}:{name}", "min": [lo[0], lo[2], -hi[1]], "max": [hi[0], hi[2], -lo[1]]}
        )

    def ground_plane(self, half: float) -> None:
        bpy.ops.mesh.primitive_plane_add(size=2 * half, location=(0, 0, 0))
        plane = bpy.context.view_layer.objects.active
        mat = bpy.data.materials.new("grass")
        mat.use_nodes = True
        mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.28, 0.45, 0.2, 1.0)
        plane.data.materials.append(mat)
        tag_mesh(plane.data, 0.5, KIND_DEFAULT)
        for c in list(plane.users_collection):
            c.objects.unlink(plane)
        self.collections["ground"].objects.link(plane)


def build(city: City, blocks: int) -> float:
    rng = city.rng
    n_tiles = blocks * BLOCK + 1
    half = n_tiles * TILE / 2
    centre = (blocks - 1) / 2

    def tile_xy(i: int, j: int) -> tuple[float, float]:
        return (i + 0.5) * TILE - half, (j + 0.5) * TILE - half

    # Roads every BLOCK tiles in both directions.
    for i in range(n_tiles):
        for j in range(n_tiles):
            ri, rj = i % BLOCK == 0, j % BLOCK == 0
            if not (ri or rj):
                continue
            x, y = tile_xy(i, j)
            if ri and rj:
                city.place("road", "roads", "road-crossroad", x, y)
                continue
            yaw = 0.0 if ri else 90.0  # road-straight runs along local Y
            city.place("road", "roads", "road-straight", x, y, yaw)
            if rng.random() < 0.18:
                side = rng.choice([-1, 1]) * 0.42 * TILE
                lx, ly = (x + side, y) if ri else (x, y + side)
                city.place("prop", "roads", "light-square", lx, ly, yaw + (180 if side > 0 else 0))
            if rng.random() < 0.22:
                lane = rng.choice([-1, 1]) * 0.18 * TILE
                cx, cy = (x + lane, y) if ri else (x, y + lane)
                city.place("vehicle", "cars", rng.choice(CARS), cx, cy, yaw + rng.choice([0, 180]))

    for bi in range(blocks):
        for bj in range(blocks):
            r = max(abs(bi - centre), abs(bj - centre)) / max(centre, 1)
            district = "downtown" if r < 0.35 else "suburb" if r < 0.75 else "park"
            if district != "park" and rng.random() < 0.12:
                district = "park"
            cells = [(bi * BLOCK + 1 + a, bj * BLOCK + 1 + b) for a in range(3) for b in range(3)]
            for i, j in cells:
                x, y = tile_xy(i, j)
                yaw = 90.0 * rng.randrange(4)
                if district == "downtown":
                    city.place("road", "roads", "tile-low", x, y)
                    tower_p = 0.55 * (1 - r / 0.35)
                    name = rng.choice(TOWERS) if rng.random() < tower_p else rng.choice(DOWNTOWN)
                    city.place("building", "commercial", name, x, y, yaw)
                elif district == "suburb":
                    roll = rng.random()
                    if roll < 0.55:
                        city.place("building", "suburban", rng.choice(HOUSES), x, y, yaw)
                    elif roll < 0.85:
                        for _ in range(rng.randint(1, 3)):
                            ox, oy = (rng.uniform(-0.35, 0.35) * TILE for _ in range(2))
                            city.place("vegetation", "suburban", rng.choice(["tree-large", "tree-small"]), x + ox, y + oy, rng.uniform(0, 360))
                    else:
                        city.place("prop", "suburban", rng.choice(["fence-2x2", "fence-3x3", "planter"]), x, y, yaw)
                else:
                    for _ in range(rng.randint(2, 5)):
                        ox, oy = (rng.uniform(-0.45, 0.45) * TILE for _ in range(2))
                        kind = rng.random()
                        if kind < 0.65:
                            city.place("vegetation", "nature", rng.choice(PARK_TREES), x + ox, y + oy, rng.uniform(0, 360), rng.uniform(0.8, 1.3))
                        elif kind < 0.85:
                            city.place("vegetation", "nature", rng.choice(BUSHES), x + ox, y + oy, rng.uniform(0, 360), rng.uniform(0.8, 1.4))
                        else:
                            city.place("prop", "nature", rng.choice(ROCKS), x + ox, y + oy, rng.uniform(0, 360), rng.uniform(0.6, 1.2))
    # Forest ring so outward-looking cameras see a horizon of trees, not the world's edge.
    for i in range(-FOREST_RING, n_tiles + FOREST_RING):
        for j in range(-FOREST_RING, n_tiles + FOREST_RING):
            if 0 <= i < n_tiles and 0 <= j < n_tiles:
                continue
            x, y = tile_xy(i, j)
            for _ in range(rng.randint(2, 4)):
                ox, oy = (rng.uniform(-0.45, 0.45) * TILE for _ in range(2))
                city.place("vegetation", "nature", rng.choice(PARK_TREES), x + ox, y + oy, rng.uniform(0, 360), rng.uniform(1.0, 1.6))
    city.ground_plane(GROUND_HALF_EXTENT)
    return half


def _join(objs: list) -> "bpy.types.Object":
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    if len(objs) > 1:
        bpy.ops.object.join()
    return bpy.context.view_layer.objects.active


def export(city: City, out: Path) -> None:
    for cls, col in city.collections.items():
        objs = list(col.objects)
        if not objs:
            continue
        # raylib meshes use 16-bit indices; loop count bounds exported glTF vertices per batch.
        batches, batch, loops = [], [], 0
        for o in objs:
            n = len(o.data.loops)
            if batch and loops + n > MAX_LOOPS_PER_MESH:
                batches.append(batch)
                batch, loops = [], 0
            batch.append(o)
            loops += n
        batches.append(batch)
        merged = [_join(b) for b in batches]
        bpy.ops.object.select_all(action="DESELECT")
        for i, m in enumerate(merged):
            m.name = f"{cls}.{i:03d}"
            m.select_set(True)
        bpy.ops.export_scene.gltf(
            filepath=str(out / f"{cls}.glb"), export_format="GLB", use_selection=True, export_yup=True, export_apply=True,
            export_vertex_color="ACTIVE",
        )
        faces = sum(len(m.data.polygons) for m in merged)
        print(f"exported {cls}: {faces} faces in {len(merged)} meshes", flush=True)


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--blocks", type=int, default=12, help="city is blocks x blocks; each block is 40 m")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    args.out.mkdir(parents=True, exist_ok=True)
    city = City(Library(args.assets.resolve()), random.Random(args.seed))
    half = build(city, args.blocks)
    export(city, args.out)
    solid = [o for o in city.layout if o["class"] in SOLID]
    (args.out / "layout.json").write_text(
        json.dumps(
            {
                "classes": CLASSES,
                "solid_classes": sorted(SOLID),
                "seed": args.seed,
                "blocks": args.blocks,
                "tile_m": TILE,
                "city_half_extent_m": half,
                "objects": solid,
            },
            indent=1,
        )
    )
    print(f"placed {len(city.layout)} objects ({len(solid)} solid); city spans {2 * half:.0f} m", flush=True)


main()
