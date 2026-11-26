"""Offline temporal-encoder pilot: left/right/never triplets of driving clips.

A clip is one camera path rendered three times through one look. Variant ``left`` has a Kenney
car in the lane on the camera's left, ``right`` the same car in the lane on the right, and
``never`` no car. The camera turns away from the car so it leaves the frustum; from
``first_hidden`` on, all three variants are pixel-identical.

generate(scene, count, seed) yields (rgb, target, first_hidden, meta) per clip:
  rgb           uint8   [3, T, H, W, 3]  variants in VARIANTS order (left, right, never)
  target        float32 [3, T, 2]        (bearing rad, planar range m); zeros for ``never``
  first_hidden  int                      common-suffix start; frames [first_hidden, T) identical
  meta          dict                     geometry and hashes, see _render_triplet

Bearing sign: positive = car to the right of the camera's horizontal forward (clockwise seen
from above). Range is the XZ distance from the camera to the car centre.

Repro (repo root):
    .venv/bin/python -c "import sys; sys.path.insert(0, 'scripts'); from pathlib import Path
    from temporal_clips import generate
    rgb, tgt, f, meta = next(generate(Path('assets/geometry/train-101'), 1, 0))
    print(rgb.shape, tgt.shape, f, meta['suffix_sha256']['never'][:16])"
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pyray as rl
from raylib import ffi

from visenc.city import DETAIL_SLOTS, SHADOW_SLOT, Appearance, CityRenderer, Pose, _srgb_u8

ROOT = Path(__file__).resolve().parents[1]
KENNEY_CARS = ROOT / "assets/kenney/cars/Models/GLB format"
POLYHAVEN = ROOT / "assets/polyhaven"

VARIANTS = ("left", "right", "never")
T, H, W = 18, 240, 320  # frames at 4 Hz; render size
VFOV = math.radians(60.0)  # fixed across clips
TURN_FRAMES = 9  # yaw ramp length; the car has left the frustum by about frame 7
MIN_SUFFIX = 9
MIN_CAR_PIXELS = 100  # frame-0 pixels that must differ from ``never`` for each positive
MAX_PATH_ATTEMPTS = 128  # cheap geometry rejections do not consume the rendering budget
MAX_RENDER_ATTEMPTS = 12
CLEARANCE_M = 1.5
LANE_OFFSET_M = 1.8  # 0.18 * tile, the lane offset build_city.py uses for cars
CAR_SCALE = 1.8  # build_city.py KIT_SCALE["cars"]; Kenney prototypes carry no other scale
CARS = (
    "sedan", "suv", "taxi", "van", "hatchback-sports", "police", "delivery", "truck", "suv-luxury", "sedan-sports",
)  # fmt: skip
SOLID_MARGIN_M = 0.5


class _TargetCity(CityRenderer):
    """CityRenderer without background vehicles; the target car is attached only while rendering."""

    def __init__(self, scene_dir: Path, width: int, height: int, textures_dir: Path):
        super().__init__(scene_dir, width, height, textures_dir)
        self.vehicle_id = self.layout.classes.index("vehicle")
        background = self.models.pop(self.vehicle_id, None)
        if background is not None:
            _free_model(background)

    def render_target(self, pose: Pose, look: Appearance, car_model) -> np.ndarray:
        if car_model is None:
            self.models.pop(self.vehicle_id, None)
        else:
            self.models[self.vehicle_id] = car_model
        try:
            return self.render_rgb(pose, look)
        finally:
            self.models.pop(self.vehicle_id, None)  # keeps close() from unloading the caller's car


def _free_model(model) -> None:
    for i in range(model.materialCount):
        for slot, _ in (*DETAIL_SLOTS, SHADOW_SLOT):
            model.materials[i].maps[slot].texture = rl.Texture(0, 0, 0, 0, 0)  # not owned by the model
    rl.unload_model(model)


def _load_car(name: str):
    model = rl.load_model(str(KENNEY_CARS / f"{name}.glb"))
    default = rl.rl_get_texture_id_default()
    for i in range(model.materialCount):
        m = model.materials[i].maps[rl.MATERIAL_MAP_ALBEDO]
        if m.texture.id == default:
            m.color = rl.Color(_srgb_u8(m.color.r), _srgb_u8(m.color.g), _srgb_u8(m.color.b), m.color.a)
    return model


def _raw_bounds(model) -> tuple[np.ndarray, np.ndarray]:
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for k in range(model.meshCount):
        mesh = model.meshes[k]
        n = int(mesh.vertexCount)
        v = np.frombuffer(ffi.buffer(mesh.vertices, n * 12), dtype=np.float32).reshape(n, 3)
        lo = np.minimum(lo, v.min(0))
        hi = np.maximum(hi, v.max(0))
    return lo, hi


@dataclass(frozen=True)
class _Car:
    name: str
    model: object
    lo: np.ndarray  # raw glTF AABB, prototype units
    hi: np.ndarray
    base_yaw: float  # rotates the long axis onto world Z (the road direction)
    hl: float  # half length along Z, scaled
    hw: float  # half width along X, scaled
    centre_y: float  # height of the scaled AABB centre above the ground


def _car_spec(name: str, model) -> _Car:
    lo, hi = _raw_bounds(model)
    ext_x, ext_z = (hi - lo)[0] * CAR_SCALE, (hi - lo)[2] * CAR_SCALE
    if ext_x > ext_z:
        base_yaw, hl, hw = math.pi / 2, ext_x / 2, ext_z / 2
    else:
        base_yaw, hl, hw = 0.0, ext_z / 2, ext_x / 2
    return _Car(name, model, lo, hi, base_yaw, float(hl), float(hw), float(CAR_SCALE * (hi[1] - lo[1]) / 2))


def _car_matrix(car: _Car, yaw: float, centre: np.ndarray):
    """Scale, then yaw about +Y, then translate so the car's AABB centre sits at ``centre`` (base on y=0).

    Same composition as raylib's DrawModelEx; positive yaw here matches the Blender yaw in layout.json.
    """
    s = CAR_SCALE
    cx, cz = ((car.lo + car.hi) * (0.5 * s))[[0, 2]]
    c, sn = math.cos(yaw), math.sin(yaw)
    rx, rz = cx * c + cz * sn, -cx * sn + cz * c
    scale = rl.matrix_scale(s, s, s)
    rot = rl.matrix_rotate((0.0, 1.0, 0.0), yaw)
    move = rl.matrix_translate(float(centre[0] - rx), float(-s * car.lo[1]), float(centre[2] - rz))
    return rl.matrix_multiply(rl.matrix_multiply(scale, rot), move)


@dataclass(frozen=True)
class _Scene:
    tile: float
    half: float
    cols: tuple[int, ...]  # road columns (x tiles) that are not the boundary
    free_rows: tuple[int, ...]  # rows (z tiles) that are not a crossing road
    bmin: np.ndarray  # non-vehicle solid AABBs; background cars are not drawn, so they do not block
    bmax: np.ndarray

    @staticmethod
    def from_layout(layout: dict) -> "_Scene":
        tile, half = float(layout["tile_m"]), float(layout["city_half_extent_m"])
        n = round(2 * half / tile)
        design = layout["design"]
        road_x = {0, *np.cumsum(design["periods_x"]).tolist()}
        road_y = {0, *np.cumsum(design["periods_y"]).tolist()}
        cols = tuple(i for i in sorted(road_x) if 0 < i < n - 1)
        rows = tuple(j for j in range(1, n - 1) if j not in road_y)
        solids = [o for o in layout["objects"] if o["class"] != "vehicle"]
        return _Scene(
            tile, half, cols, rows,
            np.array([o["min"] for o in solids], dtype=np.float64),
            np.array([o["max"] for o in solids], dtype=np.float64),
        )  # fmt: skip

    def centre(self, index: int) -> float:
        """Tile centre in build_city's Blender (x or y) coordinates."""
        return (index + 0.5) * self.tile - self.half


def _hits_solid(c: np.ndarray, car: _Car, scene: _Scene) -> bool:
    m = SOLID_MARGIN_M
    hit = (
        (scene.bmax[:, 0] > c[0] - car.hw - m) & (scene.bmin[:, 0] < c[0] + car.hw + m)
        & (scene.bmax[:, 2] > c[2] - car.hl - m) & (scene.bmin[:, 2] < c[2] + car.hl + m)
    )  # fmt: skip
    return bool(hit.any())


@dataclass(frozen=True)
class _Path:
    cam: np.ndarray  # [T,3] camera positions
    yaw: np.ndarray  # [T] radians
    pitch: float
    sun_az: float  # total sun azimuth drift over the clip (rad)
    exposure_drift: float  # fractional exposure change over the clip
    mid: np.ndarray  # [3] lane-centre point between the two lanes (y = car centre height)
    left: np.ndarray  # [3] car centre, camera-left lane
    right: np.ndarray  # [3] car centre, camera-right lane


def _sample_path(rng: np.random.Generator, scene: _Scene, car: _Car, city: _TargetCity,
                 sideways: bool, wide: bool) -> _Path | str:
    """One candidate camera path, or a string naming why it was rejected."""
    i = int(rng.choice(scene.cols))
    j = int(rng.choice(scene.free_rows))
    tx, tz = scene.centre(i), -scene.centre(j)  # world X = Blender x, world Z = -Blender y
    room = scene.tile / 2 - SOLID_MARGIN_M - car.hl
    if room < 0:
        return "car longer than the road tile"
    zc = tz + float(rng.uniform(-room, room))

    yaw0 = (math.pi if rng.uniform() < 0.5 else 0.0) + math.radians(float(rng.uniform(-5.0, 5.0)))
    fx, fz = -math.sin(yaw0), -math.cos(yaw0)  # horizontal forward (Pose.basis at pitch 0)
    rx, rz = -fz, fx  # horizontal right
    dist = float(rng.uniform(13.0, 17.0))
    altitude = float(rng.uniform(9.0, 13.0))
    drift = float(rng.uniform(0.0, 2.0))  # forward travel towards the car over the clip
    turn = (1.0 if rng.uniform() < 0.5 else -1.0) * math.radians(float(rng.uniform(76.0, 102.0)))
    pitch = -math.radians(float(rng.uniform(22.0, 30.0)))  # magnitude <= 45 deg
    if wide:
        dist = float(rng.uniform(8.0, 27.0))
        altitude = float(rng.uniform(5.0, 11.0))
        pitch = -math.atan2(altitude - car.centre_y, dist)

    mid = np.array([tx, car.centre_y, zc])
    lateral = LANE_OFFSET_M * np.array([rx, 0.0, rz])
    left, right = mid - lateral, mid + lateral
    if _hits_solid(left, car, scene) or _hits_solid(right, car, scene):
        return "car footprint overlaps a solid object"

    cam0 = np.array([tx - dist * fx, altitude, zc - dist * fz])  # car sits ``dist`` ahead of the camera
    if wide:
        cam0 += rng.uniform(-4.0, 4.0) * np.array([rx, 0.0, rz])
    progress = np.arange(T) / (T - 1)
    cam = cam0 + np.outer(progress * drift, [fx, 0.0, fz])
    if sideways:
        cam += np.outer(progress * rng.uniform(-3.0, 3.0), [rx, 0.0, rz])
    ramp = np.minimum(np.arange(T) / TURN_FRAMES, 1.0)
    yaw = yaw0 + turn * ramp * ramp * (3.0 - 2.0 * ramp)
    if min(city.layout.clearance(p) for p in cam) < CLEARANCE_M:
        return "camera within clearance of a solid object"
    return _Path(
        cam, yaw, pitch, float(rng.uniform(-0.1, 0.1)), float(rng.uniform(-0.04, 0.04)), mid, left, right,
    )  # fmt: skip


def _look_at(base: Appearance, path: _Path, t: int) -> Appearance:
    """Shared look with the sun azimuth and exposure drifting linearly over the clip."""
    s = t / (T - 1)
    az = path.sun_az * s
    c, sn = math.cos(az), math.sin(az)
    x, y, z = base.sun_dir
    return replace(
        base,
        sun_dir=(c * x + sn * z, y, -sn * x + c * z),
        exposure=base.exposure * (1.0 + path.exposure_drift * s),
    )  # fmt: skip


def _first_hidden(rgb: np.ndarray) -> int | str:
    """Start of the common identical suffix, or the reason the clip is invalid."""
    for v in (0, 1):
        diff = int(np.any(rgb[v, 0] != rgb[2, 0], axis=-1).sum())
        if diff < MIN_CAR_PIXELS:
            return f"{VARIANTS[v]} differs from never in {diff} px at frame 0"
    same = np.all(rgb[0] == rgb[2], axis=(1, 2, 3)) & np.all(rgb[1] == rgb[2], axis=(1, 2, 3))
    f = T
    while f > 0 and same[f - 1]:
        f -= 1
    if T - f < MIN_SUFFIX:
        return f"common suffix {T - f} < {MIN_SUFFIX} frames"
    return f


def _targets(poses: list[Pose], centres: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    out = np.zeros((3, T, 2), dtype=np.float32)
    cam_xz = np.array([p.position[[0, 2]] for p in poses])
    yaw = np.array([p.yaw for p in poses])
    fx, fz = -np.sin(yaw), -np.cos(yaw)
    rx, rz = -fz, fx
    for v, centre in enumerate(centres):
        d = centre[[0, 2]][None] - cam_xz
        out[v, :, 0] = np.arctan2(d[:, 0] * rx + d[:, 1] * rz, d[:, 0] * fx + d[:, 1] * fz)
        out[v, :, 1] = np.hypot(d[:, 0], d[:, 1])
    return out


def _render_triplet(city: _TargetCity, scene: _Scene, car: _Car, rng: np.random.Generator, sideways: bool, wide: bool):
    rgb = np.empty((3, T, H, W, 3), dtype=np.uint8)
    reasons: list[str] = []
    rendered = 0
    for _ in range(MAX_PATH_ATTEMPTS):
        path = _sample_path(rng, scene, car, city, sideways, wide)
        if isinstance(path, str):
            reasons.append(path)
            continue
        if rendered == MAX_RENDER_ATTEMPTS:
            break
        rendered += 1
        base = Appearance.random(rng, "seen")  # one look for all three variants
        looks = [_look_at(base, path, t) for t in range(T)]
        poses = [Pose(path.cam[t].copy(), float(path.yaw[t]), path.pitch, 0.0, VFOV) for t in range(T)]
        car_yaw = car.base_yaw + (math.pi if rng.uniform() < 0.5 else 0.0)  # facing, does not move the footprint
        for v, centre in enumerate((path.left, path.right, None)):
            if centre is not None:
                car.model.transform = _car_matrix(car, car_yaw, centre)
            for t in range(T):
                rgb[v, t] = city.render_target(poses[t], looks[t], None if centre is None else car.model)
        first = _first_hidden(rgb)
        if isinstance(first, str):
            reasons.append(first)
            continue
        digests = {
            name: hashlib.sha256(np.ascontiguousarray(rgb[v, first:]).tobytes()).hexdigest()
            for v, name in enumerate(VARIANTS)
        }
        if len(set(digests.values())) != 1:
            raise RuntimeError(f"common suffix differs across variants: {digests}")
        centres = (path.left, path.right)
        meta = {
            "variants": VARIANTS,
            "car": car.name,
            "car_yaw_deg": math.degrees(car_yaw),
            "car_scale": CAR_SCALE,
            "bearing_sign": "positive = car right of camera forward (clockwise seen from above)",
            "poses": poses,
            "camera_position": np.stack([p.position for p in poses]).astype(np.float32),
            "vfov_deg": math.degrees(VFOV),
            "pitch_deg": math.degrees(path.pitch),
            "car_world_centres": np.stack([path.left, path.right, path.mid]).astype(np.float32),
            "suffix_sha256": digests,
        }
        return rgb, _targets(poses, centres), first, meta
    raise RuntimeError(f"no valid clip for car {car.name} after {rendered} renders: {reasons[-12:]}")


def generate(scene: Path, count: int, seed: int, *, sideways: bool = False,
             wide: bool = False) -> Iterator[tuple[np.ndarray, np.ndarray, int, dict]]:
    scene = Path(scene)
    layout = json.loads((scene / "layout.json").read_text())
    context = _Scene.from_layout(layout)
    rng = np.random.default_rng(seed)
    with _TargetCity(scene, W, H, POLYHAVEN) as city:
        for _ in range(count):
            name = CARS[int(rng.integers(len(CARS)))]
            model = _load_car(name)
            try:
                car = _car_spec(name, model)
                yield _render_triplet(city, context, car, rng, sideways, wide)
            finally:
                _free_model(model)
