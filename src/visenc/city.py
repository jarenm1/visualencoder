"""Raylib renderer for the static city built by scene/build_city.py.

Randomisation is only over camera pose (where the drone is in the one city) and over
appearance (lighting/sky/fog/exposure/tint), never over geometry. Every pose is rendered
with three passes: lit RGB (2x supersampled), semantic class ids, and optical-Z depth.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyray as rl
from raylib import ffi

VERTEX = """
#version 330
in vec3 vertexPosition; in vec2 vertexTexCoord; in vec3 vertexNormal; in vec4 vertexColor;
uniform mat4 mvp; uniform mat4 matModel; uniform mat4 matNormal;
out vec3 fragPosition; out vec2 fragTexCoord; out vec4 fragColor; out vec3 fragNormal;
void main() {
    fragPosition = vec3(matModel * vec4(vertexPosition, 1.0));
    fragTexCoord = vertexTexCoord;
    fragColor = vertexColor;
    fragNormal = normalize(vec3(matNormal * vec4(vertexNormal, 0.0)));
    gl_Position = mvp * vec4(vertexPosition, 1.0);
}
"""

LIT = """
#version 330
in vec3 fragPosition; in vec2 fragTexCoord; in vec4 fragColor; in vec3 fragNormal;
uniform sampler2D texture0; uniform vec4 colDiffuse;
uniform vec3 viewPos; uniform vec3 sunDir; uniform vec3 sunColor;
uniform vec3 skyAmbient; uniform vec3 groundAmbient; uniform vec3 tint;
uniform vec3 fogColor; uniform float fogDensity; uniform float exposure;
out vec4 finalColor;
void main() {
    vec4 albedo = texture(texture0, fragTexCoord) * colDiffuse * fragColor;
    vec3 n = normalize(fragNormal);
    if (!gl_FrontFacing) n = -n;
    vec3 ambient = mix(groundAmbient, skyAmbient, 0.5 + 0.5 * n.y);
    vec3 c = albedo.rgb * tint * (ambient + sunColor * max(dot(n, sunDir), 0.0)) * exposure;
    c = mix(fogColor, c, exp(-fogDensity * length(fragPosition - viewPos)));
    finalColor = vec4(clamp(c, 0.0, 1.0), 1.0);
}
"""

SEMANTIC = """
#version 330
in vec3 fragPosition; in vec2 fragTexCoord; in vec4 fragColor; in vec3 fragNormal;
uniform vec4 classColor;
out vec4 finalColor;
void main() { finalColor = classColor; }
"""

# Optical-Z depth packed into 24 bits of RGB: value = depth / far * (2^24 - 1).
DEPTH = """
#version 330
in vec3 fragPosition; in vec2 fragTexCoord; in vec4 fragColor; in vec3 fragNormal;
uniform vec3 viewPos; uniform vec3 camForward; uniform float far;
out vec4 finalColor;
void main() {
    float v = floor(clamp(dot(fragPosition - viewPos, camForward) / far, 0.0, 1.0) * 16777215.0);
    float r = floor(v / 65536.0);
    float g = floor((v - r * 65536.0) / 256.0);
    float b = v - r * 65536.0 - g * 256.0;
    finalColor = vec4(r, g, b, 255.0) / 255.0;
}
"""

NEAR, FAR = 0.1, 2000.0
SUPERSAMPLE = 2


@dataclass(frozen=True)
class Pose:
    position: np.ndarray  # [3] world metres, +Y up
    yaw: float  # radians about +Y; 0 looks along -Z
    pitch: float  # radians, positive looks up
    roll: float  # radians about the view axis
    vfov: float  # radians

    def basis(self) -> tuple[np.ndarray, np.ndarray]:
        cp = math.cos(self.pitch)
        forward = np.array([-math.sin(self.yaw) * cp, math.sin(self.pitch), -math.cos(self.yaw) * cp])
        right = np.cross(forward, [0.0, 1.0, 0.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        up_rolled = math.cos(self.roll) * up + math.sin(self.roll) * right
        return forward, up_rolled


@dataclass(frozen=True)
class Appearance:
    sun_dir: tuple[float, float, float]
    sun_color: tuple[float, float, float]
    sky_ambient: tuple[float, float, float]
    ground_ambient: tuple[float, float, float]
    sky: tuple[float, float, float]  # background and fog colour
    fog_density: float
    exposure: float
    tint: tuple[float, float, float]

    @staticmethod
    def canonical() -> "Appearance":
        sun = np.array([0.35, 0.85, 0.4])
        return Appearance(
            tuple(sun / np.linalg.norm(sun)), (0.75, 0.72, 0.65), (0.45, 0.5, 0.58), (0.3, 0.28, 0.25),
            (0.55, 0.7, 0.88), 0.002, 1.0, (1.0, 1.0, 1.0),
        )  # fmt: skip

    @staticmethod
    def random(rng: np.random.Generator) -> "Appearance":
        """Task-valid variation: lighting, atmosphere and camera response; geometry untouched."""
        elevation, azimuth = rng.uniform(0.15, 1.45), rng.uniform(0, 2 * math.pi)
        sun = (math.cos(elevation) * math.cos(azimuth), math.sin(elevation), math.cos(elevation) * math.sin(azimuth))
        warmth = rng.uniform(-1, 1)
        sun_color = np.array([1.0 + 0.25 * warmth, 0.95, 1.0 - 0.3 * warmth]) * rng.uniform(0.3, 1.1)
        sky = np.clip(rng.uniform([0.25, 0.3, 0.35], [0.95, 0.95, 1.0]) * rng.uniform(0.4, 1.1), 0, 1)
        ambient = rng.uniform(0.15, 0.6)
        return Appearance(
            sun, tuple(sun_color), tuple(sky * ambient / max(sky.max(), 1e-3) + 0.05), tuple(rng.uniform(0.1, 0.35, 3)),
            tuple(sky), float(rng.uniform(0.0, 0.012) ** 1.5 * 10), float(rng.uniform(0.7, 1.4)), tuple(rng.uniform(0.85, 1.15, 3)),
        )  # fmt: skip


class CityLayout:
    def __init__(self, scene_dir: Path):
        layout = json.loads((scene_dir / "layout.json").read_text())
        self.classes: list[str] = layout["classes"]
        self.half_extent: float = layout["city_half_extent_m"]
        self.box_min = np.array([o["min"] for o in layout["objects"]], dtype=np.float64)
        self.box_max = np.array([o["max"] for o in layout["objects"]], dtype=np.float64)

    def clearance(self, p: np.ndarray) -> float:
        """Distance from p to the nearest solid object's AABB (0 inside)."""
        d = np.maximum(np.maximum(self.box_min - p, p - self.box_max), 0.0)
        return float(np.sqrt((d * d).sum(1)).min())

    def sample_pose(self, rng: np.random.Generator, margin: float = 1.5) -> Pose:
        limit = self.half_extent - 10.0
        while True:
            p = np.array([rng.uniform(-limit, limit), 2.0 + 43.0 * rng.uniform() ** 2, rng.uniform(-limit, limit)])
            if self.clearance(p) >= margin:
                return Pose(
                    p, rng.uniform(-math.pi, math.pi), math.radians(rng.uniform(-60, 15)),
                    math.radians(float(np.clip(rng.normal(0, 8), -20, 20))), math.radians(rng.uniform(55, 75)),
                )  # fmt: skip


STATE_NAMES = ["pos_x", "pos_y", "pos_z", "yaw_sin", "yaw_cos", "pitch", "roll", "vfov", "clearance", "forward_log_depth"]
STATE_GROUPS = {"position": slice(0, 3), "orientation": slice(3, 7), "intrinsics": slice(7, 8), "clearance": slice(8, 10)}


def pose_state(pose: Pose, clearance: float, depth: np.ndarray) -> np.ndarray:
    h, w = depth.shape
    forward = math.log(float(np.median(depth[h // 2 - h // 16 : h // 2 + h // 16, w // 2 - w // 16 : w // 2 + w // 16])))
    return np.array(
        [*pose.position, math.sin(pose.yaw), math.cos(pose.yaw), pose.pitch, pose.roll, pose.vfov, clearance, forward],
        dtype=np.float32,
    )


def _vec(values) -> object:
    return ffi.new(f"float[{len(values)}]", [float(v) for v in values])


class CityRenderer:
    def __init__(self, scene_dir: Path, width: int, height: int):
        self.layout = CityLayout(scene_dir)
        self.width, self.height = width, height
        rl.set_trace_log_level(rl.LOG_WARNING)
        rl.set_config_flags(rl.FLAG_WINDOW_HIDDEN)
        rl.init_window(width, height, "visenc-city")
        self.rgb_target = rl.load_render_texture(width * SUPERSAMPLE, height * SUPERSAMPLE)
        self.target = rl.load_render_texture(width, height)
        self.models: dict[int, object] = {}
        for class_id, name in enumerate(self.layout.classes):
            path = scene_dir / f"{name}.glb"
            if path.exists():
                self.models[class_id] = rl.load_model(str(path))
        self.lit = rl.load_shader_from_memory(VERTEX, LIT)
        self.semantic = rl.load_shader_from_memory(VERTEX, SEMANTIC)
        self.depth = rl.load_shader_from_memory(VERTEX, DEPTH)
        for shader in (self.lit, self.semantic, self.depth):
            shader.locs[rl.SHADER_LOC_MATRIX_MODEL] = rl.get_shader_location(shader, "matModel")
            shader.locs[rl.SHADER_LOC_MATRIX_NORMAL] = rl.get_shader_location(shader, "matNormal")

    def close(self) -> None:
        for model in self.models.values():
            rl.unload_model(model)
        for shader in (self.lit, self.semantic, self.depth):
            rl.unload_shader(shader)
        rl.unload_render_texture(self.rgb_target)
        rl.unload_render_texture(self.target)
        rl.close_window()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    @staticmethod
    def _set(shader, name: str, values) -> None:
        kinds = {1: rl.SHADER_UNIFORM_FLOAT, 3: rl.SHADER_UNIFORM_VEC3, 4: rl.SHADER_UNIFORM_VEC4}
        rl.set_shader_value(shader, rl.get_shader_location(shader, name), _vec(values), kinds[len(values)])

    def _draw(self, target, pose: Pose, clear, shader, per_class=None) -> np.ndarray:
        forward, up = pose.basis()
        camera = rl.Camera3D(
            tuple(pose.position), tuple(pose.position + forward), tuple(up), math.degrees(pose.vfov), rl.CAMERA_PERSPECTIVE
        )
        rl.begin_texture_mode(target)
        rl.clear_background(clear)
        rl.rl_set_clip_planes(NEAR, FAR)
        rl.begin_mode_3d(camera)
        for class_id, model in self.models.items():
            if per_class is not None:
                per_class(class_id)
            for i in range(model.materialCount):
                model.materials[i].shader = shader
            rl.draw_model(model, (0.0, 0.0, 0.0), 1.0, rl.WHITE)
        rl.end_mode_3d()
        rl.end_texture_mode()
        image = rl.load_image_from_texture(target.texture)
        w, h = target.texture.width, target.texture.height
        pixels = np.frombuffer(ffi.buffer(image.data, w * h * 4), dtype=np.uint8).reshape(h, w, 4)[::-1].copy()
        rl.unload_image(image)
        return pixels

    def render_geometry(self, pose: Pose) -> tuple[np.ndarray, np.ndarray]:
        """Appearance-independent passes: (semantic uint8 [H,W], depth float32 [H,W] metres)."""
        forward, _ = pose.basis()
        self._set(self.depth, "viewPos", pose.position)
        self._set(self.depth, "camForward", forward)
        self._set(self.depth, "far", [FAR])
        packed = self._draw(self.target, pose, (255, 255, 255, 255), self.depth).astype(np.uint32)
        depth = ((packed[..., 0] << 16) | (packed[..., 1] << 8) | packed[..., 2]).astype(np.float32) * (FAR / 16777215.0)
        semantic = self._draw(
            self.target, pose, (0, 0, 0, 255), self.semantic,
            lambda c: self._set(self.semantic, "classColor", [c / 255.0, 0.0, 0.0, 1.0]),
        )[..., 0]  # fmt: skip
        return semantic, depth

    def render_rgb(self, pose: Pose, look: Appearance) -> np.ndarray:
        """uint8 [H,W,3], box-downsampled from a SUPERSAMPLE x render."""
        s = self.lit
        self._set(s, "viewPos", pose.position)
        sun = np.asarray(look.sun_dir, dtype=np.float64)
        self._set(s, "sunDir", sun / np.linalg.norm(sun))
        for name, value in [
            ("sunColor", look.sun_color), ("skyAmbient", look.sky_ambient), ("groundAmbient", look.ground_ambient),
            ("tint", look.tint), ("fogColor", look.sky), ("fogDensity", [look.fog_density]), ("exposure", [look.exposure]),
        ]:  # fmt: skip
            self._set(s, name, value)
        sky = tuple(int(round(255 * v)) for v in look.sky) + (255,)
        big = self._draw(self.rgb_target, pose, sky, s)[..., :3].astype(np.float32)
        h, w = self.height, self.width
        small = big.reshape(h, SUPERSAMPLE, w, SUPERSAMPLE, 3).mean((1, 3))
        return np.round(small).astype(np.uint8)
