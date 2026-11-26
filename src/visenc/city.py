"""Raylib renderer for the static city built by scene/build_city.py.

Randomisation is only over camera pose (where the drone is in the one city) and over
appearance (lighting, sky, fog, exposure, tint, and which Poly Haven surface textures each
class wears), never over geometry. Every pose is rendered with three passes: lit RGB (2x
supersampled, sun shadow map), semantic class ids, and optical-Z depth.

Surface detail is world-space triplanar: Kenney meshes only carry palette UVs. Each textured
class binds three texture slots A/B/C (albedo + normal); per-object variant and surface kind
come from the vertex colour written by build_city.py (R = variant, G = kind / 4).
"""

import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyray as rl
from PIL import Image
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
uniform sampler2D detailA; uniform sampler2D normalA;
uniform sampler2D detailB; uniform sampler2D normalB;
uniform sampler2D detailC; uniform sampler2D normalC;
uniform sampler2D shadowMap;
uniform int classMode; uniform float detailStrength; uniform vec3 photoColour;
uniform vec3 tileSize; uniform vec3 meanA; uniform vec3 meanB; uniform vec3 meanC;
#ifdef PRESENTATION_LIGHT
uniform vec3 tileHeight;
#endif
uniform mat4 lightVP; uniform float shadowTexel; uniform float shadowBias;
uniform vec3 viewPos; uniform vec3 sunDir; uniform vec3 sunColor;
uniform vec3 skyAmbient; uniform vec3 groundAmbient; uniform vec3 tint;
uniform vec3 fogColor; uniform float fogDensity; uniform float exposure;
out vec4 finalColor;

vec3 linear_rgb(vec3 c) {
    return mix(pow((c + 0.055) / 1.055, vec3(2.4)), c / 12.92, lessThanEqual(c, vec3(0.04045)));
}
vec3 display_rgb(vec3 c) {
    c = clamp(c, 0.0, 1.0);
    return mix(1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, 12.92 * c, lessThanEqual(c, vec3(0.0031308)));
}

vec3 triWeights(vec3 n) { vec3 w = pow(abs(n), vec3(4.0)); return w / (w.x + w.y + w.z); }
vec3 triSample(sampler2D t, vec3 p, vec3 w, vec2 s) {
#ifdef PRESENTATION_LIGHT
    return linear_rgb(texture(t, p.zy / s).rgb) * w.x
         + linear_rgb(texture(t, p.xz / s).rgb) * w.y
         + linear_rgb(texture(t, p.xy / s).rgb) * w.z;
#else
    return texture(t, p.zy / s).rgb * w.x + texture(t, p.xz / s).rgb * w.y + texture(t, p.xy / s).rgb * w.z;
#endif
}
// Whiteout-blended triplanar normal mapping (tangent spaces aligned with the projection axes).
vec3 triNormal(sampler2D t, vec3 p, vec3 n, vec3 w, vec2 s) {
    vec3 tx = texture(t, p.zy / s).xyz * 2.0 - 1.0;
    vec3 ty = texture(t, p.xz / s).xyz * 2.0 - 1.0;
    vec3 tz = texture(t, p.xy / s).xyz * 2.0 - 1.0;
    tx = vec3(tx.xy + n.zy, abs(tx.z) * n.x);
    ty = vec3(ty.xy + n.xz, abs(ty.z) * n.y);
    tz = vec3(tz.xy + n.xy, abs(tz.z) * n.z);
    return normalize(tx.zyx * w.x + ty.xzy * w.y + tz.xyz * w.z);
}
float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float vnoise(vec2 p) {
    vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i), hash(i + vec2(1, 0)), f.x), mix(hash(i + vec2(0, 1)), hash(i + vec2(1, 1)), f.x), f.y);
}

void main() {
    vec3 palette = (texture(texture0, fragTexCoord) * colDiffuse).rgb;
#ifdef PRESENTATION_LIGHT
    palette = linear_rgb(texture(texture0, fragTexCoord).rgb) * colDiffuse.rgb;
#endif
    vec3 n = normalize(fragNormal);
    if (!gl_FrontFacing) n = -n;
    vec3 geomN = n;
    vec3 albedo = palette;
    if (classMode != 0) {
        vec3 p = fragPosition, w = triWeights(n);
        float variant = fragColor.r, kind = floor(fragColor.g * 4.0 + 0.5);
        vec2 sA = vec2(tileSize.x), sB = vec2(tileSize.y), sC = vec2(tileSize.z);
#ifdef PRESENTATION_LIGHT
        sA.y = tileHeight.x; sB.y = tileHeight.y; sC.y = tileHeight.z;
#endif
        vec3 dA = triSample(detailA, p, w, sA), dB = triSample(detailB, p, w, sB), dC = triSample(detailC, p, w, sC);
        vec3 nA = triNormal(normalA, p, n, w, sA), nB = triNormal(normalB, p, n, w, sB), nC = triNormal(normalC, p, n, w, sC);
        vec3 sel;
        if (classMode == 1) {          // ground: two grasses blended at large scale, dirt patches
            float m = vnoise(p.xz / 35.0), d = smoothstep(0.62, 0.75, vnoise(p.xz / 18.0 + 7.3));
            sel = vec3((1.0 - d) * (1.0 - m), (1.0 - d) * m, d);
        } else if (classMode == 2) {   // road: pavement tiles, else one of two asphalts
            sel = kind > 2.5 ? vec3(0, 1, 0) : (variant < 0.5 ? vec3(1, 0, 0) : vec3(0, 0, 1));
        } else if (classMode == 3) {   // building: roofs, else one of two walls
            sel = n.y > 0.7 ? vec3(0, 0, 1) : (variant < 0.5 ? vec3(1, 0, 0) : vec3(0, 1, 0));
        } else if (classMode == 4) {   // vegetation: bark, else one of two foliages
            sel = kind > 1.5 ? vec3(0, 0, 1) : (variant < 0.5 ? vec3(1, 0, 0) : vec3(0, 1, 0));
        } else {                       // props
            sel = variant < 0.5 ? vec3(1, 0, 0) : vec3(0, 1, 0);
        }
        // Slots flagged in photoColour take the texture's own colour (roofs, dirt, bark); the rest
        // keep a base hue and gain only the texture's pattern (Poly Haven grass/leaf photos are dry
        // brown, and Kenney palettes carry window/trim detail worth keeping).
        vec3 base = palette;
        if (classMode == 1) base = mix(vec3(0.22, 0.36, 0.13), vec3(0.34, 0.44, 0.17), vnoise(p.xz / 60.0));
        if (classMode == 4) base = mix(vec3(0.17, 0.30, 0.10), vec3(0.33, 0.42, 0.14), variant);
#ifdef PRESENTATION_LIGHT
        if (classMode == 1 || classMode == 4) base = linear_rgb(base);
#endif
        vec3 cA = mix(base * mix(vec3(1.0), dA / max(meanA, vec3(0.02)), detailStrength), dA, photoColour.x);
        vec3 cB = mix(base * mix(vec3(1.0), dB / max(meanB, vec3(0.02)), detailStrength), dB, photoColour.y);
        vec3 cC = mix(base * mix(vec3(1.0), dC / max(meanC, vec3(0.02)), detailStrength), dC, photoColour.z);
        // Roofs share one texture per view; per-building brightness and warmth keep them from looking cloned.
        if (classMode == 3) cC *= mix(vec3(0.7, 0.72, 0.78), vec3(1.25, 1.15, 1.0), fract(variant * 7.31));
        albedo = cA * sel.x + cB * sel.y + cC * sel.z;
        n = normalize(nA * sel.x + nB * sel.y + nC * sel.z);
    }
#ifdef PRESENTATION_LIGHT
        // Preserve the faceted mesh silhouette and keep normal detail from dominating the light.
        n = normalize(mix(geomN, n, 0.55));
#endif
    float ndl = max(dot(n, sunDir), 0.0);
    float lit = 1.0;
    vec4 ls = lightVP * vec4(fragPosition, 1.0);
    vec3 sc = ls.xyz / ls.w * 0.5 + 0.5;
    if (all(greaterThan(sc, vec3(0.0))) && all(lessThan(sc, vec3(1.0)))) {
        float bias = shadowBias * (1.0 + 3.0 * (1.0 - max(dot(geomN, sunDir), 0.0)));
        float occ = 0.0;
        for (int i = -1; i <= 1; i++)
            for (int j = -1; j <= 1; j++)
                occ += (sc.z - bias > texture(shadowMap, sc.xy + vec2(i, j) * shadowTexel).r) ? 1.0 : 0.0;
        lit = 1.0 - occ / 9.0;
    }
    vec3 ambient = mix(groundAmbient, skyAmbient, 0.5 + 0.5 * n.y);
    vec3 h = normalize(normalize(viewPos - fragPosition) + sunDir);
    float spec = pow(max(dot(n, h), 0.0), 48.0) * 0.12 * step(0.0, dot(geomN, sunDir));
#ifdef PRESENTATION_LIGHT
    float facing = max(dot(geomN, sunDir), 0.0);
    // Normal maps should not light the back of a surface. Matte surfaces get a small, broad highlight.
    ndl *= smoothstep(0.0, 0.16, facing);
    spec = pow(max(dot(n, h), 0.0), 32.0) * 0.035 * ndl;
    vec3 c = (albedo * tint * (ambient + sunColor * ndl * lit) + sunColor * spec * lit) * exposure;
    float luminance = dot(c, vec3(0.2126, 0.7152, 0.0722));
    c /= 1.0 + 0.28 * luminance;
    c = mix(linear_rgb(fogColor), c, exp(-fogDensity * length(fragPosition - viewPos)));
    finalColor = vec4(display_rgb(c), 1.0);
#else
    vec3 c = (albedo * tint * (ambient + sunColor * ndl * lit) + sunColor * spec * lit) * exposure;
    c = mix(fogColor, c, exp(-fogDensity * length(fragPosition - viewPos)));
    finalColor = vec4(clamp(c, 0.0, 1.0), 1.0);
#endif
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
SHADOW_SIZE = 4096
SHADOW_EXTENT_M = 320.0  # side of the square shadowed region, centred ahead of the camera
SHADOW_BIAS_M = 0.15

# Texture slots per textured class: (role A, role B, role C) drawn from the Poly Haven pools.
CLASS_SLOTS = {
    "ground": ("grass", "grass", "dirt"),
    "road": ("asphalt", "pavement", "asphalt"),
    "building": ("wall", "wall", "roof"),
    "vegetation": ("foliage", "foliage", "bark"),
    "prop": ("metal", "concrete", "concrete"),
}
CLASS_MODE = {"ground": 1, "road": 2, "building": 3, "vegetation": 4, "prop": 5}
DETAIL_STRENGTH = {"ground": 1.0, "road": 0.85, "building": 0.8, "vegetation": 0.8, "prop": 0.7}
PHOTO_COLOUR = {"ground": (0, 0, 1), "road": (0, 0, 0), "building": (0, 0, 1), "vegetation": (0, 0, 1), "prop": (0, 0, 0)}
# Material map slots used as extra samplers (raylib binds maps[i] to texture unit i). Slots 7-9
# are bound as cubemaps and slot 11 aliases a vertex-attribute location, so they are avoided.
DETAIL_SLOTS = (
    (rl.MATERIAL_MAP_METALNESS, "detailA"), (rl.MATERIAL_MAP_NORMAL, "normalA"),
    (rl.MATERIAL_MAP_ROUGHNESS, "detailB"), (rl.MATERIAL_MAP_OCCLUSION, "normalB"),
    (rl.MATERIAL_MAP_EMISSION, "detailC"), (rl.MATERIAL_MAP_HEIGHT, "normalC"),
)  # fmt: skip
SHADOW_SLOT = (rl.MATERIAL_MAP_BRDF, "shadowMap")
N_MATERIAL_CHOICES = sum(len(s) for s in CLASS_SLOTS.values())


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


REGIMES = ("seen", "novel")


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
    materials: tuple[int, ...]  # raw choice per (class, slot) in CLASS_SLOTS order; resolved modulo pool size
    texture_split: str = "train"  # "train" pools or eval-only "heldout" pools (see materials.json)

    @staticmethod
    def canonical() -> "Appearance":
        sun = np.array([0.35, 0.85, 0.4])
        return Appearance(
            tuple(sun / np.linalg.norm(sun)), (0.85, 0.8, 0.72), (0.42, 0.48, 0.56), (0.28, 0.26, 0.23),
            (0.55, 0.7, 0.88), 0.002, 1.0, (1.0, 1.0, 1.0), (0,) * N_MATERIAL_CHOICES,
        )  # fmt: skip

    @staticmethod
    def random(rng: np.random.Generator, regime: str = "seen") -> "Appearance":
        """Task-valid variation: lighting, atmosphere, camera response and surface materials; geometry untouched.

        seen:  the training distribution (train texture pools, moderate sun/fog/exposure ranges).
        novel: eval-only: held-out textures, low sun (long shadows), dim light, heavy fog, and
               exposure/tint outside the seen ranges. Never used for training data.
        """
        if regime == "seen":
            elevation, intensity = rng.uniform(0.35, 1.45), rng.uniform(0.45, 1.1)
            fog = float(rng.uniform() ** 2 * 0.006)
            exposure = float(rng.uniform(0.75, 1.3))
            tint = rng.uniform(0.88, 1.12, 3)
            split = "train"
        elif regime == "novel":
            elevation, intensity = rng.uniform(0.1, 0.35), rng.uniform(0.25, 0.45)
            fog = float(rng.uniform(0.005, 0.008))  # above most seen fog (mean ~0.002), distant content still visible
            exposure = float(rng.uniform(0.55, 0.75) if rng.uniform() < 0.5 else rng.uniform(1.3, 1.55))
            tint = 1.0 + rng.choice([-1.0, 1.0], 3) * rng.uniform(0.12, 0.2, 3)
            split = "heldout"
        else:
            raise ValueError(f"unknown appearance regime {regime!r}; use {REGIMES}")
        azimuth = rng.uniform(0, 2 * math.pi)
        sun = (math.cos(elevation) * math.cos(azimuth), math.sin(elevation), math.cos(elevation) * math.sin(azimuth))
        warmth = rng.uniform(-1, 1)
        sun_color = np.array([1.0 + 0.25 * warmth, 0.95, 1.0 - 0.3 * warmth]) * intensity
        sky = np.clip(rng.uniform([0.25, 0.3, 0.35], [0.95, 0.95, 1.0]) * rng.uniform(0.4, 1.1), 0, 1)
        ambient = rng.uniform(0.15, 0.6)
        return Appearance(
            sun, tuple(sun_color), tuple(sky * ambient / max(sky.max(), 1e-3) + 0.05), tuple(rng.uniform(0.1, 0.35, 3)),
            tuple(sky), fog, exposure, tuple(float(t) for t in tint),
            tuple(int(v) for v in rng.integers(0, 1 << 16, N_MATERIAL_CHOICES)), split,
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
# Yaw is kept apart: in a grid city it is mostly visible only through sun/shadow direction, which the
# appearance randomisation changes, so a probe fit on the canonical view learns a lighting shortcut.
STATE_GROUPS = {
    "position": slice(0, 3), "yaw": slice(3, 5), "pitch_roll": slice(5, 7), "intrinsics": slice(7, 8), "clearance": slice(8, 10),
}  # fmt: skip
# Headline pose score: every dimension except yaw.
OBSERVABLE_STATE = [n for n in STATE_NAMES if not n.startswith("yaw")]


def pose_state(pose: Pose, clearance: float, depth: np.ndarray) -> np.ndarray:
    h, w = depth.shape
    forward = math.log(float(np.median(depth[h // 2 - h // 16 : h // 2 + h // 16, w // 2 - w // 16 : w // 2 + w // 16])))
    return np.array(
        [*pose.position, math.sin(pose.yaw), math.cos(pose.yaw), pose.pitch, pose.roll, pose.vfov, clearance, forward],
        dtype=np.float32,
    )


def _vec(values) -> object:
    return ffi.new(f"float[{len(values)}]", [float(v) for v in values])


def _srgb_u8(c: int) -> int:
    v = c / 255.0
    return round(255 * (12.92 * v if v <= 0.0031308 else 1.055 * v ** (1 / 2.4) - 0.055))


@dataclass
class SurfaceTexture:
    albedo: object
    normal: object
    size_m: float
    height_m: float
    mean: tuple[float, float, float]


class CityRenderer:
    def __init__(
        self, scene_dir: Path, width: int, height: int, textures_dir: Path = Path("assets/polyhaven"),
        *, presentation: bool = False,
    ):
        # Preserve the benchmark image distribution; the grade is for qualitative renders only.
        self.presentation = presentation
        self.shadow_far = 640.0 if presentation else FAR
        self.layout = CityLayout(scene_dir)
        self.width, self.height = width, height
        rl.set_trace_log_level(rl.LOG_WARNING)
        rl.set_config_flags(rl.FLAG_WINDOW_HIDDEN)
        rl.init_window(width, height, "visenc-city")
        self.rgb_target = rl.load_render_texture(width * SUPERSAMPLE, height * SUPERSAMPLE)
        self.target = rl.load_render_texture(width, height)
        self.shadow = self._load_shadow_target()
        fragment = LIT.replace("#version 330", "#version 330\n#define PRESENTATION_LIGHT") if presentation else LIT
        self.lit = rl.load_shader_from_memory(VERTEX, fragment)
        if self.lit.id == rl.rl_get_shader_id_default():
            raise RuntimeError("city lighting shader did not compile")
        self.semantic = rl.load_shader_from_memory(VERTEX, SEMANTIC)
        self.depth = rl.load_shader_from_memory(VERTEX, DEPTH)
        for shader in (self.lit, self.semantic, self.depth):
            shader.locs[rl.SHADER_LOC_MATRIX_MODEL] = rl.get_shader_location(shader, "matModel")
            shader.locs[rl.SHADER_LOC_MATRIX_NORMAL] = rl.get_shader_location(shader, "matNormal")
        for slot, name in (*DETAIL_SLOTS, SHADOW_SLOT):
            self.lit.locs[rl.SHADER_LOC_MAP_DIFFUSE + slot] = rl.get_shader_location(self.lit, name)
        self.models: dict[int, object] = {}
        default_texture = rl.rl_get_texture_id_default()
        for class_id, name in enumerate(self.layout.classes):
            path = scene_dir / f"{name}.glb"
            if not path.exists():
                continue
            model = rl.load_model(str(path))
            for i in range(model.materialCount):
                m = model.materials[i].maps[rl.MATERIAL_MAP_ALBEDO]
                if not presentation and m.texture.id == default_texture:
                    # Untextured glTF materials carry a linear base colour; the shader works in sRGB values.
                    m.color = rl.Color(_srgb_u8(m.color.r), _srgb_u8(m.color.g), _srgb_u8(m.color.b), m.color.a)
            self.models[class_id] = model
        self.pools = self._load_textures(textures_dir)

    def _load_shadow_target(self):
        fbo = rl.rl_load_framebuffer()
        rl.rl_enable_framebuffer(fbo)
        depth_id = rl.rl_load_texture_depth(SHADOW_SIZE, SHADOW_SIZE, False)
        rl.rl_framebuffer_attach(fbo, depth_id, rl.RL_ATTACHMENT_DEPTH, rl.RL_ATTACHMENT_TEXTURE2D, 0)
        if not rl.rl_framebuffer_complete(fbo):
            raise RuntimeError("shadow framebuffer incomplete")
        rl.rl_disable_framebuffer()
        depth = rl.Texture(depth_id, SHADOW_SIZE, SHADOW_SIZE, 1, 19)
        return rl.RenderTexture(fbo, rl.Texture(0, SHADOW_SIZE, SHADOW_SIZE, 1, 7), depth)

    def _load_textures(self, root: Path) -> dict[str, dict[str, list[SurfaceTexture]]]:
        """pools[split][role]: 'train' excludes the eval-only textures listed in materials.json 'heldout'."""
        spec = json.loads((root / "materials.json").read_text())
        if "heldout" not in spec:
            raise ValueError(f"{root}/materials.json has no 'heldout' split; re-run scene/fetch_polyhaven.py")
        loaded: dict[str, SurfaceTexture] = {}
        pools: dict[str, dict[str, list[SurfaceTexture]]] = {"train": {}, "heldout": {}}
        for role, ids in spec["roles"].items():
            held = set(spec["heldout"][role])
            pools["train"][role], pools["heldout"][role] = [], []
            for tid in ids:
                if tid not in loaded:
                    textures = []
                    for name in ("diff.jpg", "nor.jpg"):
                        tex = rl.load_texture(str(root / tid / name))
                        rl.gen_texture_mipmaps(ffi.addressof(tex))
                        rl.set_texture_filter(tex, rl.TEXTURE_FILTER_TRILINEAR)
                        rl.rl_texture_parameters(tex.id, rl.RL_TEXTURE_FILTER_ANISOTROPIC, 8)
                        rl.set_texture_wrap(tex, rl.TEXTURE_WRAP_REPEAT)
                        textures.append(tex)
                    with Image.open(root / tid / "diff.jpg") as image:
                        if self.presentation:
                            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)[::8, ::8] / 255.0
                            mean = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4).mean((0, 1))
                        else:
                            mean = np.asarray(image.convert("RGB").reduce(16), dtype=np.float64).mean((0, 1)) / 255
                    size_x, size_y = map(float, spec["textures"][tid]["size_m"])
                    loaded[tid] = SurfaceTexture(textures[0], textures[1], size_x, size_y, tuple(mean))
                pools["heldout" if tid in held else "train"][role].append(loaded[tid])
        self._all_textures = list(loaded.values())
        return pools

    def close(self) -> None:
        for model in self.models.values():
            for i in range(model.materialCount):
                for slot, _ in (*DETAIL_SLOTS, SHADOW_SLOT):
                    model.materials[i].maps[slot].texture = rl.Texture(0, 0, 0, 0, 0)  # not owned by the model
            rl.unload_model(model)
        for t in self._all_textures:
            rl.unload_texture(t.albedo)
            rl.unload_texture(t.normal)
        for shader in (self.lit, self.semantic, self.depth):
            rl.unload_shader(shader)
        rl.rl_unload_framebuffer(self.shadow.id)
        rl.rl_unload_texture(self.shadow.depth.id)
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

    def _resolve_materials(self, look: Appearance) -> dict[str, list[SurfaceTexture]]:
        """Pick each class's three textures from the look's split; slots sharing a role get distinct
        textures while the pool allows (held-out pools may hold a single texture)."""
        pools = self.pools[look.texture_split]
        choices = iter(look.materials)
        out = {}
        for cls, roles in CLASS_SLOTS.items():
            used: dict[str, set[int]] = {}
            picked = []
            for role in roles:
                pool = pools[role]
                idx = next(choices) % len(pool)
                taken = used.setdefault(role, set())
                while idx in taken and len(taken) < len(pool):
                    idx = (idx + 1) % len(pool)
                taken.add(idx)
                picked.append(pool[idx])
            out[cls] = picked
        return out

    def _render_models(self, shader, per_class=None) -> None:
        for class_id, model in self.models.items():
            if per_class is not None:
                per_class(class_id, model)
            for i in range(model.materialCount):
                model.materials[i].shader = shader
            rl.draw_model(model, (0.0, 0.0, 0.0), 1.0, rl.WHITE)

    def _draw(self, target, pose: Pose, clear, shader, per_class=None, extra: Callable[[], None] | None = None) -> np.ndarray:
        forward, up = pose.basis()
        camera = rl.Camera3D(
            tuple(pose.position), tuple(pose.position + forward), tuple(up), math.degrees(pose.vfov), rl.CAMERA_PERSPECTIVE
        )
        rl.begin_texture_mode(target)
        rl.clear_background(clear)
        rl.rl_set_clip_planes(NEAR, FAR)
        rl.begin_mode_3d(camera)
        self._render_models(shader, per_class)
        if extra is not None:
            extra()
        rl.end_mode_3d()
        rl.end_texture_mode()
        image = rl.load_image_from_texture(target.texture)
        w, h = target.texture.width, target.texture.height
        pixels = np.frombuffer(ffi.buffer(image.data, w * h * 4), dtype=np.uint8).reshape(h, w, 4)[::-1].copy()
        rl.unload_image(image)
        return pixels

    def _render_shadow_map(self, pose: Pose, sun: np.ndarray):
        """Orthographic sun depth over a square ahead of the camera; returns the light view-projection."""
        forward, _ = pose.basis()
        flat = np.array([forward[0], 0.0, forward[2]])
        norm = np.linalg.norm(flat)
        ahead = flat / norm * (SHADOW_EXTENT_M * 0.3) if norm > 1e-6 else np.zeros(3)
        centre = np.array([pose.position[0], 0.0, pose.position[2]]) + ahead
        up = (0.0, 0.0, 1.0) if abs(sun[1]) > 0.99 else (0.0, 1.0, 0.0)
        camera = rl.Camera3D(tuple(centre + sun * (self.shadow_far / 2)), tuple(centre), up,
                             SHADOW_EXTENT_M, rl.CAMERA_ORTHOGRAPHIC)
        rl.begin_texture_mode(self.shadow)
        rl.clear_background(rl.WHITE)
        rl.rl_set_clip_planes(NEAR, self.shadow_far)
        rl.begin_mode_3d(camera)
        light_vp = rl.matrix_multiply(rl.rl_get_matrix_modelview(), rl.rl_get_matrix_projection())
        self._render_models(self.semantic)
        rl.end_mode_3d()
        rl.end_texture_mode()
        return light_vp

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
            lambda c, _m: self._set(self.semantic, "classColor", [c / 255.0, 0.0, 0.0, 1.0]),
        )[..., 0]  # fmt: skip
        return semantic, depth

    def render_rgb(self, pose: Pose, look: Appearance, *, extra: Callable[[], None] | None = None) -> np.ndarray:
        """uint8 [H,W,3], box-downsampled from a SUPERSAMPLE x render."""
        s = self.lit
        sun = np.asarray(look.sun_dir, dtype=np.float64)
        sun = sun / np.linalg.norm(sun)
        light_vp = self._render_shadow_map(pose, sun)
        rl.set_shader_value_matrix(s, rl.get_shader_location(s, "lightVP"), light_vp)
        self._set(s, "shadowTexel", [1.0 / SHADOW_SIZE])
        self._set(s, "shadowBias", [SHADOW_BIAS_M / (self.shadow_far - NEAR)])
        self._set(s, "viewPos", pose.position)
        self._set(s, "sunDir", sun)
        for name, value in [
            ("sunColor", look.sun_color), ("skyAmbient", look.sky_ambient), ("groundAmbient", look.ground_ambient),
            ("tint", look.tint), ("fogColor", look.sky), ("fogDensity", [look.fog_density]), ("exposure", [look.exposure]),
        ]:  # fmt: skip
            self._set(s, name, value)
        materials = self._resolve_materials(look)
        mode_loc = rl.get_shader_location(s, "classMode")

        def bind(class_id: int, model) -> None:
            cls = self.layout.classes[class_id]
            mode = CLASS_MODE.get(cls, 0)
            rl.set_shader_value(s, mode_loc, ffi.new("int *", mode), rl.SHADER_UNIFORM_INT)
            picked = materials.get(cls)
            if picked is not None:
                self._set(s, "detailStrength", [DETAIL_STRENGTH[cls]])
                self._set(s, "photoColour", PHOTO_COLOUR[cls])
                self._set(s, "tileSize", [t.size_m for t in picked])
                if self.presentation:
                    self._set(s, "tileHeight", [t.height_m for t in picked])
                for t, name in zip(picked, ("meanA", "meanB", "meanC")):
                    self._set(s, name, t.mean)
            for i in range(model.materialCount):
                maps = model.materials[i].maps
                if picked is not None:
                    for k, t in enumerate(picked):
                        maps[DETAIL_SLOTS[2 * k][0]].texture = t.albedo
                        maps[DETAIL_SLOTS[2 * k + 1][0]].texture = t.normal
                maps[SHADOW_SLOT[0]].texture = self.shadow.depth

        sky = tuple(int(round(255 * v)) for v in look.sky) + (255,)
        big = self._draw(self.rgb_target, pose, sky, s, bind, extra)[..., :3].astype(np.float32)
        h, w = self.height, self.width
        small = big.reshape(h, SUPERSAMPLE, w, SUPERSAMPLE, 3).mean((1, 3))
        return np.round(small).astype(np.uint8)
