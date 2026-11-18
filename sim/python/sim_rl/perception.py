"""Perception heads on frozen patch tokens: per-patch monocular depth and object class.

The spatial memory places what the camera sees on a map, which needs metric depth from one
camera, and counting needs to know which patches show a car. Both come from small heads on the
frozen encoder's patch grid, supervised by the simulator's per-patch labels (ABI v7 fields 17-18:
depth of the patch's object and the layout object hit). Labels never reach the actor.

Monocular depth gets a geometric prior the drone has without any extra sensor: from the altimeter
height and the gravity direction, every patch ray's distance to the ground plane (``ground_prior``).
Objects stand on the ground, so this anchors metric scale.

Frames come from flown trajectories (smooth random exploratory velocity commands, every
``stride`` steps across many resets) rather than start views, so the heads see the altitudes,
tilts, motion blur and distances the policy flies at. ``perception_command`` trains one head per
encoder on the same frames and reports held-out depth and class quality.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .curriculum import fixed_lanes
from .native import PATCH, CudaEnv

CLASSES = ("none", "building", "vegetation", "vehicle", "prop")
ENCODE_CHUNK = 256
PRIOR_WIDTH = 3                  # per patch: log ground distance (scaled), ray elevation, ray hits ground
GROUND_FAR = 120.0               # metres: rays above the horizon or beyond this report the far plane
HEIGHT_SCALE = 10.0              # native sensor 9 normalisation
# Exploratory commands: Ornstein-Uhlenbeck on [forward, right, up, yaw] with a forward bias.
OU_THETA, OU_SIGMA, OU_BIAS = 0.15, 0.35, (0.35, 0.0, 0.05, 0.0)


def patch_rays(grid_h, grid_w, camera, device):
    """[P, 3] patch-centre rays in the body frame (x right, y up, -z forward) with unit component
    along the uptilted camera axis, so ray * planar depth is the hit offset."""
    tilt, half = camera["uptilt_rad"], math.tan(camera["vertical_fov_rad"] / 2)
    forward = torch.tensor([0.0, math.sin(tilt), -math.cos(tilt)], device=device)
    up = torch.tensor([0.0, math.cos(tilt), math.sin(tilt)], device=device)
    right = torch.tensor([1.0, 0.0, 0.0], device=device)
    rows, columns = torch.meshgrid(torch.arange(grid_h, device=device), torch.arange(grid_w, device=device),
                                   indexing="ij")
    nx = ((columns.flatten() + 0.5) / grid_w * 2 - 1) * half * camera["aspect"]
    ny = (1 - (rows.flatten() + 0.5) / grid_h * 2) * half
    return forward + nx[:, None] * right + ny[:, None] * up


def ground_prior(proprio, rays):
    """[n, P, PRIOR_WIDTH] from the drone's own sensors: each ray's planar distance to the ground plane
    below (altimeter height, gravity direction), its elevation, and whether it reaches the ground."""
    up = F.normalize(-proprio[:, 0:3], dim=-1)
    height = (proprio[:, 9] * HEIGHT_SCALE).clamp_min(0.1)
    elevation = rays @ up.T                                        # [P, n]: ray . up per frame
    down = (elevation < -1e-3).T
    distance = (height[:, None] / (-elevation.T).clamp_min(1e-3)).clamp(0.5, GROUND_FAR)
    distance = torch.where(down, distance, torch.full_like(distance, GROUND_FAR))
    return torch.stack((distance.log() / math.log(GROUND_FAR), elevation.T, down.float()), -1)


@dataclass
class LabelledFrames:
    images: torch.Tensor   # uint8 [F, 3, H, W]
    prior: torch.Tensor    # float32 [F, P, PRIOR_WIDTH], from the frame's sensors
    depth: torch.Tensor    # float32 [F, P], 0 = no geometry
    classes: torch.Tensor  # long [F, P], index into CLASSES


def object_classes(env):
    """long [objects]: CLASSES index of every layout object."""
    layout = json.loads((env.city_path / "layout.json").read_text())["objects"]
    return torch.tensor([CLASSES.index(o["class"]) if o["class"] in CLASSES else 0 for o in layout],
                        device=env.device)


@torch.no_grad()
def collect_labelled(env, count, seed, stride, appearance):
    """``count`` frames from exploratory flight, one every ``stride`` control steps per env, with
    per-patch depth and class labels. Must run inside ``env.execution()``."""
    n, device = env.n, env.device
    lookup = object_classes(env)
    rounds = -(-count // n)
    images = torch.empty((rounds * n, 3, env.height, env.width), dtype=torch.uint8, device=device)
    patches = env.patch_depth.shape[1]
    rays = patch_rays(env.height // PATCH, env.width // PATCH, env.metadata["camera"], device)
    prior = torch.empty((rounds * n, patches, PRIOR_WIDTH), device=device)
    depth = torch.empty((rounds * n, patches), device=device)
    classes = torch.empty((rounds * n, patches), dtype=torch.long, device=device)
    env.write_curriculum(fixed_lanes(device, appearance=appearance))
    env.reset(seed)
    generator = torch.Generator(device=device).manual_seed(seed)
    bias = torch.tensor(OU_BIAS, device=device)
    actions = bias.expand(n, -1).clone()
    for round_index in range(rounds):
        for _ in range(stride):
            noise = torch.randn((n, 4), device=device, generator=generator)
            actions.add_(OU_THETA * (bias - actions) + OU_SIGMA * noise).clamp_(-1, 1)
            env.step(actions)
        rows = slice(round_index * n, (round_index + 1) * n)
        images[rows].copy_((env.obs * 255).round().to(torch.uint8))
        prior[rows].copy_(ground_prior(env.proprio, rays))
        depth[rows].copy_(env.patch_depth)
        objects = env.patch_object.long()
        classes[rows].copy_(torch.where(objects >= 0, lookup[objects.clamp_min(0)], torch.zeros_like(objects)))
    env.stream.synchronize()  # Explicit collection boundary.
    return LabelledFrames(images[:count].contiguous(), prior[:count].contiguous(), depth[:count].contiguous(),
                          classes[:count].contiguous())


class PatchHeads(nn.Module):
    """Per-patch log depth, "has geometry" logit and object-class logits from the token grid and the
    per-patch ground prior (1x1 projection, two 3x3 context layers, the second dilated)."""

    def __init__(self, dim, grid_h, grid_w, width=192):
        super().__init__()
        self.grid_h, self.grid_w = grid_h, grid_w
        self.trunk = nn.Sequential(nn.Conv2d(dim + PRIOR_WIDTH, width, 1), nn.GELU(),
                                   nn.Conv2d(width, width, 3, padding=1), nn.GELU(),
                                   nn.Conv2d(width, width, 3, padding=2, dilation=2), nn.GELU())
        self.depth = nn.Conv2d(width, 2, 1)
        self.classes = nn.Conv2d(width, len(CLASSES), 1)

    def forward(self, tokens, prior):
        """tokens [n, P, D], prior [n, P, PRIOR_WIDTH] -> (log depth [n, P], geometry logit [n, P],
        class logits [n, P, C])."""
        n = tokens.shape[0]
        grid = torch.cat((tokens.float(), prior), -1).transpose(1, 2).reshape(n, -1, self.grid_h, self.grid_w)
        hidden = self.trunk(grid)
        depth = self.depth(hidden).flatten(2)
        return depth[:, 0], depth[:, 1], self.classes(hidden).flatten(2).transpose(1, 2)

    @torch.no_grad()
    def predict(self, tokens, prior):
        """(metric depth [n, P] with 0 where no geometry is predicted, class probabilities [n, P, C])."""
        log_depth, geometry, logits = self(tokens, prior)
        return torch.where(geometry > 0, log_depth.exp(), torch.zeros_like(log_depth)), logits.softmax(-1)


def head_loss(heads, tokens, prior, depth, classes):
    log_depth, geometry, logits = heads(tokens, prior)
    has = depth > 0
    depth_loss = (log_depth - depth.clamp_min(1e-3).log()).abs()[has].mean() if has.any() else log_depth.sum() * 0
    return (depth_loss + F.binary_cross_entropy_with_logits(geometry, has.float())
            + F.cross_entropy(logits.flatten(0, 1), classes.flatten()))


@torch.no_grad()
def encode(features, images):
    tokens = torch.empty((images.shape[0], images.shape[2] // PATCH * (images.shape[3] // PATCH), features.dim),
                         dtype=torch.bfloat16, device=images.device)
    for begin in range(0, images.shape[0], ENCODE_CHUNK):
        tokens[begin:begin + ENCODE_CHUNK].copy_(features.encode(images[begin:begin + ENCODE_CHUNK].float().div_(255))[1])
    return tokens


@torch.no_grad()
def evaluate(heads, tokens, frames):
    """Depth: abs-rel and delta<1.25 on patches with geometry, geometry accuracy; classes: per-class
    IoU, mIoU, and vehicle patch recall/precision."""
    depth_rows, has_rows, class_rows = [], [], []
    for begin in range(0, tokens.shape[0], ENCODE_CHUNK):
        log_depth, geometry, logits = heads(tokens[begin:begin + ENCODE_CHUNK], frames.prior[begin:begin + ENCODE_CHUNK])
        depth_rows.append(log_depth.exp())
        has_rows.append(geometry > 0)
        class_rows.append(logits.argmax(-1))
    predicted, has, label = torch.cat(depth_rows), torch.cat(has_rows), torch.cat(class_rows)
    truth = frames.depth
    valid = truth > 0
    ratio = predicted[valid] / truth[valid]
    iou = {}
    for index, name in enumerate(CLASSES):
        p, t = label == index, frames.classes == index
        union = (p | t).sum().item()
        iou[name] = (p & t).sum().item() / union if union else None
    vehicle_p, vehicle_t = label == CLASSES.index("vehicle"), frames.classes == CLASSES.index("vehicle")
    return {
        "depth_abs_rel": ((predicted[valid] - truth[valid]).abs() / truth[valid]).mean().item(),
        "depth_delta_1.25": (torch.maximum(ratio, 1 / ratio) < 1.25).float().mean().item(),
        "geometry_accuracy": (has == valid).float().mean().item(),
        "class_iou": iou,
        "class_miou": sum(v for v in iou.values() if v is not None) / sum(v is not None for v in iou.values()),
        "vehicle_recall": ((vehicle_p & vehicle_t).sum() / vehicle_t.sum().clamp_min(1)).item(),
        "vehicle_precision": ((vehicle_p & vehicle_t).sum() / vehicle_p.sum().clamp_min(1)).item(),
        "frames": tokens.shape[0],
    }


def train_heads(tokens, frames, grid, epochs, batch_size, learning_rate, seed):
    torch.manual_seed(seed)
    heads = PatchHeads(tokens.shape[-1], *grid).to(tokens.device)
    optimizer = torch.optim.AdamW(heads.parameters(), lr=learning_rate, weight_decay=1e-4)
    steps = epochs * math.ceil(tokens.shape[0] / batch_size)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, learning_rate, total_steps=steps)
    generator = torch.Generator(device=tokens.device).manual_seed(seed)
    for _ in range(epochs):
        order = torch.randperm(tokens.shape[0], device=tokens.device, generator=generator)
        for begin in range(0, tokens.shape[0], batch_size):
            ids = order[begin:begin + batch_size]
            loss = head_loss(heads, tokens[ids], frames.prior[ids], frames.depth[ids], frames.classes[ids])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            schedule.step()
    return heads.eval()


def perception_command(args, started):
    from .cli import TEST_SEED, TEST_STYLE, TRAIN_STYLE, emit
    from .features import FrozenFeatures
    device = torch.device("cuda:0")
    splits = {}
    for name, style, seed, count in (("train", TRAIN_STYLE, args.seed, args.train_frames),
                                     ("test", TEST_STYLE, TEST_SEED, args.test_frames)):
        with CudaEnv(args.library, args.num_envs, seed, args.width, args.height, style, "velocity", args.city,
                     start_distance=tuple(args.start_distance)) as env, env.execution():
            splits[name] = collect_labelled(env, count, seed, args.stride, args.appearance)
            grid = (env.height // PATCH, env.width // PATCH)
    print(json.dumps({"collected": {k: v.images.shape[0] for k, v in splits.items()},
                      "seconds": round(time.monotonic() - started, 1)}), flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    results = {}
    for spec in args.encoders:
        features = FrozenFeatures(spec, device, args.encoder_precision)
        train_tokens = encode(features, splits["train"].images)
        heads = train_heads(train_tokens, splits["train"], grid, args.epochs, args.batch_size, args.learning_rate,
                            args.seed)
        del train_tokens
        results[features.spec] = evaluate(heads, encode(features, splits["test"].images), splits["test"])
        name = str(features.spec).removeprefix("run:runs/").replace(":", "-").replace("/", "-")
        torch.save({"state_dict": heads.state_dict(), "encoder": features.spec, "encoder_fingerprint": features.fingerprint,
                    "preprocessing": features.preprocessing, "grid": grid, "classes": CLASSES},
                   args.out / f"{name}.pt")
        print(json.dumps({features.spec: results[features.spec]}), flush=True)
        del features, heads
        torch.cuda.empty_cache()
    emit({"train_frames": args.train_frames, "test_frames": args.test_frames, "stride": args.stride,
          "start_distance": args.start_distance, "appearance": args.appearance, "epochs": args.epochs,
          "batch_size": args.batch_size, "learning_rate": args.learning_rate, "seed": args.seed,
          "classes": CLASSES, "encoders": results}, args.out / "perception.json")


def load_heads(path, features, device):
    """Heads trained on ``features``' exact encoder (fingerprint and numerics checked)."""
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if (saved["encoder_fingerprint"], saved["preprocessing"]) != (features.fingerprint, features.preprocessing):
        raise ValueError(f"{path} was trained on a different encoder or precision")
    heads = PatchHeads(features.dim, *saved["grid"])
    heads.load_state_dict(saved["state_dict"])
    return heads.to(device).eval().requires_grad_(False)
