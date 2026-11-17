"""Offline goal-localization probe: frozen encoder features -> the policy's TargetSaliency head.

Frames are rendered once per run from a random-action rollout (pre-action observations) and
kept on the GPU as uint8; every encoder sees the identical frames, reset seeds and protocol.
The probe trains only a fresh TargetSaliency (the exact module the visual policy uses) with
the policy's auxiliary saliency loss, and reports held-out visibility AUROC/AP, exact- and
within-one-patch localization, and pixel error of the predicted patch (patch = 16 px).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .curriculum import BUCKETS, UNIFORM_WEIGHTS
from .model import NDC_LANES, VISIBLE_LANE, TargetSaliency, saliency_loss
from .native import DIAGNOSTICS_WIDTH, PRIVILEGED_WIDTH
from .stats import auroc, average_precision

PATCH = 16
ENCODE_CHUNK = 256
BUCKET_LANE = 15
INITIAL_DISTANCE_LANE = 20
DISTANCE_BINS = ((0.0, 10.0, "<10 m"), (10.0, 20.0, "10-20 m"), (20.0, math.inf, ">=20 m"))
PROTOCOL_TRAIN_FRAMES, PROTOCOL_TEST_FRAMES = 8192, 4096


@dataclass
class ProbeFrames:
    images: torch.Tensor       # uint8 [F, 3, H, W]
    privileged: torch.Tensor   # float32 [F, PRIVILEGED_WIDTH]
    diagnostics: torch.Tensor  # float32 [F, DIAGNOSTICS_WIDTH]
    designs: torch.Tensor      # long [F]
    goal_images: torch.Tensor  # float32 [K, 3, H, W]


@dataclass
class EncodedSplit:
    cls: torch.Tensor          # float32 [F, D]
    tokens: torch.Tensor       # bfloat16 [F, P, D]
    goal: torch.Tensor         # float32 [F, 2D]
    privileged: torch.Tensor
    diagnostics: torch.Tensor


@torch.no_grad()
def collect_frames(env, count, seed, generator_seed):
    """Start views of independent catalog episodes under the uniform bucket curriculum.

    Each step resets every env with a fresh seed, so frames are distinct episodes rather
    than correlated rollout frames (a 64-step rollout of 128 envs covers only ~128 scenes,
    which let probes memorize). Must run inside ``env.execution()``. Returns exactly
    ``count`` frames. ``generator_seed`` offsets the reset seeds so splits never collide.
    """
    n, device = env.n, env.device
    steps = -(-count // n)
    total = steps * n
    images = torch.empty((total, 3, env.height, env.width), dtype=torch.uint8, device=device)
    privileged = torch.empty((total, PRIVILEGED_WIDTH), device=device)
    diagnostics = torch.empty((total, DIAGNOSTICS_WIDTH), device=device)
    designs = torch.empty(total, dtype=torch.long, device=device)
    lanes = torch.tensor(UNIFORM_WEIGHTS + (0.0,), device=device, dtype=torch.float32)
    env.write_curriculum(lanes)
    for step in range(steps):
        env.reset(seed + generator_seed * 1_000_003 + step)
        rows = slice(step * n, (step + 1) * n)
        # Native observations are RGBA8/255 converted to float; the round trip is exact.
        images[rows].copy_((env.obs * 255).round().to(torch.uint8))
        privileged[rows].copy_(env.privileged)
        diagnostics[rows].copy_(env.diagnostics)
        designs[rows].copy_(env.design.long())
    env.stream.synchronize()  # Explicit probe collection boundary.
    return ProbeFrames(images[:count].contiguous(), privileged[:count].contiguous(),
                       diagnostics[:count].contiguous(), designs[:count].contiguous(),
                       env.goal_images.clone())


@torch.no_grad()
def encode_frames(features, frames):
    """Frozen-encoder pass over stored frames (decoded uint8 -> float [0,1]), chunked."""
    count = frames.images.shape[0]
    height, width = frames.images.shape[2], frames.images.shape[3]
    patches = (height // PATCH) * (width // PATCH)
    cls = torch.empty((count, features.dim), device=frames.images.device)
    tokens = torch.empty((count, patches, features.dim), dtype=torch.bfloat16, device=frames.images.device)
    for begin in range(0, count, ENCODE_CHUNK):
        end = min(begin + ENCODE_CHUNK, count)
        chunk_cls, chunk_tokens = features.encode(frames.images[begin:end].float().div_(255))
        cls[begin:end].copy_(chunk_cls)
        tokens[begin:end].copy_(chunk_tokens)
    goal = features.goal_table(frames.goal_images)[frames.designs]
    return EncodedSplit(cls, tokens, goal, frames.privileged, frames.diagnostics)


def train_probe(encoded, grid_h, grid_w, epochs, batch_size, learning_rate, generator):
    """Trains a fresh TargetSaliency on the training split; returns (saliency, history)."""
    device = encoded.cls.device
    saliency = TargetSaliency(encoded.tokens.shape[-1], encoded.goal.shape[-1], grid_h, grid_w).to(device)
    optimizer = torch.optim.Adam(saliency.parameters(), lr=learning_rate, weight_decay=1e-4)
    count = encoded.cls.shape[0]
    history = []
    for epoch in range(epochs):
        order = torch.randperm(count, device=device, generator=generator)
        loss_total = torch.zeros((), device=device)
        accuracy_total = torch.zeros((), device=device)
        batches = 0
        for begin in range(0, count, batch_size):
            ids = order[begin:begin + batch_size]
            _, logits = saliency(encoded.goal[ids], encoded.cls[ids], encoded.tokens[ids])
            loss, accuracy = saliency_loss(logits, encoded.privileged[ids], grid_h, grid_w)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_total += loss.detach()
            accuracy_total += accuracy
            batches += 1
        history.append({"epoch": epoch + 1, "saliency_loss": float(loss_total) / batches,
                        "train_patch_accuracy": float(accuracy_total) / batches})
    return saliency, history


@torch.no_grad()
def predict(saliency, encoded, width, height):
    """Host lists: visibility score, visible flags, patch hit/neighbour, pixel error, distance, bucket."""
    grid_h, grid_w = saliency.grid_h, saliency.grid_w
    rows = []
    for begin in range(0, encoded.cls.shape[0], ENCODE_CHUNK):
        part = slice(begin, begin + ENCODE_CHUNK)
        rows.append(saliency(encoded.goal[part], encoded.cls[part], encoded.tokens[part])[1])
    logits = torch.cat(rows)
    # In-view score: log-odds of "some patch" against "not in view".
    score = torch.logsumexp(logits[:, :-1], dim=-1) - logits[:, -1]
    predicted = logits[:, :-1].argmax(-1)
    target = encoded.privileged[:, NDC_LANES]
    column = ((target[:, 0] + 1) / 2 * grid_w).floor().clamp(0, grid_w - 1)
    row = ((1 - target[:, 1]) / 2 * grid_h).floor().clamp(0, grid_h - 1)
    hit = (predicted // grid_w == row.long()) & (predicted % grid_w == column.long())
    near = ((predicted // grid_w - row.long()).abs() <= 1) & ((predicted % grid_w - column.long()).abs() <= 1)
    # Predicted patch centre against the exact target centre, in pixels (NDC y up, rows down).
    predicted_px = torch.stack((((predicted % grid_w).float() + 0.5) * PATCH,
                                ((predicted // grid_w).float() + 0.5) * PATCH), dim=-1)
    target_px = torch.stack(((target[:, 0] + 1) / 2 * width, (1 - target[:, 1]) / 2 * height), dim=-1)
    error = (predicted_px - target_px).norm(dim=-1)
    visible = encoded.privileged[:, VISIBLE_LANE] > 0.5
    return {"logit": score.tolist(), "visible": visible.tolist(), "hit": hit.tolist(), "near": near.tolist(),
            "error_px": error.tolist(),
            "distance": encoded.diagnostics[:, INITIAL_DISTANCE_LANE].tolist(),
            "bucket": encoded.diagnostics[:, BUCKET_LANE].round().long().tolist()}


def _group(prediction, mask):
    logits = [value for value, keep in zip(prediction["logit"], mask) if keep]
    visible = [value for value, keep in zip(prediction["visible"], mask) if keep]
    errors = [value for value, keep, seen in zip(prediction["error_px"], mask, prediction["visible"])
              if keep and seen]
    hits = [value for value, keep, seen in zip(prediction["hit"], mask, prediction["visible"]) if keep and seen]
    near = [value for value, keep, seen in zip(prediction["near"], mask, prediction["visible"]) if keep and seen]
    result = {"frames": len(logits), "visible_fraction": (sum(visible) / len(visible)) if visible else None,
              "visibility_auroc": auroc(logits, visible), "visibility_ap": average_precision(logits, visible),
              "visible_frames": len(errors),
              "patch_accuracy": (sum(hits) / len(hits)) if hits else None,
              "within_one_patch": (sum(near) / len(near)) if near else None}
    if errors:
        ordered = sorted(errors)
        result["error_px_mean"] = sum(errors) / len(errors)
        result["error_px_median"] = ordered[len(ordered) // 2]
        result["error_px_p90"] = ordered[min(len(ordered) - 1, math.ceil(0.9 * len(ordered)) - 1)]
        result["error_patches_mean"] = result["error_px_mean"] / PATCH
    else:
        result["error_px_mean"] = result["error_px_median"] = result["error_px_p90"] = None
        result["error_patches_mean"] = None
    return result


def summarize_split(prediction):
    everything = [True] * len(prediction["logit"])
    result = _group(prediction, everything)
    result["by_bucket"] = {}
    for index, name in enumerate(BUCKETS):
        result["by_bucket"][name] = _group(prediction, [bucket == index for bucket in prediction["bucket"]])
    result["by_initial_distance"] = {}
    for low, high, label in DISTANCE_BINS:
        mask = [low <= distance < high for distance in prediction["distance"]]
        result["by_initial_distance"][label] = _group(prediction, mask)
    return result


def run_encoder(features, train_frames, test_frames, grid_h, grid_w, config, seed):
    """One encoder under the shared protocol: identical seed, epochs, batch, learning rate."""
    torch.manual_seed(seed)
    generator = torch.Generator(device=train_frames.images.device).manual_seed(seed)
    train = encode_frames(features, train_frames)
    saliency, history = train_probe(train, grid_h, grid_w, config["epochs"], config["batch_size"],
                                    config["learning_rate"], generator)
    del train
    test = encode_frames(features, test_frames)
    height, width = test_frames.images.shape[2], test_frames.images.shape[3]
    prediction = predict(saliency, test, width, height)
    return {"dim": features.dim, "history": history, "test": summarize_split(prediction)}


def protocol_of(train_frames, test_frames):
    """Protocol runs use the full frame counts; smaller runs are labelled smoke and not comparable."""
    full = train_frames >= PROTOCOL_TRAIN_FRAMES and test_frames >= PROTOCOL_TEST_FRAMES
    return "protocol" if full else "smoke"


def write_markdown(payload):
    lines = ["# Goal-localization probe", "",
             f"Protocol: `{payload['protocol']}` (train frames {payload['train_frames']}, test frames "
             f"{payload['test_frames']}, control `{payload['control_mode']}`).", "",
             "| encoder | visibility AUROC | visibility AP | visible frames | exact patch | within 1 patch | "
             "error px median | p90 |",
             "|---|---|---|---|---|---|---|---|"]
    for name, result in payload["encoders"].items():
        test = result["test"]
        lines.append(f"| {name} | {_fmt(test['visibility_auroc'])} | {_fmt(test['visibility_ap'])} | "
                     f"{test['visible_frames']} | {_fmt(test['patch_accuracy'])} | "
                     f"{_fmt(test['within_one_patch'])} | {_fmt(test['error_px_median'])} | "
                     f"{_fmt(test['error_px_p90'])} |")
    lines += ["", "Per visibility bucket (test):", ""]
    for name, result in payload["encoders"].items():
        for bucket, stats in result["test"]["by_bucket"].items():
            lines.append(f"- {name} / {bucket}: frames {stats['frames']}, AUROC {_fmt(stats['visibility_auroc'])}, "
                         f"within 1 patch {_fmt(stats['within_one_patch'])}")
    lines += ["", "Per initial distance (test):", ""]
    for name, result in payload["encoders"].items():
        for label, stats in result["test"]["by_initial_distance"].items():
            lines.append(f"- {name} / {label}: frames {stats['frames']}, AUROC {_fmt(stats['visibility_auroc'])}, "
                         f"within 1 patch {_fmt(stats['within_one_patch'])}")
    return "\n".join(lines) + "\n"


def _fmt(value):
    return "n/a" if value is None else f"{value:.3f}"
