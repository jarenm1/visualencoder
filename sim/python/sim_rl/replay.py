"""Capture policy-consumed frames, poses, and encoder maps on GPU, then render replay clips offline."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .cli import (DEFAULT_LIBRARY, ROOT, TEST_SEED, TEST_STYLE, TRAIN_STYLE, VALIDATION_SEED,
                  VALIDATION_STYLE, build_policy, load_checkpoint, sha256, validate_checkpoint)
from .curriculum import fixed_lanes
from .features import RefreshedGoals, checkpoint_features, crop_goals, square_goal_box
from .memory import CHANNELS, EGO, EGO_RADIUS, SpatialMemory, observe as observe_memory
from .model import GOAL_COMMAND, deterministic_action, saliency_targets
from .native import DEFAULT_CITY, PRIVILEGED_WIDTH, CudaEnv

# Diagnostics indices (ABI v6, width 28) recorded per frame, pre-action: pose, target centre and
# half extents, catalog index, layout class and object, clock, landed.
STATE_INDEX = (0, 1, 2, 6, 7, 8, 9, 3, 4, 5, 24, 25, 26, 16, 14, 27, 17, 18)
# Privileged lanes appended per frame as ground-truth labels for saliency diagnosis
# (replay-only; the actor never receives them): anchor visible, anchor NDC, anchor in frustum,
# and the target box's screen bounds (NDC, y up).
LABEL_INDEX = (14, 18, 19, 20, 25, 26, 27, 28)
STATE_FIELDS = ["world_x", "world_y", "world_z", "quat_x", "quat_y", "quat_z", "quat_w",
                "target_x", "target_y", "target_z", "target_half_x", "target_half_y", "target_half_z",
                "catalog", "target_class", "object", "clock", "landed",
                "target_visible", "target_ndc_x", "target_ndc_y", "anchor_in_frustum",
                "target_box_x0", "target_box_y0", "target_box_x1", "target_box_y1"]
COLUMN = {name: index for index, name in enumerate(STATE_FIELDS)}
TARGET_FIELDS = ["catalog", "target_class", "object", "target_half_x", "target_half_y", "target_half_z"]
# Transition columns: action_names (4), then the outcome fields below.
OUTCOME_FIELDS = ["reward", "done", "success", "collision", "out_of_bounds"]
START_LANDED_INDEX = 19
GPS_AVAILABLE_INDEX = 23
PCA_SAMPLE = 65536
CHUNK = 64                       # frames buffered on the GPU before each copy to host memory
TORQUE_NOTE = "Goal-facing visualization proxy; the task has no rotor/aerodynamic attitude state."


@torch.no_grad()
def pca_basis(tokens):
    """3-component PCA basis (mean [D], axes [D, 3]) of one frame batch's patch tokens [N, P, D]."""
    rows = tokens.reshape(-1, tokens.shape[-1]).float()
    if rows.shape[0] > PCA_SAMPLE:
        rows = rows[torch.randperm(rows.shape[0], device=rows.device,
                                   generator=torch.Generator(device=rows.device).manual_seed(0))[:PCA_SAMPLE]]
    mean = rows.mean(0)
    _, _, principal = torch.linalg.svd(rows - mean, full_matrices=False)
    return mean, principal[:3].T


@torch.no_grad()
def encoder_maps(projected, valid):
    """PCA projections [T, N, P, 3] -> uint8 maps, scaled by the 1st/99th percentile of valid frames."""
    generator = torch.Generator(device=projected.device).manual_seed(0)
    sample = projected[valid].reshape(-1, 3).float()
    if sample.shape[0] > PCA_SAMPLE:
        sample = sample[torch.randperm(sample.shape[0], device=sample.device, generator=generator)[:PCA_SAMPLE]]
    low, high = torch.quantile(sample, 0.01, dim=0), torch.quantile(sample, 0.99, dim=0)
    scaled = ((projected.float() - low) / (high - low).clamp_min(1e-6)).clamp(0, 1)
    return (scaled * 255).round().to(torch.uint8)


@torch.no_grad()
def goal_crops(env):
    """uint8 [n, H, W, 3]: each env's current goal crop, as the policy's goal encoder sees it.

    One goal render per env (slot 0 shows the requested catalog entry); render_goals restores the
    current observation, so this is safe right after reset. Startup-time host loop.
    """
    images = torch.empty((env.n, env.height, env.width, 3), dtype=torch.uint8, device=env.device)
    for slot, entry in enumerate(env.catalog_index.cpu().tolist()):
        env.render_goals(entry)
        crop = crop_goals(env.goal_views[:1], env.goal_crops[:1])[0]
        images[slot].copy_((crop.clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0))
    return images


@torch.no_grad()
def ground_goals(env, features, grounded):
    """Override goals with crops of the start frame at language-model boxes (see gemma_goal.py).

    Returns (goal embeddings [n, 2D], has_box [n] bool, goal crops uint8 [n,H,W,3] or None,
    retarget rows [n, 8]). Only boxes matched to a city object count: the env is retargeted to that
    object, so GPS, the goal crop and success all name the same object. Unmatched boxes are dropped:
    flying them would give the policy GPS to the catalog target and an image of something else.
    Call right after reset.
    """
    n = env.n
    boxes = torch.zeros((n, 4), device=env.device)
    has_box = torch.zeros(n, dtype=torch.bool, device=env.device)
    targets = torch.full((n, 8), -1.0, device=env.device)
    for entry in grounded["boxes"]:
        if entry["object"] is None:
            continue
        boxes[entry["env"]] = torch.tensor(entry["box"])
        has_box[entry["env"]] = True
        targets[entry["env"]] = torch.tensor([*entry["center"], entry["object"], *entry["half"],
                                              env.layout_classes.index(entry["object_class"])])
    crops = crop_goals(env.obs.clone(), square_goal_box(boxes))
    goals = features.goal_embedding(crops) if features.visual else None
    images = (crops.clamp(0, 1) * 255).round().to(torch.uint8).permute(0, 2, 3, 1)
    env.retarget(targets)
    return goals, has_box, images, targets


@torch.no_grad()
def capture(args, saved, features, seed, style, no_gps_fraction, appearance, grounded=None):
    """Roll out the deterministic policy (tanh of the mean) until every environment's first episode
    has ended (at most the horizon).

    Buckets are uniform; episodes run without GPS with probability ``no_gps_fraction`` and
    goal crops are rendered at ``appearance`` strength. With ``grounded`` (gemma_goal.py
    boxes.json), envs with a box use the start-frame crop at that box as their goal. Frames move
    to host memory every CHUNK steps; encoder maps are projected onto a PCA basis fit on the first
    frame, and spatial-memory policies also record their ego-centric memory crop.
    """
    metadata = saved["metadata"]
    n = args.num_envs
    start = metadata.get("start_distance")   # the checkpoint's training start range
    with CudaEnv(args.library, n, seed, metadata["width"], metadata["height"], style,
                 metadata["control_mode"], args.city, start_distance=tuple(start) if start else None) as env:
        with env.execution():
            validate_checkpoint(saved, env, features)
            policy = build_policy(env, features, saved)
            policy.eval()
            device, horizon = env.device, env.horizon
            h, w = env.height, env.width
            patch_h, patch_w = h // 16, w // 16
            lanes = fixed_lanes(device, no_gps=no_gps_fraction, appearance=appearance)
            env.write_curriculum(lanes)
            goals = features.catalog_goals(env, appearance) if features.visual else None
            env.reset(seed)
            goal_images = goal_crops(env)
            if grounded is not None:
                # retargets stays referenced until the export sync: native reads it on the stream.
                grounded_goals, has_box, grounded_images, retargets = ground_goals(env, features, grounded)
                goal_images = torch.where(has_box[:, None, None, None], grounded_images, goal_images)
            start_landed = env.diagnostics[:, START_LANDED_INDEX].clone()
            refresh_steps = metadata.get("goal_refresh_steps", 0) if grounded is None else 0
            goal_source = RefreshedGoals(env, features, goals, refresh_steps) if features.visual else None
            gps_available = env.diagnostics[:, GPS_AVAILABLE_INDEX].clone()
            index = torch.tensor(STATE_INDEX, device=device)
            outcome_width = len(OUTCOME_FIELDS)
            action_width = env.action_dim
            images = torch.empty((horizon, n, h, w, 3), dtype=torch.uint8)  # host memory
            chunk = torch.empty((CHUNK, n, h, w, 3), dtype=torch.uint8, device=device)
            states = torch.empty((horizon, n, len(STATE_INDEX) + len(LABEL_INDEX)), device=device)
            label_index = torch.tensor(LABEL_INDEX, device=device)
            transitions = torch.empty((horizon, n, action_width + outcome_width), device=device)
            ended = torch.empty((horizon, n), device=device)
            projected = (torch.empty((horizon, n, patch_h * patch_w, 3), dtype=torch.float16, device=device)
                         if features.visual else None)
            # Policy target saliency: softmax over P patches + "not in view".
            saliency = (torch.empty((horizon, n, patch_h * patch_w + 1), dtype=torch.float16, device=device)
                        if features.visual else None)
            spatial = (SpatialMemory(n, patch_h, patch_w, env.metadata["camera"], device) if policy.spatial else None)
            memory_maps = (torch.empty((horizon, n, CHANNELS, EGO, EGO), dtype=torch.uint8, device=device)
                           if spatial is not None else None)
            starts = torch.ones(n, device=device)
            finished = torch.zeros(n, dtype=torch.bool, device=device)
            hidden = torch.zeros((n, policy.hidden_size), device=device)
            basis = None
            goal_command = torch.full((n,), GOAL_COMMAND, dtype=torch.long, device=device)
            for t in range(horizon):
                chunk[t % CHUNK].copy_((env.obs.clamp(0, 1) * 255).round().to(torch.uint8).permute(0, 2, 3, 1))
                states[t, :, :len(STATE_INDEX)].copy_(env.diagnostics.index_select(1, index))
                states[t, :, len(STATE_INDEX):].copy_(env.privileged.index_select(1, label_index))
                if features.visual:
                    cls, patch_tokens = features.observe(env)
                    if basis is None:
                        basis = pca_basis(patch_tokens)
                    projected[t].copy_((patch_tokens.float() - basis[0]) @ basis[1])
                    goal = goal_source.step(starts)
                    if grounded is not None:
                        goal = torch.where(has_box[:, None], grounded_goals, goal)
                else:
                    cls, patch_tokens, goal = None, None, None
                memory = (observe_memory(policy, spatial, cls, patch_tokens, goal, env.proprio, starts)
                          if spatial is not None else None)
                if memory is not None:
                    memory_maps[t].copy_((memory[0].clamp(0, 1) * 255).round().to(torch.uint8))
                inputs, logits = policy.actor_inputs_and_features(cls, patch_tokens, goal, env.proprio,
                                                                  env.privileged, memory, goal_command)
                if saliency is not None:
                    saliency[t].copy_(logits.softmax(-1))
                mean, hidden = policy.step(inputs, hidden, starts)
                applied = deterministic_action(mean).contiguous()
                env.step(applied)
                transitions[t, :, :action_width].copy_(applied)
                transitions[t, :, 4].copy_(env.reward)
                transitions[t, :, 5].copy_(env.done)
                transitions[t, :, 6].copy_(env.success)
                transitions[t, :, 7].copy_(env.collision)
                transitions[t, :, 8].copy_(env.out_of_bounds)
                ended[t].copy_(env.ended_bucket)
                starts.copy_(env.done)
                finished |= env.done.bool()
                if (t + 1) % CHUNK == 0 or t + 1 == horizon:
                    first = t - t % CHUNK
                    images[first:t + 1].copy_(chunk[:t + 1 - first])  # Export boundary: one copy per chunk.
                    if bool(finished.all()):
                        break
            length = t + 1
            states, transitions, ended = states[:length], transitions[:length], ended[:length]
            done = transitions[:, :, 5]
            # Frame t is pre-terminal when no episode ended strictly before it.
            valid = (torch.cumsum(done, 0) - done) == 0
            maps = encoder_maps(projected[:length], valid).cpu() if features.visual else None
            env.stream.synchronize()  # Explicit export boundary: one readback of the recording.
            return {"images": images[:length].numpy(), "states": states.cpu().numpy(),
                    "transitions": transitions.cpu().numpy(), "ended": ended.cpu().numpy(),
                    "start_landed": start_landed.cpu().numpy(), "gps_available": gps_available.cpu().numpy(),
                    "goal_images": goal_images.cpu().numpy(),
                    "maps": None if maps is None else maps.numpy(),
                    "saliency": None if saliency is None else saliency[:length].float().cpu().numpy(),
                    "memory": None if memory_maps is None else memory_maps[:length].cpu().numpy(),
                    "grid": (patch_h, patch_w),
                    "native": {"control_hz": env.control_hz, "control_mode": env.control_mode,
                               "goal_refresh_steps": refresh_steps,
                               "goal_box_source": "simulator" if refresh_steps else None,
                               "action_names": list(env.action_names), "city_path": str(env.city_path),
                               "target_classes": list(env.target_classes), "layout_classes": list(env.layout_classes),
                               "city_fingerprint": env.metadata["city_fingerprint"]}}


def diagnose(probs, labels, grid):
    """Per-episode detection diagnosis from policy saliency against ground truth.

    probs float [L, P+1] (softmax over patches + "not in view"); labels [L, 8] = (visible, anchor
    ndc_x, ndc_y, in frustum, box x0, y0, x1, y1). A frame counts as detected when the target is
    visible, P(in view) > 0.5 and the most likely patch lies on the target, the training label.
    """
    _, _, on_target = saliency_targets(torch.as_tensor(labels_privileged(labels)), *grid)
    on_target = on_target.numpy()
    visible = labels[:, 0] > 0.5
    best = probs[:, :-1].argmax(-1)
    detected = visible & on_target[np.arange(len(best)), best] & (probs[:, -1] < 0.5)
    first = lambda mask: int(np.flatnonzero(mask)[0]) if mask.any() else None
    return {"visible_frames": int(visible.sum()), "detected_frames": int(detected.sum()),
            "first_visible_frame": first(visible), "first_detected_frame": first(detected),
            "detection_rate": float(detected.sum() / visible.sum()) if visible.any() else None}


def labels_privileged(labels):
    """Scatter recorded label columns back into privileged-lane positions for saliency_targets."""
    privileged = np.zeros((len(labels), PRIVILEGED_WIDTH), dtype=np.float32)
    privileged[:, list(LABEL_INDEX)] = labels
    return privileged


def failure_kind(outcome, diagnosis):
    """Coarse cause for a non-success: target never seen, seen but not detected, or detected."""
    if outcome == "success":
        return None
    if diagnosis["visible_frames"] == 0:
        return "never_visible"
    if diagnosis["detection_rate"] < 0.5:
        return "visible_not_detected"
    return "detected_not_reached"


def record(args):
    if args.num_envs < 1:
        raise ValueError("--num-envs must be positive")
    if not 0.0 <= args.no_gps_fraction <= 1.0:
        raise ValueError("--no-gps-fraction must be in [0, 1]")
    if not 0.0 <= args.appearance <= 1.0:
        raise ValueError("--appearance must be in [0, 1]")
    renderer = args.renderer.resolve(strict=True)
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required; use the sim Nix development shell")
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite replay: {args.out}")
    saved = load_checkpoint(args.checkpoint)
    metadata = saved["metadata"]
    features = checkpoint_features(metadata, torch.device("cuda:0"))
    styles = {"train": TRAIN_STYLE, "validation": VALIDATION_STYLE, "test": TEST_STYLE}
    seeds = {"train": 0, "validation": VALIDATION_SEED, "test": TEST_SEED}
    seed = seeds[args.split] if args.seed is None else args.seed
    if not 0 <= seed < 2**64:
        raise ValueError("seed must fit unsigned 64 bits")
    grounded = None
    if args.goal_boxes is not None:
        grounded = json.loads(args.goal_boxes.read_text())
        if (args.split, args.num_envs) != (grounded["split"], grounded["num_envs"]):
            raise ValueError(f"--goal-boxes was grounded on {grounded['num_envs']} {grounded['split']} envs; "
                             "use the same --split and --num-envs")
        seed, args.appearance = grounded["seed"], grounded["appearance"]
    recording = capture(args, saved, features, seed, styles[args.split], args.no_gps_fraction, args.appearance, grounded)
    torch.cuda.empty_cache()  # Release this exporter's unused memory before the renderer process.
    if args.target_class is not None and args.target_class not in recording["native"]["target_classes"]:
        raise ValueError(f"--target-class must be one of {', '.join(recording['native']['target_classes'])}")
    states, transitions = recording["states"], recording["transitions"]
    if not np.isfinite(states).all() or not np.isfinite(transitions).all():
        raise RuntimeError("native recording contains non-finite state or transitions")
    action_names = recording["native"]["action_names"]
    transition_fields = action_names + OUTCOME_FIELDS
    args.out.mkdir(parents=True)
    environments = []
    for index in range(args.num_envs):
        terminal = np.flatnonzero(transitions[:, index, 5])
        if not len(terminal):
            raise RuntimeError(f"environment {index} did not terminate within its finite horizon")
        length = int(terminal[0]) + 1
        episode = states[:length, index]
        identity = [COLUMN[name] for name in TARGET_FIELDS]
        clock = episode[:, COLUMN["clock"]]
        if not np.all(episode[:, identity] == episode[0, identity]) or not np.array_equal(clock, np.arange(length)):
            raise RuntimeError(f"environment {index} recording crosses an autoreset boundary")
        target_class = recording["native"]["layout_classes"][int(episode[0, COLUMN["target_class"]])]
        if args.target_class is not None and target_class != args.target_class:
            continue
        grounding = None
        if grounded is not None:
            grounding = next(entry for entry in grounded["boxes"] if entry["env"] == index)
            if grounding["object"] is None:
                continue
        final = transitions[length - 1, index]
        outcome = ("success" if final[6] else "collision" if final[7]
                   else "out_of_bounds" if final[8] else "timeout")
        basename = f"env-{index:03d}"
        with (args.out / f"{basename}.rgb").open("wb") as stream:
            for frame in recording["images"][:length, index]:
                stream.write(memoryview(np.ascontiguousarray(frame)))
        Image.fromarray(recording["goal_images"][index]).save(args.out / f"{basename}-goal.png")
        entry = {
            "id": len(environments), "frame_count": length, "duration_seconds": length / recording["native"]["control_hz"],
            "outcome": outcome, "bucket": int(recording["ended"][length - 1, index]),
            "start_landed": bool(recording["start_landed"][index] > 0.5),
            "gps_available": bool(recording["gps_available"][index] > 0.5),
            "target_class": target_class, "object": int(episode[0, COLUMN["object"]]),
            "goal": f"{basename}-goal.png",
            # Language-grounded goal: Gemma's box and the city object it matched (the scored target).
            "grounding": None if grounding is None else {
                key: grounding.get(key) for key in ("box", "object", "object_class", "asset", "iou", "answer")},
            "video": f"{basename}.mp4", "states": episode.tolist(),
            "transitions": transitions[:length, index].tolist(),
        }
        if recording["maps"] is not None:
            map_path = args.out / f"{basename}.encoder"
            np.ascontiguousarray(recording["maps"][:length, index]).tofile(map_path)
            entry["encoder_map"] = map_path.name
        if recording["saliency"] is not None:
            probs = recording["saliency"][:length, index]
            labels = episode[:, len(STATE_INDEX):]
            entry["diagnosis"] = diagnose(probs, labels, recording["grid"])
            entry["diagnosis"]["failure_kind"] = failure_kind(outcome, entry["diagnosis"])
            # uint8 per frame: patch probabilities conditional on "in view", scaled by the
            # frame maximum, followed by P(in view).
            patches = probs[:, :-1] / np.maximum(probs[:, :-1].max(-1, keepdims=True), 1e-12)
            packed = np.concatenate((patches, 1 - probs[:, -1:]), axis=-1)
            saliency_path = args.out / f"{basename}.saliency"
            np.round(np.clip(packed, 0, 1) * 255).astype(np.uint8).tofile(saliency_path)
            entry["saliency"] = saliency_path.name
        if recording["memory"] is not None:
            memory_path = args.out / f"{basename}.memory"
            np.ascontiguousarray(recording["memory"][:length, index]).tofile(memory_path)
            entry["memory_map"] = memory_path.name
        environments.append(entry)
    if not environments:
        raise RuntimeError("no recorded episode matches the --target-class filter or has a grounded goal box")
    width, height = metadata["width"], metadata["height"]
    manifest = {
        "instruction": None if grounded is None else {
            key: grounded[key] for key in ("instruction", "model", "found", "matched", "match_iou")},
        "format": "sim-city-replay-v3", "title": "City drone goal replay", "style": styles[args.split],
        "split": args.split, "control_hz": recording["native"]["control_hz"],
        "control_mode": recording["native"]["control_mode"], "action_names": action_names,
        "no_gps_fraction": args.no_gps_fraction,
        "appearance": args.appearance,
        "goal_refresh_steps": recording["native"]["goal_refresh_steps"],
        "goal_box_source": recording["native"]["goal_box_source"],
        "frame_width": width, "frame_height": height, "video_width": width * 2, "video_height": height,
        "views": {"first_person": [0, 0, width, height], "third_person": [width, 0, width, height]},
        "encoder_map": None if recording["maps"] is None else {
            "width": width // 16, "height": height // 16, "channels": 3, "dtype": "uint8",
            "method": "Unpooled patch tokens of the frozen encoder, same pass the policy consumed; "
                      "3-component PCA basis fit on the first frame, 1st/99th percentile of pre-terminal frames."},
        "memory_map": None if recording["memory"] is None else {
            "width": EGO, "height": EGO, "channels": CHANNELS, "dtype": "uint8", "radius_m": EGO_RADIUS,
            "layout": "per frame: channel-major [goal evidence, looked-at] x rows x columns, 0-255",
            "method": "the policy's spatial-memory input: top-down, heading-aligned (forward up), drone at the centre"},
        "saliency": None if recording["saliency"] is None else {
            "width": width // 16, "height": height // 16, "dtype": "uint8",
            "layout": "per frame: width*height patch probabilities conditional on in-view (row-major, scaled "
                      "by the frame maximum), then P(in view)",
            "method": "softmax of the policy's TargetSaliency logits on the frame it acted on"},
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": sha256(args.checkpoint),
        "native_sha256": metadata["native_library_sha256"], "encoder": metadata["encoder"],
        "actor_inputs": metadata["actor_inputs"], "selected_steps": metadata["steps"],
        "seed": seed, "city_fingerprint": recording["native"]["city_fingerprint"],
        "city_path": recording["native"]["city_path"],
        "state_fields": STATE_FIELDS, "transition_fields": transition_fields,
        "state_semantics": "Pre-action poses and diagnostics. Terminal flag belongs to the action applied in the same "
                           "record; no autoreset pose is appended.",
        "third_person_attitude": TORQUE_NOTE,
        "environments": environments,
    }
    staging = args.out / "recording.json"
    staging.write_text(json.dumps(manifest, allow_nan=False) + "\n")
    subprocess.run([str(renderer), "--recording", str(staging.resolve())], check=True)
    for env_entry in environments:
        video = args.out / env_entry["video"]
        if not video.is_file() or video.stat().st_size == 0:
            raise RuntimeError(f"renderer did not produce {video}")
    staging.replace(args.out / "manifest.json")
    for env_entry in environments:
        (args.out / env_entry["video"]).with_suffix(".rgb").unlink()
    kinds = [e["diagnosis"]["failure_kind"] for e in environments if e.get("diagnosis")]
    print(json.dumps({"replay": str(args.out.resolve()), "environments": len(environments),
                      "outcomes": {kind: sum(e["outcome"] == kind for e in environments)
                                   for kind in ("success", "collision", "out_of_bounds", "timeout")},
                      "failure_kinds": {kind: kinds.count(kind) for kind in
                                        ("never_visible", "visible_not_detected", "detected_not_reached")}},
                     indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--renderer", type=Path, default=ROOT / ".sim-target/training/sim-replay-render")
    parser.add_argument("--city", type=Path, default=DEFAULT_CITY)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/city-drone-replay")
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--no-gps-fraction", type=float, default=0.0,
                        help="probability that the replayed episodes run without GPS")
    parser.add_argument("--appearance", type=float, default=1.0,
                        help="randomized appearance strength of the goal crops (0 = plain render)")
    parser.add_argument("--target-class",
                        help="keep only episodes whose target is a city object of this layout class (e.g. building)")
    parser.add_argument("--goal-boxes", type=Path,
                        help="gemma_goal.py boxes.json: goals are start-frame crops at these boxes (its seed and appearance)")
    record(parser.parse_args())
