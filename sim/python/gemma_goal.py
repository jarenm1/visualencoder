"""Language goals by grounding: Gemma 4 boxes the instructed object in the drone's first frame.

No training. The policy's goal is the crop of the start frame at Gemma's box, the same crop an
operator tap would give; the env scores the city object that box matches. Separate processes
so Gemma and the simulator never share the GPU:

  frames  reset a held-out batch and save each env's first-person start frame   (sim shell)
  ground  Gemma 4 boxes the instructed object in each frame and the box is
          matched to the city object whose projected bounding box overlaps it most   (sim shell)

then `sim/python/replay.py --goal-boxes DIR/boxes.json` flies and records those episodes.

    nix develop path:sim --command .venv/bin/python sim/python/gemma_goal.py frames --out runs/gemma-ground
    nix develop path:sim --command .venv/bin/python sim/python/gemma_goal.py ground --out runs/gemma-ground \
        --instruction "fly to the tallest building"
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "sim/python"), str(ROOT / "src")]

# A box counts as a city object when their image boxes overlap at least this much.
MATCH_IOU = 0.3
INSTRUCTION_LIMIT = 300


def rotate(q, v):
    """Rotate vectors v [..., 3] by unit quaternions q [..., 4] (xyzw)."""
    u, w = q[..., :3], q[..., 3:]
    t = 2 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def start_camera(diagnostics, metadata):
    """Eye, forward, up of each env's body camera from diagnostics (position 0-2, quaternion 6-9)."""
    camera = metadata["camera"]
    tilt = camera["uptilt_rad"]
    q = diagnostics[:, 6:10]
    forward = rotate(q, np.array([0.0, np.sin(tilt), -np.cos(tilt)]))
    up = rotate(q, np.array([0.0, np.cos(tilt), np.sin(tilt)]))
    eye = diagnostics[:, 0:3] + rotate(q, np.array(camera["offset_body_m"]))
    return eye, forward, up


def project_boxes(eye, forward, up, centers, halves, camera):
    """Normalized image boxes [k, 4] (x0, y0, x1, y1, top-left origin) of world AABBs seen from one
    camera; NaN rows for boxes with a corner behind the near plane or wholly outside the frame."""
    right = np.cross(forward, up)
    half_tan = np.tan(camera["vertical_fov_rad"] / 2)
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=float)
    corners = centers[:, None] + signs[None] * halves[:, None]
    d = corners - eye
    depth = d @ forward
    x = (d @ right) / (depth * half_tan * camera["aspect"])
    y = (d @ up) / (depth * half_tan)
    u, v = 0.5 * (x + 1), 0.5 * (1 - y)
    boxes = np.stack((u.min(1), v.min(1), u.max(1), v.max(1)), -1)
    valid = (depth > camera["near_m"]).all(1) & (boxes[:, 2] > 0) & (boxes[:, 0] < 1) & (boxes[:, 3] > 0) & (boxes[:, 1] < 1)
    boxes = boxes.clip(0, 1)
    boxes[~valid] = np.nan
    return boxes


def iou(box, boxes):
    x0, y0 = np.maximum(box[0], boxes[:, 0]), np.maximum(box[1], boxes[:, 1])
    x1, y1 = np.minimum(box[2], boxes[:, 2]), np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area = lambda b: (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return np.nan_to_num(inter / (area(box) + area(boxes) - inter + 1e-9), nan=0.0)


def frames(args):
    """Save start frames and cameras of a fresh held-out batch (same seed/envs replay later)."""
    from sim_rl.cli import DEFAULT_LIBRARY, TEST_SEED, TEST_STYLE
    from sim_rl.curriculum import UNIFORM_WEIGHTS
    from sim_rl.native import CudaEnv
    args.out.mkdir(parents=True, exist_ok=True)
    seed = TEST_SEED if args.seed is None else args.seed
    with CudaEnv(DEFAULT_LIBRARY, args.num_envs, seed, 320, 240, TEST_STYLE, "velocity") as env, env.execution():
        env.write_curriculum(torch.tensor(UNIFORM_WEIGHTS + (0.0, args.appearance), device=env.device))
        env.reset(seed)
        images = (env.obs.clamp(0, 1) * 255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        diagnostics = env.diagnostics.cpu().numpy().astype(float)
        metadata, layout = env.metadata, json.loads((env.city_path / "layout.json").read_text())
    eye, forward, up = start_camera(diagnostics, metadata)
    for index, image in enumerate(images):
        Image.fromarray(image).save(args.out / f"frame-{index:03d}.png")
    targets = [i for i, o in enumerate(layout["objects"]) if o["class"] in metadata["target_classes"]]
    record = {"seed": seed, "num_envs": args.num_envs, "split": "test", "appearance": args.appearance,
              "camera": metadata["camera"], "targets": targets,
              "envs": [{"env": i, "eye": eye[i].tolist(), "forward": forward[i].tolist(), "up": up[i].tolist(),
                        "catalog": int(diagnostics[i, 16]), "object": int(diagnostics[i, 27])}
                       for i in range(args.num_envs)]}
    (args.out / "frames.json").write_text(json.dumps(record) + "\n")
    print(f"wrote {args.num_envs} start frames (seed {seed}) to {args.out}")


def load_gemma(model_id):
    """bf16 Gemma 4 on the GPU except the per-layer embedding table (5.6 GB for E4B).

    That table is only a per-token lookup, so it stays in host memory and its result is
    copied to the GPU; accelerate's offload instead re-uploads the whole table per call.
    """
    from transformers import AutoModelForMultimodalLM
    model = AutoModelForMultimodalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cpu")
    language = model.model.language_model
    table = language._modules.pop("embed_tokens_per_layer", None)
    model.to("cuda")
    if table is not None:
        language._modules["embed_tokens_per_layer"] = table
        lookup = table.forward
        table.forward = lambda ids, *rest, **kw: lookup(ids.cpu(), *rest, **kw).to("cuda", non_blocking=True)
    return model.eval()


def detect(model, processor, image, instruction):
    """Gemma's box for the instructed object -> (normalized [x0, y0, x1, y1] or None, raw answer, seconds)."""
    prompt = (f"A drone receives the instruction: \"{instruction}\". Find the object it should fly to in this "
              "drone camera image. Output a JSON list with that one object as {\"box_2d\": [ymin, xmin, ymax, xmax] "
              "normalized to 0-1000, \"label\": name}. Output [] if it is not visible.")
    content = [{"type": "image", "image": image}, {"type": "text", "text": prompt}]
    inputs = processor.apply_chat_template([{"role": "user", "content": content}], tokenize=True, return_dict=True,
                                           return_tensors="pt", add_generation_prompt=True).to("cuda")
    start = time.monotonic()
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=96, do_sample=False)
    latency = time.monotonic() - start
    text = processor.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
    match = re.search(r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]", text)
    if not match:
        return None, text, latency
    y0, x0, y1, x1 = (min(max(int(v) / 1000, 0.0), 1.0) for v in match.groups())
    box = [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]
    return (box if box[2] - box[0] > 0.01 and box[3] - box[1] > 0.01 else None), text, latency


def ground(args):
    """Box the instructed object in every saved start frame and match each box to a city object."""
    from transformers import AutoProcessor
    instruction = args.instruction.strip() if args.instruction else ""
    if not 0 < len(instruction) <= INSTRUCTION_LIMIT:
        raise SystemExit(f"instruction must be 1-{INSTRUCTION_LIMIT} characters")
    record = json.loads((args.out / "frames.json").read_text())
    layout = json.loads((ROOT / "assets/city/layout.json").read_text())["objects"]
    targets = np.array(record["targets"])
    lo = np.array([layout[i]["min"] for i in targets], dtype=float)
    hi = np.array([layout[i]["max"] for i in targets], dtype=float)
    centers, halves = (lo + hi) / 2, (hi - lo) / 2
    processor = AutoProcessor.from_pretrained(args.model)
    model = load_gemma(args.model)
    boxes = []
    for env in record["envs"]:
        image = Image.open(args.out / f"frame-{env['env']:03d}.png").convert("RGB")
        box, answer, latency = detect(model, processor, image, instruction)
        entry = {"env": env["env"], "box": box, "answer": answer, "latency_s": round(latency, 3), "object": None}
        if box is not None:
            projected = project_boxes(np.array(env["eye"]), np.array(env["forward"]), np.array(env["up"]),
                                      centers, halves, record["camera"])
            overlap = iou(np.array(box), projected)
            best = int(overlap.argmax())
            if overlap[best] >= MATCH_IOU:
                index = int(targets[best])
                entry.update(object=index, object_class=layout[index]["class"], asset=layout[index]["asset"],
                             iou=round(float(overlap[best]), 3), center=centers[best].tolist(), half=halves[best].tolist())
        boxes.append(entry)
        print(json.dumps({k: entry[k] for k in ("env", "box", "object", "latency_s")}), flush=True)
    found = sum(entry["box"] is not None for entry in boxes)
    matched = sum(entry["object"] is not None for entry in boxes)
    summary = {"instruction": instruction, "model": args.model, "seed": record["seed"], "num_envs": record["num_envs"],
               "split": record["split"], "appearance": record["appearance"], "found": found, "matched": matched,
               "match_iou": MATCH_IOU, "boxes": boxes}
    (args.out / "boxes.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({"instruction": instruction, "found": found, "matched": matched}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("frames", "ground"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="google/gemma-4-E4B-it")
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--seed", type=int, help="held-out reset seed (default: the test seed)")
    parser.add_argument("--appearance", type=float, default=1.0)
    parser.add_argument("--instruction", help="free-text goal for the ground stage")
    args = parser.parse_args()
    {"frames": frames, "ground": ground}[args.stage](args)


if __name__ == "__main__":
    main()
