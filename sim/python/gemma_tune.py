"""Post-train Gemma's grounding from simulator telemetry (LoRA; the base model is unchanged).

The grounding benchmark (gemma_goal.py bench) found two failures: Gemma boxes a different instance of
the named class, and when the target is out of view it still boxes a look-alike 58% of the time. The
simulator labels both for free: every frame's per-patch object labels give each visible object's box.

  data   fly exploratory routes (the perception heads' flights) and write grounding examples:
         "Fly to the <name>." -> its box (the nearest when several objects share the name, asked as
         "the nearest <name>"), and names of objects not in view -> []. Held-out districts give the
         evaluation split.
  train  LoRA on the language model's attention and MLP projections; loss on the answer tokens only.
  eval   hit rate (IoU >= 0.3) on visible targets and false-box rate on absent ones, base vs tuned.

    nix develop path:sim --command .venv/bin/python sim/python/gemma_tune.py data --out runs/gemma-tune
    nix develop path:sim --command .venv/bin/python sim/python/gemma_tune.py train --out runs/gemma-tune
    nix develop path:sim --command .venv/bin/python sim/python/gemma_tune.py eval --out runs/gemma-tune
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "sim/python"), str(ROOT / "src")]

from gemma_goal import MATCH_IOU, ask, iou, load_gemma, object_phrase, parse_box  # noqa: E402

MIN_PATCHES = 4                  # an object must cover this many 16 px patches to be asked for
GRID_W, GRID_H = 20, 15
LOSS_SCALE = 8.0                 # summed answer-token loss / LOSS_SCALE keeps the old gradient size


def prompt(instruction):
    """The exact grounding prompt of gemma_goal.detect (train and deployment must match)."""
    return (f"A drone receives the instruction: \"{instruction}\". Find the object it should fly to in this "
            "drone camera image. Output a JSON list with that one object as {\"box_2d\": [ymin, xmin, ymax, xmax] "
            "normalized to 0-1000, \"label\": name}. Output [] if it is not visible.")


def answer(box, label):
    if box is None:
        return "[]"
    x0, y0, x1, y1 = (round(v * 1000) for v in box)
    return json.dumps([{"box_2d": [y0, x0, y1, x1], "label": label}])


def frame_examples(objects, depth, layout, rng, absent_names):
    """Grounding examples for one frame from its patch labels: one visible object (if any) and one
    absent name. ``objects``/``depth`` are [P] patch labels."""
    examples = []
    counts = {}
    for index in objects[objects >= 0].tolist():
        counts[index] = counts.get(index, 0) + 1
    visible = [index for index, count in counts.items() if count >= MIN_PATCHES]
    names = {}
    for index in visible:
        names.setdefault(object_phrase(layout[index]["asset"]), []).append(index)
    if visible:
        name = rng.choice(sorted(names))
        members = names[name]
        target = min(members, key=lambda index: float(depth[objects == index].mean()))
        patches = np.flatnonzero(objects == target)
        rows, columns = patches // GRID_W, patches % GRID_W
        box = [columns.min() / GRID_W, rows.min() / GRID_H, (columns.max() + 1) / GRID_W, (rows.max() + 1) / GRID_H]
        instruction = f"Fly to the {'nearest ' if len(members) > 1 else ''}{name}."
        examples.append({"instruction": instruction, "answer": answer(box, name), "box": box, "object": int(target)})
    in_view = {object_phrase(layout[index]["asset"]) for index in counts}
    missing = [name for name in absent_names if name not in in_view]
    if missing:
        name = rng.choice(missing)
        examples.append({"instruction": f"Fly to the {name}.", "answer": "[]", "box": None, "object": None})
    return examples


def data(args):
    from sim_rl.cli import DEFAULT_LIBRARY, TEST_SEED, TEST_STYLE, TRAIN_STYLE
    from sim_rl.native import CudaEnv
    from sim_rl.perception import OU_BIAS, OU_SIGMA, OU_THETA
    from sim_rl.curriculum import fixed_lanes
    layout = json.loads((ROOT / "assets/city/layout.json").read_text())["objects"]
    names = sorted({object_phrase(o["asset"]) for o in layout if o["class"] in ("vehicle", "building", "vegetation", "prop")})
    rng = random.Random(args.seed)
    for split, style, seed, count in (("train", TRAIN_STYLE, args.seed, args.train_frames),
                                      ("test", TEST_STYLE, TEST_SEED, args.test_frames)):
        folder = args.out / split
        folder.mkdir(parents=True, exist_ok=True)
        rows = []
        with CudaEnv(DEFAULT_LIBRARY, args.num_envs, seed, 320, 240, style, "velocity",
                     start_distance=(8.0, 60.0)) as env, env.execution():
            device = env.device
            env.write_curriculum(fixed_lanes(device))
            env.reset(seed)
            generator = torch.Generator(device=device).manual_seed(seed)
            bias = torch.tensor(OU_BIAS, device=device)
            actions = bias.expand(env.n, -1).clone()
            frame = 0
            while frame < count:
                for _ in range(args.stride):
                    actions.add_(OU_THETA * (bias - actions) + OU_SIGMA * torch.randn((env.n, 4), device=device,
                                 generator=generator)).clamp_(-1, 1)
                    env.step(actions)
                images = (env.obs * 255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
                objects, depth = env.patch_object.cpu().numpy(), env.patch_depth.cpu().numpy()
                for k in range(env.n):
                    if frame >= count:
                        break
                    path = folder / f"{frame:06d}.png"
                    Image.fromarray(images[k]).save(path)
                    for example in frame_examples(objects[k], depth[k], layout, rng, names):
                        rows.append({"image": path.name, **example})
                    frame += 1
        (folder / "examples.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        print(json.dumps({"split": split, "frames": frame, "examples": len(rows),
                          "absent": sum(row["box"] is None for row in rows)}), flush=True)


def load(folder):
    return [json.loads(line) for line in (folder / "examples.jsonl").read_text().splitlines()]


def phrasing(row, rng):
    """Training instruction: "the nearest X" rows are asked half the time as plain "the X" (same
    nearest box), the way the agent words it; the first adapter, trained only on "the nearest X"
    for repeated names, declined plain requests whenever several X were in view."""
    if "the nearest " in row["instruction"] and rng.random() < 0.5:
        return row["instruction"].replace("the nearest ", "the ")
    return row["instruction"]


def train(args):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(args.model)
    model = load_gemma(args.model, args.quantize)
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
                                             target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"))
    model.print_trainable_parameters()
    rows = load(args.out / "train")
    visible, absent = [row for row in rows if row["box"]], [row for row in rows if not row["box"]]
    rng = random.Random(args.seed)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate)
    model.train()
    log = (args.out / f"{args.adapter}-log.jsonl").open("w")
    for step in range(args.steps):
        # Absent examples at --absent-fraction (absent-heavy data taught refusals).
        row = rng.choice(absent if rng.random() < args.absent_fraction else visible)
        image = Image.open(args.out / "train" / row["image"]).convert("RGB")
        messages = [{"role": "user", "content": [{"type": "image", "image": image},
                                                 {"type": "text", "text": prompt(phrasing(row, rng))}]}]
        context = processor.apply_chat_template(messages, tokenize=True, return_dict=True, return_tensors="pt",
                                                add_generation_prompt=True)
        target = processor.tokenizer(row["answer"] + processor.tokenizer.eos_token, add_special_tokens=False,
                                     return_tensors="pt")["input_ids"]
        inputs = {key: value.to("cuda") for key, value in context.items()}
        inputs["input_ids"] = torch.cat((inputs["input_ids"], target.to("cuda")), dim=1)
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
        # Summed, not averaged, over answer tokens: with the mean, the one token that decides box or
        # "[]" carried 1/25 of a box answer's loss but 1/3 of an absent one's, and greedy decoding
        # of those adapters answered "[]" to most visible targets despite a low training loss.
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(**inputs, logits_to_keep=target.shape[1] + 1).logits[:, :-1].float()
        total = F.cross_entropy(logits.flatten(0, 1), target.to("cuda").flatten(), reduction="sum")
        loss = total.detach() / target.shape[1]   # per answer token, for the log only
        (total / (LOSS_SCALE * args.accumulate)).backward()
        if (step + 1) % args.accumulate == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if step % 50 == 0:
            log.write(json.dumps({"step": step, "loss": float(loss)}) + "\n")
            log.flush()
            print(json.dumps({"step": step, "loss": round(float(loss), 4)}), flush=True)
    model.save_pretrained(args.out / args.adapter)


def evaluate(args):
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(args.model)
    model = load_gemma(args.model, args.quantize)
    rows = load(args.out / "test")[:args.eval_examples]
    results = {}
    for name in ("base", args.adapter):
        if name != "base":
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, args.out / name).eval()
        counts = {key: 0 for key in ("visible", "hits", "boxed", "plain", "plain_hits", "absent", "false_box")}
        for row in rows:
            image = Image.open(args.out / "test" / row["image"]).convert("RGB")
            if row["box"] is None:
                text, _ = ask(model, processor, image, prompt(row["instruction"]))
                counts["absent"] += 1
                counts["false_box"] += parse_box(text) is not None
                continue
            # "the nearest X" rows are also asked as plain "the X" (how the agent words it)
            for instruction in {row["instruction"], row["instruction"].replace("the nearest ", "the ")}:
                text, _ = ask(model, processor, image, prompt(instruction))
                box = parse_box(text)
                hit = box is not None and float(iou(np.array(row["box"]), np.array([box]))[0]) >= MATCH_IOU
                if instruction == row["instruction"]:
                    counts["visible"] += 1
                    counts["boxed"] += box is not None
                    counts["hits"] += hit
                if "nearest" in row["instruction"] and "nearest" not in instruction:
                    counts["plain"] += 1
                    counts["plain_hits"] += hit
        results[name] = {"visible": counts["visible"], "visible_hit": counts["hits"] / max(counts["visible"], 1),
                         "visible_boxed": counts["boxed"] / max(counts["visible"], 1),
                         "plain_phrasing": counts["plain"], "plain_phrasing_hit": counts["plain_hits"] / max(counts["plain"], 1),
                         "absent": counts["absent"], "absent_false_box": counts["false_box"] / max(counts["absent"], 1)}
        print(json.dumps({name: results[name]}), flush=True)
    (args.out / f"eval-{args.adapter}.json").write_text(json.dumps(results, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("data", "train", "eval"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="google/gemma-4-E4B-it")
    parser.add_argument("--quantize", action="store_true", help="4-bit NF4 base model (QLoRA when training)")
    parser.add_argument("--adapter", default="lora", help="adapter directory name under --out")
    parser.add_argument("--absent-fraction", type=float, default=0.25, help="share of 'not visible' training examples")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--train-frames", type=int, default=6000)
    parser.add_argument("--test-frames", type=int, default=600)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--accumulate", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--eval-examples", type=int, default=400)
    args = parser.parse_args()
    {"data": data, "train": train, "eval": evaluate}[args.stage](args)


if __name__ == "__main__":
    main()
