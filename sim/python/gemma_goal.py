"""Gemma 4 picks the drone's goal from an instruction; the frozen policy flies to it.

No training. Separate processes so Gemma and the simulator never share the GPU:
  dump     render the held-out split's goal reference images             (sim shell)
  pick     Gemma 4 chooses one reference image per scripted instruction  (any shell)
  resolve  Gemma 4 chooses one reference image for --instruction TEXT    (any shell; used by the replay site)
  fly      roll out the trained policy with Gemma's chosen goals          (sim shell)

    nix develop path:sim --command .venv/bin/python sim/python/gemma_goal.py dump --out runs/gemma-goal
    .venv/bin/python sim/python/gemma_goal.py pick --out runs/gemma-goal
    nix develop path:sim --command .venv/bin/python sim/python/gemma_goal.py fly --out runs/gemma-goal \
        --checkpoint runs/drone4-mean-seed0/best.pt
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "sim/python"), str(ROOT / "src")]

# sim-scene PALETTE order; odd variants carry a checker pattern.
COLORS = ["red", "blue", "yellow", "green", "white", "purple", "orange", "teal"]
SHAPE_ANALOGY = {"sphere": "a ball", "cube": "a box", "cylinder": "a can", "cone": "a traffic cone",
                 "octahedron": "a diamond", "torus": "a donut"}


def describe(name):
    shape, variant = name.rsplit("-", 1)
    variant = int(variant)
    pattern = "checkered " if variant % 2 == 1 else ""
    color = COLORS[variant]
    return {
        "full": f"Fly to the {pattern}{color} {shape}.",
        "color": f"Go to the {color} one.",
        "riddle": f"Find the {pattern}{color} object that is shaped like {SHAPE_ANALOGY[shape]}.",
    }


def dump(args):
    from sim_rl.cli import TEST_SEED, TEST_STYLE
    from sim_rl.native import CudaEnv
    args.out.mkdir(parents=True, exist_ok=True)
    with CudaEnv(ROOT / ".sim-target/training/libsim_env.so", 1, TEST_SEED, 320, 240, TEST_STYLE, "velocity") as env:
        with env.execution():
            images = (env.goal_images.clamp(0, 1) * 255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        names = [d["name"] for d in env.metadata["designs"]]
    for index, image in enumerate(images):
        Image.fromarray(image).save(args.out / f"goal-{index}.png")
    (args.out / "designs.json").write_text(json.dumps(
        [{"index": i, "name": n, "instructions": describe(n)} for i, n in enumerate(names)], indent=2) + "\n")
    print(f"wrote {len(names)} goal images and instructions to {args.out}")


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


def choose(model, processor, images, instruction):
    """Gemma picks one reference image for the instruction -> (index or None, raw answer, seconds)."""
    content = []
    for option, image in enumerate(images, start=1):
        content += [{"type": "text", "text": f"Option {option}:"}, {"type": "image", "image": image}]
    content.append({"type": "text", "text":
                    f"These are reference images of possible targets for a drone. Instruction: \"{instruction}\" "
                    "Which option is the target? Answer with only the option number."})
    inputs = processor.apply_chat_template([{"role": "user", "content": content}], tokenize=True, return_dict=True,
                                           return_tensors="pt", add_generation_prompt=True).to("cuda")
    start = time.monotonic()
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=8, do_sample=False)
    latency = time.monotonic() - start
    text = processor.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
    match = re.search(r"\d+", text)
    chosen = int(match.group()) - 1 if match and 1 <= int(match.group()) <= len(images) else None
    return chosen, text, latency


def library(out):
    designs = json.loads((out / "designs.json").read_text())
    return designs, [Image.open(out / f"goal-{d['index']}.png").convert("RGB") for d in designs]


def pick(args):
    from transformers import AutoProcessor
    designs, images = library(args.out)
    processor = AutoProcessor.from_pretrained(args.model)
    model = load_gemma(args.model)
    picks, records = {}, []
    for design in designs:
        for style, instruction in design["instructions"].items():
            chosen, text, latency = choose(model, processor, images, instruction)
            picks.setdefault(style, {})[design["index"]] = chosen
            records.append({"design": design["index"], "name": design["name"], "style": style,
                            "instruction": instruction, "answer": text, "chosen": chosen,
                            "correct": chosen == design["index"], "latency_s": round(latency, 3)})
            print(json.dumps(records[-1]), flush=True)
    accuracy = {style: sum(r["correct"] for r in records if r["style"] == style) / len(designs) for style in picks}
    (args.out / "picks.json").write_text(json.dumps({"model": args.model, "accuracy": accuracy, "picks": picks,
                                                     "records": records}, indent=2) + "\n")
    print("accuracy", accuracy)


def resolve(args):
    """One free-text instruction -> one JSON line {design, name, answer, latency_s} on stdout."""
    from transformers import AutoProcessor
    if not args.instruction or len(args.instruction) > 300:
        raise SystemExit("instruction must be 1-300 characters")
    designs, images = library(args.out)
    processor = AutoProcessor.from_pretrained(args.model)
    chosen, text, latency = choose(load_gemma(args.model), processor, images, args.instruction)
    print(json.dumps({"instruction": args.instruction, "model": args.model, "design": chosen,
                      "name": None if chosen is None else designs[chosen]["name"], "answer": text,
                      "latency_s": round(latency, 3)}), flush=True)


@torch.no_grad()
def rollout(env, features, policy, goal_of, seed, weights):
    from sim_rl.curriculum import UNIFORM_WEIGHTS  # noqa: F401  (weights passed explicitly)
    env.write_curriculum(torch.tensor(weights, device=env.device, dtype=torch.float32))
    env.reset(seed)
    n = env.n
    hidden = torch.zeros((n, policy.hidden_size), device=env.device)
    starts = torch.ones(n, device=env.device)
    finished = torch.zeros(n, dtype=torch.bool, device=env.device)
    success = torch.zeros(n, device=env.device)
    designs = torch.full((n,), -1, dtype=torch.long, device=env.device)
    for _ in range(env.horizon):
        designs = torch.where(finished | (designs >= 0), designs, env.design.long())
        cls, tokens = features.observe(env)
        inputs, _ = policy.actor_inputs_and_features(cls, tokens, goal_of(env.design.long()), env.proprio, env.privileged)
        mean, hidden = policy.step(inputs, hidden, starts)
        env.step(mean.clamp(-1, 1).contiguous())
        newly = (env.done > 0.5) & ~finished
        success = torch.where(newly, env.success, success)
        finished |= newly
        starts.copy_(env.done)
    torch.cuda.synchronize()
    return success.cpu().numpy(), designs.cpu().numpy()


def fly(args):
    from sim_rl.cli import TEST_SEED, TEST_STYLE, build_policy, load_checkpoint, validate_checkpoint
    from sim_rl.curriculum import UNIFORM_WEIGHTS
    from sim_rl.features import FrozenFeatures
    from sim_rl.native import CudaEnv
    from sim_rl.stats import wilson
    chosen = json.loads((args.out / "picks.json").read_text())
    saved = load_checkpoint(args.checkpoint)
    meta = saved["metadata"]
    features = FrozenFeatures(meta["encoder"], torch.device("cuda:0"))
    results = {}
    with CudaEnv(ROOT / ".sim-target/training/libsim_env.so", args.num_envs, TEST_SEED, meta["width"], meta["height"],
                 TEST_STYLE, meta["control_mode"]) as env, env.execution():
        validate_checkpoint(saved, env, features)
        policy = build_policy(env, features)
        policy.load_state_dict(saved["model"])
        policy.eval()
        table = features.goal_table(env.goal_images)
        k = table.shape[0]
        conditions = {"oracle_goal": torch.arange(k, device=env.device)}
        for style, mapping in chosen["picks"].items():
            # An unparseable answer gives no goal: zero vector, recorded as a miss.
            conditions[f"gemma_{style}"] = torch.tensor([mapping[str(i)] if mapping[str(i)] is not None else -1
                                                         for i in range(k)], device=env.device)
        weights = UNIFORM_WEIGHTS + (0.0,)
        for name, index in conditions.items():
            def goal_of(design, index=index):
                selected = index[design]
                goals = table[selected.clamp_min(0)]
                return torch.where((selected >= 0).unsqueeze(-1), goals, torch.zeros_like(goals))
            success, _ = rollout(env, features, policy, goal_of, TEST_SEED, weights)
            hits = int(success.sum())
            results[name] = {"episodes": len(success), "success": hits / len(success),
                             "success_wilson95": wilson(hits, len(success))}
            print(name, json.dumps(results[name]), flush=True)
    (args.out / "flight.json").write_text(json.dumps({"checkpoint": str(args.checkpoint), "pick_accuracy": chosen["accuracy"],
                                                      "results": results}, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("dump", "pick", "resolve", "fly"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="google/gemma-4-E4B-it")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "runs/drone4-mean-seed0/best.pt")
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--instruction", help="free-text goal for the resolve stage")
    args = parser.parse_args()
    {"dump": dump, "pick": pick, "resolve": resolve, "fly": fly}[args.stage](args)


if __name__ == "__main__":
    main()
