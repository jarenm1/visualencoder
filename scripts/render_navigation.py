"""Record an actual trained GPU policy; one explicit post-rollout image export."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "sim/python"), str(ROOT / "src")]
from sim_rl.cli import sha256
from sim_rl.features import FrozenFeatures
from sim_rl.native import CudaEnv
from sim_rl.ppo import Policy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("reports/experiments/navigation-rollout"))
    args = parser.parse_args()
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if saved.get("format") != "sim-visual-ppo-v1" or saved.get("steps", 0) < 1:
        raise ValueError("an actual trained native policy checkpoint is required")
    metadata = saved["metadata"]
    if not metadata["actor_inputs"].startswith("pooled visual"):
        raise ValueError("this artifact must show an actual visual policy, not a privileged oracle")
    if sha256(metadata["native_library"]) != metadata["native_sha256"]:
        raise ValueError("checkpoint and actual native simulator binary differ")
    width, height = metadata["width"], metadata["height"]
    args.out.mkdir(parents=True, exist_ok=False)
    with CudaEnv(metadata["native_library"], 1, metadata["final_test_seed"], width, height,
                 metadata["final_test_style"]) as env, env.execution(), torch.no_grad():
        features = FrozenFeatures(metadata["encoder"], env.device)
        if features.fingerprint != metadata["encoder_sha256"]:
            raise ValueError("checkpoint and actual encoder weights differ")
        policy = Policy(features.dim, metadata["ppo"]["hidden_size"]).to(env.device).eval()
        policy.load_state_dict(saved["policy"], strict=True)
        env.reset(metadata["final_test_seed"])
        hidden = torch.zeros((1, policy.hidden_size), device=env.device)
        starts = torch.ones(1, device=env.device)
        images = torch.empty((150, height, width, 3), dtype=torch.uint8, device=env.device)
        transitions = torch.empty((150, 4), device=env.device)
        action = torch.empty(1, dtype=torch.int64, device=env.device)
        for frame in range(150):
            images[frame].copy_((env.obs[0].permute(1, 2, 0) * 255).round().clamp(0, 255).to(torch.uint8))
            logits, _, hidden = policy(features(env), env.proprio, hidden, starts)
            action.copy_(logits.argmax(-1))
            env.step(action)
            transitions[frame].copy_(torch.stack((action[0].float(), env.done[0], env.success[0], env.collision[0])))
            starts.copy_(env.done)
        # Artifact export is explicitly outside timed/profiled rollout collection.
        env.stream.synchronize()
        images_cpu = images.cpu().numpy()
        trace = transitions.cpu().numpy()
    completed = np.flatnonzero(trace[:, 1])
    if not len(completed):
        raise ValueError("finite-horizon task did not complete")
    last = int(completed[0])
    frames = []
    for index in range(0, last + 1, 3):
        image = Image.fromarray(images_cpu[index]).resize((width * 2, height * 2), Image.Resampling.NEAREST)
        draw = ImageDraw.Draw(image)
        draw.text((8, 8), f"t={index * 0.05:.2f}s   action={int(trace[index,0])}", fill="white", stroke_width=1, stroke_fill="black")
        frames.append(image)
    frames[0].save(args.out / "rollout.gif", save_all=True, append_images=frames[1:], duration=150, loop=0)
    indices = sorted(set([0, min(40, last), min(60, last), last]))
    sheet = Image.new("RGB", (width * len(indices), height))
    for column, index in enumerate(indices):
        image = Image.fromarray(images_cpu[index])
        ImageDraw.Draw(image).text((6, 6), f"{index * 0.05:.2f}s", fill="white", stroke_width=1, stroke_fill="black")
        sheet.paste(image, (width * column, 0))
    sheet.save(args.out / "contact.png")
    result = {"checkpoint": str(args.checkpoint), "encoder": metadata["encoder"],
              "selected_steps": saved["steps"], "seed": metadata["final_test_seed"],
              "style": metadata["final_test_style"], "episode_steps": last + 1,
              "success": bool(trace[last, 2]), "collision": bool(trace[last, 3]),
              "note": "Single explicit visual-inspection episode, not the aggregate 128-episode success estimate."}
    (args.out / "rollout.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
