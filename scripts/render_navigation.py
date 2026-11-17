"""Record a trained city GPU policy and world-path diagnostics after rollout."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "sim/python"), str(ROOT / "src")]
from sim_rl.cli import load_checkpoint, validate_checkpoint
from sim_rl.features import FrozenFeatures
from sim_rl.native import DEFAULT_CITY, CudaEnv
from sim_rl.ppo import Policy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("reports/experiments/city-navigation-rollout"))
    parser.add_argument("--city", type=Path, default=DEFAULT_CITY)
    parser.add_argument("--library", type=Path, default=ROOT / ".sim-target/debug/libsim_env.so")
    args = parser.parse_args()
    saved = load_checkpoint(args.checkpoint, args)
    metadata = saved["metadata"]
    if not metadata["actor_inputs"].startswith("pooled visual"):
        raise ValueError("this artifact must show an actual visual policy, not a privileged oracle")
    width, height = metadata["width"], metadata["height"]
    args.out.mkdir(parents=True, exist_ok=False)
    with CudaEnv(args.library, 1, metadata["final_test_seed"], width, height,
                 metadata["final_test_style"], args.city) as env, env.execution(), torch.no_grad():
        features = FrozenFeatures(metadata["encoder"], env.device)
        validate_checkpoint(saved, env, features)
        policy = Policy(features.dim, env.action_count, metadata["ppo"]["hidden_size"]).to(env.device).eval()
        policy.load_state_dict(saved["policy"], strict=True)
        env.reset(metadata["final_test_seed"])
        hidden = torch.zeros((1, policy.hidden_size), device=env.device)
        starts = torch.ones(1, device=env.device)
        images = torch.empty((env.horizon, height, width, 3), dtype=torch.uint8, device=env.device)
        transitions = torch.empty((env.horizon, 4), device=env.device)
        diagnostics = torch.empty((env.horizon, 8), device=env.device)
        action = torch.empty(1, dtype=torch.int64, device=env.device)
        for frame in range(env.horizon):
            diagnostics[frame].copy_(env.diagnostics[0])
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
        path = diagnostics.cpu().numpy()
        native_metadata = env.metadata
        control_hz = native_metadata["control_hz"]
    completed = np.flatnonzero(trace[:, 1])
    if not len(completed):
        raise ValueError("finite-horizon task did not complete")
    last = int(completed[0])
    frames = []
    for index in range(0, last + 1, 3):
        image = Image.fromarray(images_cpu[index]).resize((width * 2, height * 2), Image.Resampling.NEAREST)
        draw = ImageDraw.Draw(image)
        draw.text((8, 8), f"city t={index / control_hz:.2f}s   action={int(trace[index,0])}", fill="white", stroke_width=1, stroke_fill="black")
        frames.append(image)
    frames[0].save(args.out / "rollout.gif", save_all=True, append_images=frames[1:],
                   duration=round(3000 / control_hz), loop=0)
    indices = sorted(set([0, min(40, last), min(60, last), last]))
    sheet = Image.new("RGB", (width * len(indices), height))
    for column, index in enumerate(indices):
        image = Image.fromarray(images_cpu[index])
        ImageDraw.Draw(image).text((6, 6), f"{index / control_hz:.2f}s", fill="white", stroke_width=1, stroke_fill="black")
        sheet.paste(image, (width * column, 0))
    sheet.save(args.out / "contact.png")
    # Diagnostics are exported only after GPU collection; never actor inputs.
    episode_path = path[:last + 1]
    points = episode_path[:, [0, 2]]
    goal = episode_path[0, [3, 5]]
    lower = np.minimum(points.min(0), goal) - 2
    upper = np.maximum(points.max(0), goal) + 2
    scale = 640 / max(upper - lower)
    project = lambda point: tuple(((point - lower) * scale + 40).tolist())
    map_image = Image.new("RGB", (720, 760), "white")
    draw = ImageDraw.Draw(map_image)
    pixels = [project(point) for point in points]
    if len(pixels) > 1:
        draw.line(pixels, fill="navy", width=3)
    for point, color in ((points[0], "green"), (goal, "red")):
        x, y = project(point)
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color)
    draw.text((12, 720), "Actual world X/Z path (m); green=start, red=goal. Not a geometry map.", fill="black")
    map_image.save(args.out / "path.png")
    (args.out / "path.json").write_text(json.dumps(
        {"fields": ["world_x", "world_y", "world_z", "goal_x", "goal_y", "goal_z", "route_id", "clock"],
         "semantics": "pre-action states, first episode only; terminal position precedes the final action",
         "diagnostics": episode_path.tolist(), "action_done_success_collision": trace[:last + 1].tolist(),
         "native_metadata": native_metadata}, indent=2) + "\n")
    result = {"checkpoint": str(args.checkpoint), "encoder": metadata["encoder"],
              "selected_steps": saved["steps"], "seed": metadata["final_test_seed"],
              "style": metadata["final_test_style"], "episode_steps": last + 1,
              "success": bool(trace[last, 2]), "collision": bool(trace[last, 3]),
              "native_metadata": native_metadata,
              "note": "Single explicit visual-inspection episode, not the aggregate 128-episode success estimate."}
    (args.out / "rollout.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
