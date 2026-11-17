"""Render one visual-goal rollout of a trained city-drone policy after the GPU rollout completes."""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "sim/python")]
from sim_rl.cli import DEFAULT_LIBRARY, TEST_SEED, TEST_STYLE, load_checkpoint  # noqa: E402
from sim_rl.features import FrozenFeatures, precision_of  # noqa: E402
from sim_rl.native import DEFAULT_CITY  # noqa: E402
from sim_rl.replay import OUTCOME_FIELDS, STATE_FIELDS, capture  # noqa: E402


def draw_label(image, text):
    ImageDraw.Draw(image).text((8, 8), text, fill="white", stroke_width=1, stroke_fill="black")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("reports/experiments/city-drone-rollout"))
    parser.add_argument("--city", type=Path, default=DEFAULT_CITY)
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--seed", type=int, default=TEST_SEED)
    parser.add_argument("--no-gps-fraction", type=float, default=0.0,
                        help="probability the episode denies GPS (0 = GPS available, 1 = denied)")
    args = parser.parse_args()
    if not 0.0 <= args.no_gps_fraction <= 1.0:
        raise ValueError("--no-gps-fraction must be in [0, 1]")
    saved = load_checkpoint(args.checkpoint)
    metadata = saved["metadata"]
    features = FrozenFeatures(metadata["encoder"], torch.device("cuda:0"), precision_of(metadata["preprocessing"]))
    recording = capture(SimpleNamespace(library=args.library, city=args.city, num_envs=1),
                        saved, features, args.seed, TEST_STYLE, args.no_gps_fraction)
    native = recording["native"]
    action_names = native["action_names"]
    transition_fields = action_names + OUTCOME_FIELDS
    images = recording["images"][:, 0]
    states = recording["states"][:, 0]
    transitions = recording["transitions"][:, 0]
    control_hz = native["control_hz"]
    width, height = metadata["width"], metadata["height"]
    completed = np.flatnonzero(transitions[:, transition_fields.index("done")])
    if not len(completed):
        raise ValueError("episode did not reach a terminal state within the horizon")
    last = int(completed[0])
    length = last + 1
    args.out.mkdir(parents=True, exist_ok=False)

    frames = []
    for index in range(0, length, 3):
        image = Image.fromarray(images[index]).resize((width * 2, height * 2), Image.Resampling.NEAREST)
        draw_label(image, f"t={index / control_hz:.2f}s {action_names[0]}={transitions[index, 0]:+.2f}")
        frames.append(image)
    frames[0].save(args.out / "rollout.gif", save_all=True, append_images=frames[1:],
                   duration=round(3000 / control_hz), loop=0)

    indices = sorted(set([0, min(40, last), min(60, last), last]))
    sheet = Image.new("RGB", (width * len(indices), height))
    for column, index in enumerate(indices):
        image = Image.fromarray(images[index])
        draw_label(image, f"{index / control_hz:.2f}s")
        sheet.paste(image, (width * column, 0))
    sheet.save(args.out / "contact.png")

    # Diagnostics come from the post-rollout export only; they are never actor inputs.
    episode = states[:length]
    points = episode[:, [STATE_FIELDS.index("world_x"), STATE_FIELDS.index("world_z")]]
    goal = episode[0, [STATE_FIELDS.index("target_x"), STATE_FIELDS.index("target_z")]]
    lower = np.minimum(points.min(0), goal) - 2
    upper = np.maximum(points.max(0), goal) + 2
    scale = 640 / max(upper - lower)

    def project(point):
        return tuple(((point - lower) * scale + 40).tolist())

    map_image = Image.new("RGB", (720, 760), "white")
    draw = ImageDraw.Draw(map_image)
    pixels = [project(point) for point in points]
    if len(pixels) > 1:
        draw.line(pixels, fill="navy", width=3)
    for point, color in ((points[0], "green"), (goal, "red")):
        x, y = project(point)
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color)
    draw.text((12, 720), "Actual world X/Z path (m); green=start, red=goal target. Not a geometry map.", fill="black")
    map_image.save(args.out / "path.png")
    (args.out / "path.json").write_text(json.dumps({
        "state_fields": STATE_FIELDS, "transition_fields": transition_fields,
        "semantics": "pre-action states and the action applied at each tick; first episode only",
        "states": episode.tolist(), "transitions": transitions[:length].tolist(),
        "native": native}, indent=2) + "\n")

    success_column = transition_fields.index("success")
    outcome = ("success" if transitions[last, success_column] else
               "collision" if transitions[last, transition_fields.index("collision")] else
               "out_of_bounds" if transitions[last, transition_fields.index("out_of_bounds")] else "timeout")
    result = {"checkpoint": str(args.checkpoint), "encoder": metadata["encoder"],
              "selected_steps": metadata["steps"], "seed": args.seed, "style": TEST_STYLE,
              "control_mode": native["control_mode"], "action_names": action_names,
              "no_gps_fraction": args.no_gps_fraction,
              "gps_available": bool(recording["gps_available"][0] > 0.5),
              "episode_steps": length, "outcome": outcome, "native": native,
              "note": "Single visual-inspection episode, not the aggregate success estimate."}
    (args.out / "rollout.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
