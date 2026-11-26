"""Re-render both recorded cameras with Poly Haven materials and presentation lighting.

Run `python -m scripts.render_textured_replay` from the root dev shell. The
recorded frame timing, policy tensors, and outcomes are preserved.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pyray as rl

from visenc.city import CityRenderer, Pose

from .render_presentation import daylight, pose_at


def chase_at(eye: np.ndarray, target: np.ndarray) -> Pose:
    dx, dy, dz = target - eye
    return Pose(eye, math.atan2(-dx, -dz), math.atan2(dy, math.hypot(dx, dz)), 0., math.pi / 3)


def draw_drone(position: np.ndarray, rotation: np.ndarray) -> None:
    """Replay proxy, using the actual recorded body attitude."""
    axis = rotation[:3]
    axis_length = float(np.linalg.norm(axis))
    rl.rl_push_matrix()
    rl.rl_translatef(*position)
    if axis_length > 1e-6:
        rl.rl_rotatef(math.degrees(2 * math.atan2(axis_length, rotation[3])), *(axis / axis_length))
    rl.draw_cube((0., 0., 0.), .20, .09, .24, rl.ORANGE)
    for angle in (-45., 45.):
        rl.rl_push_matrix()
        rl.rl_rotatef(angle, 0., 1., 0.)
        rl.draw_cube((0., 0., 0.), .43, .035, .035, rl.DARKGRAY)
        rl.rl_pop_matrix()
    for x in (-.145, .145):
        for z in (-.145, .145):
            rl.draw_cylinder((x, .025, z), .085, .085, .018, 16, rl.DARKGRAY)
            rl.draw_cylinder((x, .04, z), .021, .021, .025, 12, rl.ORANGE)
    rl.draw_cube((0., 0., -.128), .11, .05, .025, rl.GOLD)
    rl.rl_pop_matrix()


def render_video(renderer: CityRenderer, manifest: dict, episode: dict, eyes: list, dest: Path,
                 ffmpeg: str) -> None:
    width, height = manifest["frame_width"], manifest["frame_height"]
    fps = manifest["control_hz"]
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
           "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", f"{2 * width}x{height}", "-framerate", str(fps),
           "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "19", "-threads", "3",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dest)]
    fields = manifest["state_fields"]
    xyz = [fields.index(f"world_{axis}") for axis in "xyz"]
    xyzw = [fields.index(f"quat_{axis}") for axis in "xyzw"]
    look = daylight()
    frame = np.empty((height, 2 * width, 3), dtype=np.uint8)
    with subprocess.Popen(cmd, stdin=subprocess.PIPE) as process:
        assert process.stdin is not None
        with process.stdin:
            for row, eye in zip(episode["states"], eyes, strict=True):
                state = np.asarray(row)
                position, rotation = state[xyz], state[xyzw]
                frame[:, :width] = renderer.render_rgb(pose_at(state, fields), look)
                frame[:, width:] = renderer.render_rgb(chase_at(np.asarray(eye), position), look,
                                                      extra=lambda: draw_drone(position, rotation))
                process.stdin.write(memoryview(frame))
        if process.wait():
            raise subprocess.CalledProcessError(process.returncode, cmd)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("out", type=Path, help="new replay directory")
    parser.add_argument("--chase-poses", required=True, type=Path,
                        help="output from sim-replay-render --chase-poses on the source manifest")
    parser.add_argument("--episode", type=int, help="render just one environment ID")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()
    manifest = json.loads((args.source / "manifest.json").read_text())
    if manifest["format"] != "sim-city-replay-v3":
        parser.error("expected a sim-city-replay-v3 manifest")
    width, height = manifest["frame_width"], manifest["frame_height"]
    if manifest["views"] != {"first_person": [0, 0, width, height], "third_person": [width, 0, width, height]}:
        parser.error("expected a side-by-side onboard and external recording")
    if args.episode is not None:
        manifest["environments"] = [e for e in manifest["environments"] if e["id"] == args.episode]
        if not manifest["environments"]:
            parser.error(f"no environment {args.episode} in {args.source}")
    city = (args.source / manifest["city_path"]).resolve()
    with args.chase_poses.open() as handle:
        chase = dict(json.loads(line) for line in handle)
    for episode in manifest["environments"]:
        if len(chase.get(episode["id"], ())) != episode["frame_count"]:
            parser.error(f"chase camera frame count mismatch for environment {episode['id']}")
    args.out.mkdir(parents=True)
    print(f"Rendering {len(manifest['environments'])} episodes from {city}", flush=True)
    with CityRenderer(city, width, height, presentation=True) as renderer:
        for episode in manifest["environments"]:
            print(f"env {episode['id']}: {episode['frame_count']} frames", flush=True)
            video = episode["video"]
            render_video(renderer, manifest, episode, chase[episode["id"]], args.out / video, args.ffmpeg)
            for key in ("goal", "encoder_map", "saliency", "memory_map"):
                if filename := episode.get(key):
                    shutil.copyfile(args.source / filename, args.out / filename)
    manifest["presentation_render"] = "Textured re-render · policy overlays and outcomes are from the original run"
    manifest["city_path"] = str(city)
    (args.out / "manifest.json").write_text(json.dumps(manifest, separators=(",", ":")) + "\n")
    print(f"Wrote {args.out / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
