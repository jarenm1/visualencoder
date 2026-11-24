"""Re-render the presentation flights with Poly Haven materials, without rerunning the policy.

Run in the root dev shell with the checkout's venv and FFmpeg on PATH. The
right pane comes straight from each recorded replay's external camera.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

from visenc.city import Appearance, CityRenderer, Pose

SHOTS = (("replay-drone9-s7", 3, "Held-out takeoff and car approach"),
         ("replay-drone11-train101", 15, "Memory-enabled pilot in a training district"))
# Both recorded checkpoints use this native monocular camera calibration.
UPTILT = math.pi / 12
VFOV = 5 * math.pi / 12
WIDTH, HEIGHT = 480, 360


def pose_at(state: np.ndarray, fields: list[str]) -> Pose:
    eye = state[[fields.index(f"world_{c}") for c in "xyz"]]
    q = state[[fields.index(f"quat_{c}") for c in "xyzw"]]
    q = q / np.linalg.norm(q)

    def rotate(vector):
        cross = np.cross(q[:3], vector)
        return vector + 2 * (q[3] * cross + np.cross(q[:3], cross))

    forward = rotate(np.array([0., math.sin(UPTILT), -math.cos(UPTILT)]))
    up = rotate(np.array([0., math.cos(UPTILT), math.sin(UPTILT)]))
    yaw = math.atan2(-forward[0], -forward[2])
    pitch = math.asin(np.clip(forward[1], -1., 1.))
    right = np.array([math.cos(yaw), 0., -math.sin(yaw)])
    roll = math.atan2(up @ right, up @ np.cross(right, forward))
    return Pose(eye + rotate(np.array([0., .05, -.1])), yaw, pitch, roll, VFOV)


def daylight() -> Appearance:
    return replace(Appearance.canonical(), sun_dir=(-.6, .65, .45), sun_color=(1.45, 1.30, 1.13),
                   sky_ambient=(.36, .46, .59), ground_ambient=(.21, .19, .17),
                   sky=(.61, .75, .89), fog_density=.0014)


def export(encoder, ffmpeg: str, manifest, episode, source: Path, title: str, out: Path):
    x, y, w, h = manifest["views"]["third_person"]
    fps = manifest["control_hz"]
    labels = (
        "pad=960:420:0:30:color=0x131722,"
        f"drawtext=text='{title}  /  Location hint enabled':x=12:y=6:fontsize=16:fontcolor=white,"
        "drawtext=text='Textured re-render from recorded pose':x=12:y=396:fontsize=15:fontcolor=white,"
        "drawtext=text='Recorded external camera':x=492:y=396:fontsize=15:fontcolor=white"
    )
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-filter_complex_threads", "2",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{WIDTH}x{HEIGHT}", "-r", str(fps), "-i", "pipe:0",
           "-threads", "2", "-i", str(source), "-filter_complex",
           (f"[1:v]crop={w}:{h}:{x}:{y},scale={WIDTH}:{HEIGHT}[external];"
            f"[0:v][external]hstack=shortest=1,{labels}[out]"), "-map", "[out]", "-an", "-c:v", "libx264",
           "-preset", "fast", "-crf", "19", "-threads", "3", "-pix_fmt", "yuv420p", str(out)]
    with subprocess.Popen(cmd, stdin=subprocess.PIPE) as process:
        assert process.stdin is not None
        try:
            for row in episode["states"]:
                frame = encoder.render_rgb(pose_at(np.array(row), manifest["state_fields"]), daylight())
                process.stdin.write(frame.tobytes())
        finally:
            process.stdin.close()
        if process.wait():
            raise RuntimeError(f"FFmpeg failed writing {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--figure", type=Path, required=True, help="path for the two-frame comparison still")
    p.add_argument("--runs", type=Path, default=Path("runs"))
    p.add_argument("--video", action="store_true", help="also generate pilot-demo-textured.mp4")
    p.add_argument("--ffmpeg", default="ffmpeg")
    args = p.parse_args()
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    work = args.runs / "presentation"
    work.mkdir(parents=True, exist_ok=True)
    stills, parts = [], []
    for index, ((directory, ident, title), frame) in enumerate(zip(SHOTS, (90, 75), strict=True), start=1):
        folder = args.runs / directory
        manifest = json.loads((folder / "manifest.json").read_text())
        episode = next(e for e in manifest["environments"] if e["id"] == ident)
        with CityRenderer(Path(manifest["city_path"]), WIDTH, HEIGHT, presentation=True) as renderer:
            pose = pose_at(np.array(episode["states"][frame]), manifest["state_fields"])
            stills.append(renderer.render_rgb(pose, daylight()))
            if args.video:
                out = work / f"textured-shot-{index}.mp4"
                export(renderer, args.ffmpeg, manifest, episode, folder / episode["video"], title, out)
                parts.append(out.name)
                print(f"Wrote {out}", flush=True)
    Image.fromarray(np.concatenate(stills, axis=1)).save(args.figure)
    print(f"Wrote {args.figure}", flush=True)
    if args.video:
        listing = work / "textured-shots.txt"
        listing.write_text("".join(f"file '{name}'\n" for name in parts))
        subprocess.run([args.ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "1",
                        "-i", str(listing), "-c", "copy", "-movflags", "+faststart",
                        str(work / "pilot-demo-textured.mp4")], check=True)


if __name__ == "__main__":
    main()
