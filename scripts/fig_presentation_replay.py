"""Presentation figures from recorded replay runs. Recorded data only; no rendering of model outputs.

    python scripts/fig_presentation_replay.py

Writes into report/figures/:
  patch-pca.png      FPV with 16px patch grid | same-frame encoder PCA RGB (20x15, nearest-upsampled)
  pilot-match.png    goal crop | current FPV | FPV + recorded conditional patch saliency
  pilot-memory.png   heading-up ego memory at three timestamps (goal evidence, looked-at coverage)
  pilot-rollout.png  third-person frames from the same v4 visual policy's successful rollout
  pilot-transition.png  policy camera at the illustrative PPO transition

Sources: per-frame mp4 (decoded with ffmpeg) and .encoder/.memory/.saliency arrays laid out as in
each run's manifest.json. Saliency is the conditional patch probability within the frame, scaled by
the frame maximum; the trailing byte is P(in view) (value/255) and is reported, not drawn.
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "figures"

V4 = ROOT / "runs" / "replay-drone11-train101"  # v4 memory checkpoint, train split

MATCH_ENV = 15  # vehicle goal, outcome success, 347 frames
MATCH_FRAME = 240  # 8.0 s
MEMORY_FRAMES = (0, 120, 240)  # 0.0 s, 4.0 s, 8.0 s
ROLLOUT_FRAMES = (0, 120, 346)  # 0.0 s, 4.0 s, 11.53 s (terminal success record)
TRANSITION_FRAME = 165  # 5.5 s; logged action and reward shown on the PPO slide

CELL = 16  # px per patch: 320/20 == 240/15
GUTTER = 8
HIGHLIGHT = (255, 210, 0)
SALIENCY_RGB = np.array([255, 140, 0], np.float32)
SALIENCY_MAX_ALPHA = 0.5
MEM_DARK = np.array([14, 20, 32], np.float32)
MEM_LOOKED = np.array([64, 196, 206], np.float32)  # looked-at coverage (channel 1)
MEM_GOAL = np.array([255, 150, 40], np.float32)  # goal evidence (channel 0)
MEM_SCALE = 10  # 32 cells -> 320 px


def ffmpeg_bin() -> str:
    if os.environ.get("FFMPEG"):
        return os.environ["FFMPEG"]
    found = shutil.which("ffmpeg")
    if found:
        return found
    nix = sorted(glob.glob("/nix/store/*-ffmpeg-full-*-bin/bin/ffmpeg"))
    nix = nix or sorted(glob.glob("/nix/store/*-ffmpeg-*-bin/bin/ffmpeg"))
    if not nix:
        raise SystemExit("ffmpeg not found: set FFMPEG or put it on PATH")
    return nix[-1]


class Episode:
    def __init__(self, run: Path, env_id: int, ffmpeg: str):
        self.ffmpeg = ffmpeg
        self.run = run
        self.manifest = json.loads((run / "manifest.json").read_text())
        self.env = next(e for e in self.manifest["environments"] if e["id"] == env_id)
        self.n = self.env["frame_count"]
        self.hz = self.manifest["control_hz"]
        m = self.manifest
        enc, mem, sal = m["encoder_map"], m["memory_map"], m["saliency"]
        self.enc = np.fromfile(run / self.env["encoder_map"], np.uint8).reshape(
            self.n, enc["height"], enc["width"], enc["channels"]
        )
        self.mem = np.fromfile(run / self.env["memory_map"], np.uint8).reshape(
            self.n, mem["channels"], mem["height"], mem["width"]
        )
        cells = sal["width"] * sal["height"]
        raw = np.fromfile(run / self.env["saliency"], np.uint8).reshape(self.n, cells + 1)
        self.cond = raw[:, :cells].reshape(self.n, sal["height"], sal["width"])
        self.p_in_view = raw[:, cells].astype(np.float32) / 255.0

    def label(self) -> str:
        return f"{self.run.name} env-{self.env['id']:03d} ({self.env['outcome']}, {self.n} frames)"

    def frames(self, wanted) -> dict[int, np.ndarray]:
        """Decode the video up to the last wanted frame; returns full side-by-side RGB frames."""
        m = self.manifest
        w, h = m["video_width"], m["video_height"]
        last = max(wanted)
        cmd = [
            self.ffmpeg, "-v", "error", "-i", str(self.run / self.env["video"]),
            "-frames:v", str(last + 1), "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        ]
        raw = subprocess.run(cmd, check=True, capture_output=True).stdout
        size = w * h * 3
        assert len(raw) >= (last + 1) * size, "video shorter than requested frame"
        arr = np.frombuffer(raw, np.uint8, count=(last + 1) * size).reshape(last + 1, h, w, 3)
        return {k: arr[k] for k in wanted}

    def view(self, frame: np.ndarray, name: str) -> np.ndarray:
        x, y, w, h = self.manifest["views"][name]
        return frame[y : y + h, x : x + w]

    def t(self, k: int) -> float:
        return k / self.hz


def upsample(cells: np.ndarray) -> np.ndarray:
    return np.repeat(np.repeat(cells, CELL, axis=0), CELL, axis=1)


def patch_grid(img: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    out = img.astype(np.float32)
    h, w = img.shape[:2]
    for x in range(CELL, w, CELL):
        out[:, x] = (1 - alpha) * out[:, x] + alpha * 255
    for y in range(CELL, h, CELL):
        out[y] = (1 - alpha) * out[y] + alpha * 255
    return out.round().clip(0, 255).astype(np.uint8)


def highlight(img: np.ndarray, row: int, col: int) -> np.ndarray:
    im = Image.fromarray(img)
    x0, y0 = col * CELL, row * CELL
    ImageDraw.Draw(im).rectangle([x0, y0, x0 + CELL - 1, y0 + CELL - 1], outline=HIGHLIGHT, width=2)
    return np.asarray(im)


def saliency_overlay(fpv: np.ndarray, cond: np.ndarray) -> np.ndarray:
    alpha = (upsample(cond.astype(np.float32) / 255.0) * SALIENCY_MAX_ALPHA)[..., None]
    out = fpv.astype(np.float32) * (1 - alpha) + SALIENCY_RGB * alpha
    return out.round().clip(0, 255).astype(np.uint8)


def memory_panel(mem: np.ndarray) -> np.ndarray:
    """mem: (2, 32, 32) uint8, channel 0 = goal evidence, channel 1 = looked-at, forward up."""
    looked = mem[1].astype(np.float32)[..., None] / 255.0
    goal = mem[0].astype(np.float32)[..., None] / 255.0
    base = MEM_DARK + (MEM_LOOKED - MEM_DARK) * looked
    rgb = base + (MEM_GOAL - base) * goal
    img = np.repeat(np.repeat(rgb.round().clip(0, 255).astype(np.uint8), MEM_SCALE, 0), MEM_SCALE, 1)
    im = Image.fromarray(img)
    cx, cy = img.shape[1] // 2, img.shape[0] // 2
    ImageDraw.Draw(im).polygon(
        [(cx, cy - 12), (cx - 9, cy + 10), (cx, cy + 4), (cx + 9, cy + 10)],
        fill=(255, 255, 255), outline=(20, 20, 20),
    )
    return np.asarray(im)


def hstack(panels: list[np.ndarray]) -> np.ndarray:
    gutter = np.full((panels[0].shape[0], GUTTER, 3), 255, np.uint8)
    parts = []
    for i, p in enumerate(panels):
        if i:
            parts.append(gutter)
        parts.append(p)
    return np.concatenate(parts, axis=1)


def save(img: np.ndarray, name: str) -> None:
    Image.fromarray(img).save(OUT / name, optimize=True)
    print(f"wrote report/figures/{name}  {img.shape[1]}x{img.shape[0]}")


def main() -> None:
    ffmpeg = ffmpeg_bin()
    print(f"ffmpeg: {ffmpeg}")
    OUT.mkdir(parents=True, exist_ok=True)

    # patch-pca.png: FPV grid + same-frame encoder PCA, same highlighted patch.
    ep = Episode(V4, MATCH_ENV, ffmpeg)
    frames = ep.frames(sorted({MATCH_FRAME, TRANSITION_FRAME, *ROLLOUT_FRAMES}))
    frame = frames[MATCH_FRAME]
    fpv = ep.view(frame, "first_person")
    row, col = np.unravel_index(ep.cond[MATCH_FRAME].argmax(), ep.cond[MATCH_FRAME].shape)
    row, col = int(row), int(col)
    pca = upsample(ep.enc[MATCH_FRAME])
    save(hstack([highlight(patch_grid(fpv), row, col), highlight(patch_grid(pca), row, col)]), "patch-pca.png")
    print(
        f"  patch-pca: {ep.label()} frame {MATCH_FRAME} t={ep.t(MATCH_FRAME):.2f}s "
        f"P(in view)={ep.p_in_view[MATCH_FRAME]:.3f} highlight row {row} col {col}"
    )

    # pilot-match.png: goal crop | current FPV | FPV + recorded conditional saliency.
    goal = np.asarray(Image.open(ep.run / ep.env["goal"]).convert("RGB"))
    if goal.shape[:2] != (240, 320):
        goal = np.asarray(Image.fromarray(goal).resize((320, 240), Image.LANCZOS))
    save(hstack([goal, fpv, saliency_overlay(fpv, ep.cond[MATCH_FRAME])]), "pilot-match.png")
    print(
        f"  pilot-match: goal {ep.env['goal']} | FPV frame {MATCH_FRAME} | overlay conditional saliency "
        f"(max-normalised, alpha<={SALIENCY_MAX_ALPHA}); P(in view) {ep.p_in_view[MATCH_FRAME]:.3f} not drawn"
    )

    # pilot-memory.png: ego memory at three timestamps, shared palette.
    save(hstack([memory_panel(ep.mem[k]) for k in MEMORY_FRAMES]), "pilot-memory.png")
    for k in MEMORY_FRAMES:
        print(
            f"  pilot-memory: frame {k} t={ep.t(k):.2f}s P(in view)={ep.p_in_view[k]:.3f} "
            f"goal-evidence max={int(ep.mem[k, 0].max())} looked-at cells={int((ep.mem[k, 1] > 0).sum())}"
        )

    # pilot-rollout.png: third-person frames from the same visual pilot's successful flight.
    save(hstack([ep.view(frames[k], "third_person") for k in ROLLOUT_FRAMES]), "pilot-rollout.png")
    print(f"  pilot-rollout: {ep.label()} third-person frames {ROLLOUT_FRAMES}")
    save(ep.view(frames[TRANSITION_FRAME], "first_person"), "pilot-transition.png")
    print(f"  pilot-transition: frame {TRANSITION_FRAME}, {ep.env['transitions'][TRANSITION_FRAME]}")


if __name__ == "__main__":
    main()
