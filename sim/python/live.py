"""Live drone: the trained policy flies in real time in free flight; steer it from a browser.

Drag a box around any object in the drone's view (the operator tap) or, with --gemma, type an
instruction that Gemma 4 grounds to a box in the current frame. The box becomes the goal crop and
the scored target (the city object it frames; boxes framing none are rejected). Free flight: the whole city is open (only its
edge and a 100 m ceiling end a flight), success is reported without ending the flight and there
is no timeout; a crash restarts in a fresh held-out episode with its catalog goal.

    nix develop path:sim --command .venv/bin/python sim/python/live.py \
        --checkpoint runs/drone5-mean-seed0/best.pt [--gemma]
    open http://127.0.0.1:8790/
"""
from __future__ import annotations

import argparse
import io
import json
import os
import queue
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "sim/python"), str(ROOT / "src")]

from agent import Agent, summarize  # noqa: E402
from gemma_goal import MATCH_IOU, detect, iou, project_boxes, start_camera  # noqa: E402
from sim_rl.cli import (DEFAULT_LIBRARY, TEST_STYLE, build_policy, load_checkpoint,  # noqa: E402
                        validate_checkpoint)
from sim_rl.curriculum import UNIFORM_WEIGHTS  # noqa: E402
from sim_rl.features import checkpoint_features, crop_goals, square_goal_box  # noqa: E402
from sim_rl.memory import CELL, GRID, SpatialMemory, observe as observe_memory  # noqa: E402
from sim_rl.model import EXPLORE_COMMAND, GOAL_COMMAND, deterministic_action  # noqa: E402
from sim_rl.native import CudaEnv  # noqa: E402
from sim_rl.perception import CLASSES  # noqa: E402
from sim_rl.tracks import CLASS_THRESHOLD, ObjectTracks  # noqa: E402

PAGE = ROOT / "sim/live"
PAGES = {"/": ("index.html", "text/html; charset=utf-8"), "/app.js": ("app.js", "text/javascript; charset=utf-8"),
         "/style.css": ("style.css", "text/css; charset=utf-8")}
TARGET_CLASSES = ("building", "vehicle", "vegetation", "prop")
MEMORY_IMAGE_EVERY = 6           # control ticks between spatial-memory map images (5 Hz at 30 Hz)
VEHICLE = CLASSES.index("vehicle")


def jpeg(image):
    """float [3,H,W] in [0,1] on GPU -> JPEG bytes."""
    array = (image.clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()


class Pilot:
    """Owns the simulator, encoder and policy; runs on the main thread."""

    def __init__(self, args):
        saved = load_checkpoint(args.checkpoint)
        meta = saved["metadata"]
        self.checkpoint = {"path": str(args.checkpoint), "steps": meta["steps"], "live_goal_refresh": False}
        self.device = torch.device("cuda:0")
        self.features = checkpoint_features(meta, self.device)
        self.env = CudaEnv(DEFAULT_LIBRARY, 1, args.seed, meta["width"], meta["height"], TEST_STYLE, meta["control_mode"])
        self.hz, self.appearance = args.hz, args.appearance
        layout = json.loads((self.env.city_path / "layout.json").read_text())
        targets = [i for i, o in enumerate(layout["objects"]) if o["class"] in TARGET_CLASSES]
        lo = np.array([layout["objects"][i]["min"] for i in targets], dtype=float)
        hi = np.array([layout["objects"][i]["max"] for i in targets], dtype=float)
        self.objects = {"index": np.array(targets), "center": (lo + hi) / 2, "half": (hi - lo) / 2,
                        "class": [layout["objects"][i]["class"] for i in targets],
                        "asset": [layout["objects"][i]["asset"] for i in targets]}
        self.object_info = {i: {"asset": layout["objects"][i]["asset"],
                                "object_index": i} for i in targets}
        district = self.env.metadata["district"]
        # Free flight spans the whole city; the held-out district (where episodes start) is outlined.
        self.map = {"district": [*district["min"], *district["max"]],
                    "city_half_extent": self.env.metadata["city"]["half_extent_m"],
                    "objects": [[round(c[0], 2), round(c[2], 2), round(h[0], 2), round(h[2], 2), cls]
                                for c, h, cls in zip(self.objects["center"], self.objects["half"], self.objects["class"])]}
        self.gemma = None
        if args.gemma:
            from transformers import AutoProcessor
            from gemma_goal import load_gemma
            self.gemma = (load_gemma(args.model), AutoProcessor.from_pretrained(args.model))
        self.requests = queue.Queue()
        self.lock = threading.Lock()
        self.frame, self.goal_image, self.frame_id = b"", b"", 0
        self.status = {"message": "starting"}
        self.paused, self.error, self.last_outcome = False, None, None
        self.goal_revision = 0
        # Kept alive while native reads it on the stream (retarget is asynchronous).
        self.retarget = None
        with self.env.execution():
            validate_checkpoint(saved, self.env, self.features)
            self.policy = build_policy(self.env, self.features, saved)
            self.policy.eval()
            self.hidden = torch.zeros((1, self.policy.hidden_size), device=self.device)
            self.starts = torch.ones(1, device=self.device)
            self.command = torch.full((1,), GOAL_COMMAND, dtype=torch.long, device=self.device)
            self.spatial = (SpatialMemory(1, self.env.height // 16, self.env.width // 16, self.env.metadata["camera"],
                                          self.device) if self.policy.spatial else None)
            self.tracks = ObjectTracks(1, self.device) if self.spatial is not None else None
            self.memory_image, self.memory_info = b"", None
            self.agent, self.agent_info, self.steps = None, None, 0
            self.lanes = torch.tensor(UNIFORM_WEIGHTS + (0.0, args.appearance), device=self.device)
            self.env.write_curriculum(self.lanes)
            self.env.set_free_flight(True)
            self.new_episode(int.from_bytes(os.urandom(7), "little"))

    # --- episode and goal management (simulator thread only) -------------------------------
    def new_episode(self, seed):
        self.env.reset(seed)
        self.starts.fill_(1)   # fresh recurrent state, memory map and car tracks
        self.agent = None
        self.command.fill_(GOAL_COMMAND)
        self.catalog_goal()

    def catalog_goal(self):
        """Goal = the current episode's catalog goal crop (what the drone was asked to find)."""
        self.env.render_goals(int(self.env.catalog_index[0]))
        crop = crop_goals(self.env.goal_views[:1], self.env.goal_crops[:1])
        self.set_goal(crop, "catalog goal")

    def set_goal(self, crop, source, box=None, match_iou=None):
        """A new goal resets the recurrent state and the old goal's map evidence; what the drone has
        looked at and the car tracks stay."""
        self.goal = self.features.goal_embedding(crop)
        self.hidden.zero_()
        if self.spatial is not None:
            self.spatial.map[:, 0].zero_()
        self.goal_revision += 1
        self.goal_started = time.monotonic()
        d = self.env.diagnostics[0].cpu().numpy()
        distance = float(np.linalg.norm(np.maximum(np.abs(d[:3] - d[3:6]) - d[24:27], 0)))
        with self.lock:
            self.goal_image = jpeg(crop[0])
            self.goal_info = {"source": source, "box": box, "revision": self.goal_revision,
                              "match_iou": match_iou, "distance_at_selection_m": round(distance, 2)}

    def box_goal(self, box, source):
        """Crop the current frame at a normalized box and retarget to the city object it frames.
        A box that frames no city object is rejected: keeping the old target would give the policy
        GPS to one object and an image of another."""
        square = square_goal_box(torch.tensor([box], device=self.device))
        crop = crop_goals(self.env.obs.clone(), square)
        diagnostics = self.env.diagnostics.cpu().numpy().astype(float)
        eye, forward, up = start_camera(diagnostics, self.env.metadata)
        camera = self.env.metadata["camera"]
        overlap = iou(np.array(box), project_boxes(eye[0], forward[0], up[0], self.objects["center"],
                                                   self.objects["half"], camera))
        best = int(overlap.argmax())
        if overlap[best] < MATCH_IOU:
            raise RuntimeError(f"{source}: box frames no city object (best IoU {overlap[best]:.2f} < {MATCH_IOU})")
        row = [*self.objects["center"][best], float(self.objects["index"][best]), *self.objects["half"][best],
               float(self.env.layout_classes.index(self.objects["class"][best]))]
        self.retarget = torch.tensor([row], device=self.device, dtype=torch.float32)
        self.env.retarget(self.retarget)
        self.set_goal(crop, f"{source}: {self.objects['asset'][best]}", box=box, match_iou=round(float(overlap[best]), 3))

    def instruct(self, text):
        """With a spatial-memory policy, the instruction starts the Gemma agent (agent.py), which plans
        over the memory; otherwise Gemma boxes the instructed object in the current frame once."""
        if self.gemma is None:
            raise RuntimeError("start live.py with --gemma to use text instructions")
        if self.spatial is not None:
            self.agent = Agent(text, self.gemma, self.policy.commands > 1)
            return
        model, processor = self.gemma
        box, answer, latency = detect(model, processor, Image.fromarray(self.camera_frame()), text)
        if box is None:
            raise RuntimeError(f"Gemma does not see \"{text}\" in this view ({latency:.1f} s)")
        self.box_goal(box, f"\"{text}\" ({latency:.1f} s)")

    def camera_frame(self):
        return (self.env.obs[0].clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()

    def explore(self):
        """Explore command: no goal; the memory keeps what was seen."""
        self.command.fill_(EXPLORE_COMMAND)
        self.goal = torch.zeros_like(self.goal)
        self.hidden.zero_()
        with self.lock:
            self.goal_info = {**self.goal_info, "source": "explore", "revision": self.goal_revision}

    def think(self, skill_ended):
        """Agent planning step (simulated time waits while Gemma answers)."""
        agent = self.agent
        skill = "explore" if int(self.command[0]) == EXPLORE_COMMAND else self.goal_info.get("source", "goal")
        summary = summarize(self.spatial, self.tracks, agent.elapsed(self.steps), skill)
        action, text, latency = agent.plan(self.steps, Image.fromarray(self.camera_frame()), summary)
        note = f"{latency:.1f} s"
        if action is None:
            note += f", unparsed: {text[:120]}"
        elif action["action"] == "explore":
            self.explore()
        elif action["action"] == "go":
            try:
                self.command.fill_(GOAL_COMMAND)
                self.box_goal(action["box"], f"agent: {action['label']}")
            except RuntimeError as error:
                note += f", {error}"
        with self.lock:
            self.agent_info = {"instruction": agent.instruction, "action": action, "note": note,
                               "answer": agent.answer, "summary": summary,
                               "history": [h["action"] for h in agent.history][-8:]}

    def handle(self, request):
        kind, payload = request
        try:
            if kind == "box":
                self.box_goal(payload, "your box")
            elif kind == "instruct":
                self.instruct(payload)
            elif kind == "new":
                self.new_episode(int.from_bytes(os.urandom(7), "little"))
            elif kind == "pause":
                self.paused = not self.paused
            elif kind == "catalog":
                self.catalog_goal()
            self.error = None
        except RuntimeError as error:
            self.error = str(error)

    # --- control loop -------------------------------------------------------------------------
    @torch.no_grad()
    def tick(self):
        cls, tokens = self.features.observe(self.env)
        if self.paused:
            # The memory integrates motion, so it only advances while flying.
            _, logits = self.policy.attention(self.goal, cls, tokens)
        else:
            memory = self.remember(cls, tokens) if self.spatial is not None else None
            inputs, logits = self.policy.actor_inputs_and_features(cls, tokens, self.goal, self.env.proprio,
                                                                   self.env.privileged, memory, self.command)
        probs = logits.softmax(-1)[0]
        frame = jpeg(self.env.obs[0])
        privileged = self.env.privileged[0].cpu().numpy()
        diagnostics = self.env.diagnostics[0].cpu().numpy()
        peak = int(probs[:-1].argmax())
        in_view = float(1 - probs[-1])
        if self.paused:
            self.publish(frame, privileged, diagnostics, peak, in_view, None)
            return
        mean, self.hidden = self.policy.step(inputs, self.hidden, self.starts)
        action = deterministic_action(mean).contiguous()
        self.env.step(action)
        self.steps += 1
        self.starts.copy_(self.env.done)
        outcome = None
        if bool(self.env.success[0] > 0.5):
            outcome = "success"   # free flight: reported once, the drone keeps holding there
        if bool(self.env.done[0] > 0.5):
            outcome = "collision" if bool(self.env.collision[0] > 0.5) else "left the city"
        self.publish(frame, privileged, diagnostics, peak, in_view, outcome, action[0].tolist())
        if bool(self.env.done[0] > 0.5):
            # Publish the pre-action frame with its own goal before adopting the reset episode.
            self.agent = None
            self.command.fill_(GOAL_COMMAND)
            self.catalog_goal()
        elif self.agent is not None and self.agent.due(self.steps, outcome == "success"):
            self.think(outcome == "success")

    def remember(self, cls, tokens):
        """Spatial-memory tick; the map image for the viewer is refreshed every MEMORY_IMAGE_EVERY ticks.
        Its placement on the overhead map uses the simulator pose at the memory's reset (display only)."""
        if bool(self.starts[0] > 0.5):
            d = self.env.diagnostics[0].cpu().numpy()
            q = d[6:10]
            forward = -np.array([2 * (q[0] * q[2] + q[3] * q[1]), 1 - 2 * (q[0] ** 2 + q[1] ** 2)])
            self.memory_info = {"origin": [float(d[0]), float(d[2])],
                                "forward": (forward / max(np.linalg.norm(forward), 1e-6)).round(5).tolist(),
                                "cell_m": CELL, "grid": GRID, "revision": 0}
        memory = observe_memory(self.policy, self.spatial, cls, tokens, self.goal, self.env.proprio, self.starts,
                                (self.command == GOAL_COMMAND).float())
        self.tracks.reset(self.starts)
        ranges = (self.spatial.points - self.spatial.position[:, None]).norm(dim=-1)
        cars = (self.spatial.classes[..., VEHICLE] >= CLASS_THRESHOLD) & (self.spatial.depth > 0)
        self.tracks.update(self.spatial.points, ranges, cars)
        if self.frame_id % MEMORY_IMAGE_EVERY == 0:
            evidence, seen = self.spatial.map[0].clamp(0, 1).cpu().numpy()
            rgba = np.zeros((GRID, GRID, 4), dtype=np.uint8)
            rgba[..., :3] = np.where(evidence[..., None] > 0.05, [255, 190, 40], [70, 120, 190])
            rgba[..., 3] = np.round(np.maximum(evidence * 230, seen * 70)).astype(np.uint8)
            buffer = io.BytesIO()
            Image.fromarray(rgba, "RGBA").save(buffer, "PNG")
            with self.lock:
                self.memory_image = buffer.getvalue()
                self.memory_info = {**self.memory_info, "revision": self.frame_id}
        return memory

    def publish(self, frame, privileged, diagnostics, peak, in_view, outcome, action=None):
        p, c, h = diagnostics[0:3], diagnostics[3:6], diagnostics[24:27]
        distance = float(np.linalg.norm(np.maximum(np.abs(p - c) - h, 0)))
        q = diagnostics[6:10]
        forward = np.array([2 * (q[0] * q[2] + q[3] * q[1]), 0.0, 1 - 2 * (q[0] ** 2 + q[1] ** 2)]) * -1
        visible = bool(privileged[14] > 0.5)
        rows, columns = self.env.height // 16, self.env.width // 16
        px = 2 * (peak % columns + 0.5) / columns - 1
        py = 1 - 2 * (peak // columns + 0.5) / rows
        x0, y0, x1, y1 = privileged[25:29]
        on_target = bool(x0 <= px <= x1 and y0 <= py <= y1) if visible and in_view > 0.5 else None
        with self.lock:
            self.frame, self.frame_id = frame, self.frame_id + 1
            if outcome is not None:
                self.last_outcome = outcome
            self.status = {
                "frame": self.frame_id, "paused": self.paused, "error": self.error,
                "position": p.round(2).tolist(), "heading": [float(forward[0]), float(forward[2])],
                "target": {"center": c.round(2).tolist(), "half": h.round(2).tolist(),
                           **self.object_info[int(diagnostics[27])],
                           "class": self.env.layout_classes[int(diagnostics[14])], "visible": visible,
                           "in_frustum": bool(privileged[20] > 0.5),
                           "box": privileged[25:29].round(3).tolist()},
                "distance_m": round(distance, 2), "speed_mps": round(float(np.linalg.norm(diagnostics[10:13])), 2),
                "altitude_m": round(float(p[1]), 2), "hold": round(float(privileged[24]), 2),
                "clock_s": round(float(diagnostics[17]) / self.env.control_hz, 1),
                "saliency": {"peak": peak, "in_view": round(in_view, 3),
                             "grid": [rows, columns], "on_target": on_target},
                "action": None if action is None else [round(a, 2) for a in action],
                "goal": {**self.goal_info, "age_s": round(time.monotonic() - self.goal_started, 1)},
                "checkpoint": self.checkpoint, "last_outcome": self.last_outcome,
                "gemma": self.gemma is not None, "memory": self.memory_info, "agent": self.agent_info,
            }

    def run(self):
        period = 1.0 / self.hz
        with self.env.execution():
            while True:
                started = time.monotonic()
                while not self.requests.empty():
                    self.handle(self.requests.get())
                self.tick()
                time.sleep(max(0.0, period - (time.monotonic() - started)))


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "LiveDrone/1.0"

    def log_message(self, *_):
        pass

    def send(self, status, body, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        pilot = self.server.pilot
        path = urlsplit(self.path).path
        if path in PAGES:
            name, content_type = PAGES[path]
            return self.send(200, (PAGE / name).read_bytes(), content_type)
        if path == "/state":
            with pilot.lock:
                body = json.dumps(pilot.status).encode()
            return self.send(200, body, "application/json")
        if path == "/map":
            return self.send(200, json.dumps(pilot.map).encode(), "application/json")
        if path == "/goal.jpg":
            with pilot.lock:
                body = pilot.goal_image
            return self.send(200, body, "image/jpeg")
        if path == "/memory.png":
            with pilot.lock:
                body = pilot.memory_image
            return self.send(200 if body else 404, body or b"no spatial memory\n", "image/png" if body else "text/plain")
        if path == "/stream":
            return self.stream(pilot)
        self.send(404, b"not found\n", "text/plain")

    def stream(self, pilot):
        """MJPEG: one part per published frame."""
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        last = -1
        try:
            while True:
                with pilot.lock:
                    frame, frame_id = pilot.frame, pilot.frame_id
                if frame_id != last and frame:
                    last = frame_id
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                     + str(len(frame)).encode() + b"\r\n\r\n" + frame + b"\r\n")
                    self.wfile.flush()
                time.sleep(0.01)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self):
        pilot = self.server.pilot
        path = urlsplit(self.path).path
        length = int(self.headers.get("Content-Length", "0"))
        try:
            body = json.loads(self.rfile.read(length) or b"{}") if 0 <= length <= 4096 else None
            if path == "/box":
                box = [float(v) for v in body["box"]]
                if len(box) != 4 or not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1):
                    raise ValueError
                pilot.requests.put(("box", box))
            elif path == "/instruct":
                text = str(body["instruction"]).strip()
                if not 0 < len(text) <= 300:
                    raise ValueError
                pilot.requests.put(("instruct", text))
            elif path in ("/new", "/pause", "/catalog"):
                pilot.requests.put((path[1:], None))
            else:
                return self.send(404, b"not found\n", "text/plain")
        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            return self.send(400, b"bad request\n", "text/plain")
        self.send(202, b"{}", "application/json")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "runs/drone9-mean-patch-seed0/best.pt")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8790)
    parser.add_argument("--hz", type=float, default=30.0, help="control rate of the live loop (30 = real time)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--appearance", type=float, default=1.0)
    parser.add_argument("--gemma", action="store_true", help="load Gemma 4 for typed instructions (~10 GB GPU)")
    parser.add_argument("--model", default="google/gemma-4-E4B-it")
    args = parser.parse_args()
    pilot = Pilot(args)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    server.pilot = pilot
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"Live drone: http://{args.host}:{args.port}/", flush=True)
    pilot.run()


if __name__ == "__main__":
    main()
