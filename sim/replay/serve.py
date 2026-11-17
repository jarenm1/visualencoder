#!/usr/bin/env python3
"""Serve the offline city replay UI and only its selected recording files.

With --live, POST /api/instruct {"instruction": "..."} renders the start frames of a fresh
held-out batch (sim/python/gemma_goal.py frames), asks Gemma 4 to box the instructed object in
each frame (gemma_goal.py ground), records the trained policy flying to those boxes
(sim/python/replay.py --goal-boxes), and swaps the new recording in. GET /api/status reports
the pipeline state. One instruction runs at a time.
"""

import argparse
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlsplit


UI_ROOT = Path(__file__).resolve().parent
UI_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
}
VIDEO_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\.mp4\Z")
MAP_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\.encoder\Z")
SALIENCY_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\.saliency\Z")
GOAL_FILE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*-goal\.png\Z")
BYTE_RANGE = re.compile(r"bytes=(\d*)-(\d*)\Z")
RECORDING_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


def recording(data):
    """Validate a recording directory -> (name, manifest bytes tagged with "recording", {url: (path, type, root)})."""
    root = data.resolve(strict=True)
    name = root.name
    if not RECORDING_NAME.fullmatch(name):
        raise ValueError(f"recording directory name must match {RECORDING_NAME.pattern}: {name}")
    manifest_path = root / "manifest.json"
    if not manifest_path.resolve(strict=True).is_relative_to(root):
        raise ValueError("manifest.json must stay inside the selected data directory")
    manifest = json.loads(manifest_path.read_bytes())
    if manifest.get("format") != "sim-city-replay-v3":
        raise ValueError("expected a sim-city-replay-v3 manifest")
    environments = manifest.get("environments")
    if not isinstance(environments, list) or not environments:
        raise ValueError("manifest contains no recorded environments")
    files = {}
    kinds = [("video", VIDEO_NAME, "video/mp4"), ("goal", GOAL_FILE, "image/png")]
    # Non-visual policies (privileged state, hint-only) have no encoder output.
    if manifest.get("encoder_map") is not None:
        kinds.append(("encoder_map", MAP_NAME, "application/octet-stream"))
    if manifest.get("saliency") is not None:
        kinds.append(("saliency", SALIENCY_NAME, "application/octet-stream"))
    for environment in environments:
        for key, pattern, content_type in kinds:
            file_name = environment.get(key)
            if not isinstance(file_name, str) or not pattern.fullmatch(file_name):
                raise ValueError(f"invalid replay {key} filename: {file_name!r}")
            path = (root / file_name).resolve(strict=True)
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError(f"{key} must be a file inside the selected data directory: {file_name}")
            if path.stat().st_size == 0:
                raise ValueError(f"{key} is empty: {file_name}")
            # Per-recording URLs: a page holding an older manifest can never fetch another recording's clip.
            files[f"/replay/{name}/{file_name}"] = (root / file_name, content_type, root)
    if not files:
        raise ValueError("manifest lists no replay files")
    manifest["recording"] = name
    return name, json.dumps(manifest).encode("utf-8"), files


class ReplayServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, loaded, live=None):
        self.lock = threading.Lock()
        self.current = loaded[:2]
        self.files = dict(loaded[2])
        self.live = live
        self.status = {"live": live is not None, "state": "idle"}
        super().__init__(address, ReplayHandler)

    def install(self, loaded):
        """Serve a new recording as current; earlier recordings stay reachable under their own URLs."""
        with self.lock:
            self.current = loaded[:2]
            self.files.update(loaded[2])

    def report(self, **fields):
        with self.lock:
            self.status = {"live": True, **fields}


REPO = Path(__file__).resolve().parents[2]
PYTHON = REPO / ".venv/bin/python"
INSTRUCTION_LIMIT = 300
# Pipeline states during which a new instruction is refused.
WORKING = ("framing", "grounding", "flying")


def tail(text, lines=6):
    return "\n".join(text.strip().splitlines()[-lines:])


def sim_python(script, *args, timeout):
    """Run a repo script with the venv inside the sim nix shell (CUDA, libstdc++, Vulkan)."""
    result = subprocess.run(["nix", "develop", "path:sim", "--command", str(PYTHON), str(REPO / script), *args],
                            cwd=REPO, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(f"{Path(script).name} failed: " + tail(result.stderr or result.stdout))
    return result.stdout


def run_instruction(server, instruction):
    """Start frames -> Gemma boxes -> grounded policy recording -> swap. Runs on a worker thread."""
    live = server.live
    try:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        work, out = live["out"] / f"{stamp}-ground", live["out"] / stamp
        # Fresh scenes per instruction; repeating an instruction must not replay the same episodes.
        seed = int.from_bytes(os.urandom(7), "little")
        server.report(state="framing", instruction=instruction)
        sim_python("sim/python/gemma_goal.py", "frames", "--out", str(work), "--num-envs", str(live["num_envs"]),
                   "--seed", str(seed), timeout=900)
        server.report(state="grounding", instruction=instruction)
        sim_python("sim/python/gemma_goal.py", "ground", "--out", str(work), "--instruction", instruction,
                   "--model", live["model"], timeout=1800)
        grounded = json.loads((work / "boxes.json").read_text())
        if grounded["found"] == 0:
            raise RuntimeError(f"Gemma found nothing matching \"{instruction}\" in {grounded['num_envs']} start views")
        server.report(state="flying", instruction=instruction, found=grounded["found"], matched=grounded["matched"])
        sim_python("sim/python/replay.py", "--checkpoint", str(live["checkpoint"]), "--out", str(out),
                   "--num-envs", str(live["num_envs"]), "--split", "test", "--goal-boxes", str(work / "boxes.json"),
                   timeout=1800)
        loaded = recording(out)
        server.install(loaded)
        server.report(state="done", instruction=instruction, found=grounded["found"], matched=grounded["matched"],
                      recording=loaded[0], episodes=len(json.loads(loaded[1])["environments"]))
    except Exception as error:  # Reported to the page; the previous recording stays served.
        server.report(state="error", instruction=instruction, message=str(error))


class ReplayHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "CityReplay/1.0"
    sys_version = ""

    def do_HEAD(self):
        self.serve(head=True)

    def do_GET(self):
        self.serve(head=False)

    def json_reply(self, status, payload):
        body = (json.dumps(payload) + "\n").encode("utf-8")
        self.headers_for(status, len(body), "application/json; charset=utf-8")
        self.wfile.write(body)

    def do_POST(self):
        if urlsplit(self.path).path != "/api/instruct" or self.server.live is None:
            self.fail(404, "Replay resource not found")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 4096:
                raise ValueError
            instruction = str(json.loads(self.rfile.read(length))["instruction"]).strip()
            if not 0 < len(instruction) <= INSTRUCTION_LIMIT:
                raise ValueError
        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            self.fail(400, f"Expected JSON {{\"instruction\": text of 1-{INSTRUCTION_LIMIT} characters}}")
            return
        with self.server.lock:
            busy = self.server.status["state"] in WORKING
            if not busy:
                self.server.status = {"live": True, "state": "framing", "instruction": instruction}
        if busy:
            self.json_reply(409, {"error": "an instruction is already running"})
            return
        threading.Thread(target=run_instruction, args=(self.server, instruction), daemon=True).start()
        self.json_reply(202, {"state": "framing", "instruction": instruction})

    def headers_for(self, status, length, content_type, content_range=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self'; media-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
        self.send_header("Cache-Control", "no-store")
        if content_range is not None:
            self.send_header("Content-Range", content_range)
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()

    def fail(self, status, message, head=False, content_range=None):
        body = (message + "\n").encode("utf-8")
        self.headers_for(status, len(body), "text/plain; charset=utf-8", content_range)
        if not head:
            self.wfile.write(body)

    def serve(self, head):
        try:
            parsed = urlsplit(self.path)
            path = unquote(parsed.path, errors="strict")
            if parsed.scheme or parsed.netloc or "\\" in path or "\x00" in path:
                raise ValueError("invalid path")
            if any(part in (".", "..") for part in path.split("/")):
                raise ValueError("invalid path")
        except (ValueError, UnicodeError):
            self.fail(400, "Invalid request path", head)
            return

        if path == "/api/status":
            with self.server.lock:
                status = {**self.server.status, "current": self.server.current[0]}
            if head:
                self.headers_for(200, 0, "application/json; charset=utf-8")
            else:
                self.json_reply(200, status)
            return
        if path == "/replay/manifest.json":
            with self.server.lock:
                body = self.server.current[1]
            self.headers_for(200, len(body), "application/json; charset=utf-8")
            if not head:
                self.wfile.write(body)
            return
        if path in UI_FILES:
            name, content_type = UI_FILES[path]
            file_path = UI_ROOT / name
            root = UI_ROOT
        else:
            with self.server.lock:
                entry = self.server.files.get(path)
            if entry is None:
                self.fail(404, "Replay resource not found", head)
                return
            file_path, content_type, root = entry

        try:
            resolved = file_path.resolve(strict=True)
            if not resolved.is_relative_to(root):
                self.fail(404, "Replay resource not found", head)
                return
            stream = resolved.open("rb")
        except (OSError, ValueError):
            self.fail(404, "Replay resource unavailable", head)
            return
        with stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode):
                self.fail(404, "Replay resource not found", head)
                return
            size = info.st_size
            start, end = 0, size - 1
            status = 200
            ranges = self.headers.get_all("Range", [])
            if ranges:
                match = BYTE_RANGE.fullmatch(ranges[0].strip()) if len(ranges) == 1 else None
                try:
                    if match is None or not any(match.groups()) or size == 0:
                        raise ValueError("invalid byte range")
                    first, last = match.groups()
                    if first:
                        start = int(first)
                        end = min(int(last), size - 1) if last else size - 1
                        if start >= size or end < start:
                            raise ValueError("unsatisfiable byte range")
                    else:
                        suffix = int(last)
                        if suffix == 0:
                            raise ValueError("empty suffix")
                        start = max(0, size - suffix)
                    status = 206
                except ValueError:
                    self.fail(416, "Requested byte range is not satisfiable", head, f"bytes */{size}")
                    return
            length = max(0, end - start + 1)
            content_range = f"bytes {start}-{end}/{size}" if status == 206 else None
            self.headers_for(status, length, content_type, content_range)
            if head:
                return
            stream.seek(start)
            remaining = length
            try:
                while remaining:
                    block = stream.read(min(256 * 1024, remaining))
                    if not block:
                        self.close_connection = True
                        break
                    self.wfile.write(block)
                    remaining -= len(block)
            except (BrokenPipeError, ConnectionResetError):
                pass  # Browsers cancel previous ranges when the user seeks.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("runs/city-replay"), help="directory containing manifest.json and its MP4 clips")
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default: localhost only)")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--live", action="store_true", help="enable typed instructions (Gemma 4 + policy recording)")
    parser.add_argument("--checkpoint", type=Path, default=REPO / "runs/drone5-mean-seed0/best.pt")
    parser.add_argument("--live-out", type=Path, default=REPO / "runs/replay-live")
    parser.add_argument("--model", default="google/gemma-4-E4B-it")
    parser.add_argument("--live-envs", type=int, default=32, help="environments recorded per instruction")
    args = parser.parse_args()
    live = None
    if args.live:
        if not args.checkpoint.is_file():
            parser.exit(1, "Replay server: --live needs a trained --checkpoint\n")
        args.live_out.mkdir(parents=True, exist_ok=True)
        live = {"checkpoint": args.checkpoint.resolve(), "out": args.live_out.resolve(),
                "model": args.model, "num_envs": args.live_envs}
    try:
        loaded = recording(args.data)
        server = ReplayServer((args.host, args.port), loaded, live)
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.exit(1, f"Replay server: {error}\n")
    print(f"City replay: http://{args.host}:{args.port}/", flush=True)
    count = len(json.loads(loaded[1])["environments"])
    print(f"Serving recording {loaded[0]}: {count} environments ({len(loaded[2])} files)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
