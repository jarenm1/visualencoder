#!/usr/bin/env python3
"""Serve the offline city replay UI and only its selected recording files."""

import argparse
import json
import os
from pathlib import Path
import re
import stat
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
BYTE_RANGE = re.compile(r"bytes=(\d*)-(\d*)\Z")


def recording(data):
    root = data.resolve(strict=True)
    manifest_path = root / "manifest.json"
    if not manifest_path.resolve(strict=True).is_relative_to(root):
        raise ValueError("manifest.json must stay inside the selected data directory")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("format") != "sim-city-replay-v2":
        raise ValueError("expected a sim-city-replay-v2 manifest")
    environments = manifest.get("environments")
    if not isinstance(environments, list) or not environments:
        raise ValueError("manifest contains no recorded environments")
    files = {}
    # Non-visual policies (privileged state, hint-only) have no encoder output.
    kinds = [("video", VIDEO_NAME, "video/mp4")]
    if manifest.get("encoder_map") is not None:
        kinds.append(("encoder_map", MAP_NAME, "application/octet-stream"))
    if manifest.get("saliency") is not None:
        kinds.append(("saliency", SALIENCY_NAME, "application/octet-stream"))
    for environment in environments:
        for key, pattern, content_type in kinds:
            name = environment.get(key)
            if not isinstance(name, str) or not pattern.fullmatch(name):
                raise ValueError(f"invalid replay {key} filename: {name!r}")
            path = (root / name).resolve(strict=True)
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError(f"{key} must be a file inside the selected data directory: {name}")
            if path.stat().st_size == 0:
                raise ValueError(f"{key} is empty: {name}")
            files["/replay/" + name] = (root / name, content_type)
    if not files:
        raise ValueError("manifest lists no replay files")
    return root, manifest_bytes, files


class ReplayServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, root, manifest, files):
        self.data_root = root
        self.manifest = manifest
        self.files = files
        super().__init__(address, ReplayHandler)


class ReplayHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "CityReplay/1.0"
    sys_version = ""

    def do_HEAD(self):
        self.serve(head=True)

    def do_GET(self):
        self.serve(head=False)

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

        if path == "/replay/manifest.json":
            body = self.server.manifest
            self.headers_for(200, len(body), "application/json; charset=utf-8")
            if not head:
                self.wfile.write(body)
            return
        if path in UI_FILES:
            name, content_type = UI_FILES[path]
            file_path = UI_ROOT / name
            root = UI_ROOT
        elif path in self.server.files:
            file_path, content_type = self.server.files[path]
            root = self.server.data_root
        else:
            self.fail(404, "Replay resource not found", head)
            return

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
    args = parser.parse_args()
    try:
        root, manifest, files = recording(args.data)
        server = ReplayServer((args.host, args.port), root, manifest, files)
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.exit(1, f"Replay server: {error}\n")
    print(f"City replay: http://{args.host}:{args.port}/", flush=True)
    count = len(json.loads(manifest)["environments"])
    print(f"Serving {count} recorded environments ({len(files)} files) from {root}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
