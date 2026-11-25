"""Local progress dashboard server (stdlib only). Run via `make progress`."""

from __future__ import annotations

import argparse
import json
import mimetypes
import socket
import sys
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from common.course import load_modules  # noqa: E402
from common.progress import compute_state, render_text  # noqa: E402
from common.video import load_video, mark_watched  # noqa: E402

DASHBOARD = REPO / "dashboard"
HOST = "127.0.0.1"


def find_free_port(start: int = 8765, tries: int = 20) -> int:
    for port in range(start, start + tries):
        with socket.socket() as s:
            # Same option the server uses, so a port with only closed (TIME_WAIT)
            # connections counts as free. A live listener still makes bind fail.
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((HOST, port))
            except OSError:
                continue
            return port
    raise RuntimeError(f"no free port in {start}-{start + tries - 1}")


def _handler(root: Path):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # keep the terminal quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _static(self, rel: str) -> None:
            base = DASHBOARD.resolve()
            target = (base / rel).resolve()
            if not target.is_relative_to(base) or not target.is_file():
                return self._send(404, b"not found", "text/plain")
            ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            if ctype.startswith("text/") or ctype.endswith("javascript"):
                ctype += "; charset=utf-8"
            self._send(200, target.read_bytes(), ctype)

        def _module(self, mid: str):
            modules, _ = load_modules(root)
            return next((m for m in modules if m.id == mid), None)

        def _video(self, rest: str) -> None:
            mid, _, rel = rest.partition("/")
            module = self._module(mid)
            if module is None:
                return self._send(404, b"not found", "text/plain")
            folder = module.path
            if rel in ("", "index.html"):
                return self._static("video.html")
            if rel == "script.json":
                video, problems = load_video(folder)
                if video is None:
                    return self._send(404, b"not found", "text/plain")
                lesson = next((s["command"] for s in module.steps if s["kind"] == "lesson"), "")
                body = json.dumps(
                    {**video, "module": mid, "problems": problems, "next_command": lesson}
                ).encode()
                return self._send(200, body, "application/json; charset=utf-8")
            allowed = {"scenes.js": "text/javascript; charset=utf-8"}
            base = (folder / "video").resolve()
            if rel.startswith("audio/") and rel.endswith(".mp3"):
                base = (folder / "video" / "audio").resolve()
                rel, ctype = rel[len("audio/") :], "audio/mpeg"
            elif rel in allowed:
                ctype = allowed[rel]
            else:
                return self._send(404, b"not found", "text/plain")
            target = (base / rel).resolve()
            if not target.is_relative_to(base) or not target.is_file():
                return self._send(404, b"not found", "text/plain")
            self._send(200, target.read_bytes(), ctype)

        def do_POST(self):
            path = urllib.parse.urlsplit(self.path).path
            if path != "/api/watched":
                return self._send(404, b"not found", "text/plain")
            try:
                length = min(int(self.headers.get("Content-Length") or 0), 10_000)
                mid = json.loads(self.rfile.read(length)).get("module")
            except (ValueError, AttributeError):
                mid = None
            if not isinstance(mid, str) or self._module(mid) is None:
                return self._send(400, b"unknown module", "text/plain")
            mark_watched(root, mid)
            self.send_response(204)
            self.end_headers()

        def do_GET(self):
            path = urllib.parse.unquote(urllib.parse.urlsplit(self.path).path)
            if path == "/api/state":
                try:
                    state = compute_state(root)
                    body = json.dumps(state, ensure_ascii=False, allow_nan=False).encode()
                except Exception as e:  # never drop the connection: tell the page what broke
                    err = json.dumps({"error": f"{type(e).__name__}: {e}"}).encode()
                    return self._send(500, err, "application/json; charset=utf-8")
                return self._send(200, body, "application/json; charset=utf-8")
            if path in ("/", "/index.html"):
                return self._static("index.html")
            if path.startswith("/static/"):
                return self._static(path[len("/static/") :])
            if path.startswith("/video/"):
                return self._video(path[len("/video/") :])
            self._send(404, b"not found", "text/plain")

    return Handler


def make_server(root: Path, port: int) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((HOST, port), _handler(Path(root)))


def main() -> int:
    ap = argparse.ArgumentParser(description="Course progress dashboard")
    ap.add_argument("--text", action="store_true", help="print progress to the terminal")
    ap.add_argument("--no-open", action="store_true", help="don't open a browser")
    ap.add_argument("--root", type=Path, default=REPO, help="course root (default: this repo)")
    args = ap.parse_args()

    if args.text:
        print(render_text(compute_state(args.root)))
        return 0

    port = find_free_port()
    srv = make_server(args.root, port)
    url = f"http://{HOST}:{port}"
    print(f"Dashboard: {url}   (Ctrl-C to stop)", flush=True)
    if not args.no_open:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
