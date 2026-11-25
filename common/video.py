"""Module videos: a narrated, animated explainer at the start of each module.

Each module that has one keeps it in ``video/``:

- ``script.toml``: the title and the scenes, each with an ``id``, a ``visual`` (the name of a draw
  function in ``scenes.js``) and the ``narration`` text.
- ``scenes.js``: the animations.
- ``audio/<id>.mp3`` and ``timings.json``: the narration, made by ``make narration M=NN``.
"""

from __future__ import annotations

import hashlib
import json
import time
import tomllib
from pathlib import Path

VOICE = "af_heart"
WATCHED_PATH = Path(".progress/watched.jsonl")


def scene_sha(narration: str, voice: str = VOICE) -> str:
    """Fingerprint of what a clip says, so stale audio can be spotted."""
    return hashlib.sha256(f"{voice}\n{narration}".encode()).hexdigest()


def load_video(module_path: Path) -> tuple[dict | None, list[str]]:
    """The module's video, ready for the player, plus any problems (missing or stale audio)."""
    folder = Path(module_path) / "video"
    script_file = folder / "script.toml"
    if not script_file.exists():
        return None, []
    problems: list[str] = []
    try:
        script = tomllib.loads(script_file.read_text())
    except tomllib.TOMLDecodeError as e:
        return None, [f"video/script.toml is invalid ({e})"]
    try:
        timings = json.loads((folder / "timings.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        timings = {}
        problems.append("video/timings.json is missing: run make narration")
    scenes, seen = [], set()
    for sc in script.get("scene", []):
        sid = str(sc.get("id", ""))
        if not sid or sid in seen:
            problems.append(f"scene id {sid!r} is empty or repeated")
            continue
        seen.add(sid)
        narration = str(sc.get("narration", "")).strip()
        timing = timings.get(sid, {})
        if not (folder / "audio" / f"{sid}.mp3").exists() or not timing:
            problems.append(f"scene {sid}: no audio yet (run make narration)")
        elif timing.get("sha256") != scene_sha(narration):
            problems.append(f"scene {sid}: narration changed since the audio was made")
        scenes.append(
            {
                "id": sid,
                "visual": str(sc.get("visual", "")),
                "narration": narration,
                "seconds": float(timing.get("seconds", 0.0)),
                "audio": f"audio/{sid}.mp3",
            }
        )
    video = {
        "title": str(script.get("title", "")),
        "scenes": scenes,
        "total_seconds": round(sum(s["seconds"] for s in scenes), 2),
    }
    return video, problems


def mark_watched(root: Path, module_id: str) -> None:
    path = Path(root) / WATCHED_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(
            json.dumps({"module": module_id, "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z")}) + "\n"
        )


def watched_modules(root: Path) -> set[str]:
    path = Path(root) / WATCHED_PATH
    if not path.exists():
        return set()
    out = set()
    for line in path.read_text().splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict) and isinstance(rec.get("module"), str):
            out.add(rec["module"])
    return out
