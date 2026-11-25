"""Make (or refresh) a module video's narration with Kokoro. For course authors.

    make narration M=05            # only scenes whose text changed
    make narration M=05 FORCE=1    # every scene

Reads modules/mNN_*/video/script.toml, writes video/audio/<scene id>.mp3 and video/timings.json.
Needs the media dependency group (uv sync --group media), Homebrew's espeak-ng
(brew install espeak-ng) and ffmpeg. The first run downloads Kokoro-82M (about 330 MB).
Learners never run this: the audio is committed.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from common.video import VOICE, load_video, scene_sha  # noqa: E402

KOKORO = "mlx-community/Kokoro-82M-bf16"
ESPEAK = Path("/opt/homebrew/lib/libespeak-ng.dylib")
SPEED = 0.95


def load_kokoro():
    try:
        import misaki.espeak  # noqa: F401  (points phonemizer at its bundled espeak first)
        from mlx_audio.tts.utils import load_model
        from phonemizer.backend.espeak.wrapper import EspeakWrapper
    except ImportError as e:
        sys.exit(f"Missing the media tools ({e}). Run: uv sync --group media")
    if not ESPEAK.exists():
        sys.exit("Needs Homebrew's espeak-ng: brew install espeak-ng")
    # misaki's bundled espeak-ng looks for its data in a path that only exists on its build
    # machine, so use Homebrew's library instead.
    EspeakWrapper.set_library(str(ESPEAK))
    return load_model(KOKORO)


def speak(model, text: str, out_mp3: Path) -> float:
    import numpy as np

    parts = [
        np.array(r.audio)
        for r in model.generate(text=text, voice=VOICE, speed=SPEED, lang_code="a")
    ]
    audio = np.concatenate(parts)
    pad = np.zeros(int(0.35 * model.sample_rate), dtype=audio.dtype)  # a breath after each scene
    audio = np.concatenate([audio, pad])
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "scene.wav"
        with wave.open(str(wav), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(model.sample_rate)
            w.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
        out_mp3.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(wav),
                "-ac",
                "1",
                "-b:a",
                "48k",
                str(out_mp3),
            ],
            check=True,
        )
    return round(len(audio) / model.sample_rate, 3)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("module", help="module number, e.g. 05")
    ap.add_argument("--force", action="store_true", help="remake every scene")
    args = ap.parse_args()
    if shutil.which("ffmpeg") is None:
        sys.exit("Needs ffmpeg: brew install ffmpeg")
    folders = sorted(REPO.glob(f"modules/m{args.module}_*"))
    if not folders:
        sys.exit(f"No module m{args.module}")
    folder = folders[0]
    video, _ = load_video(folder)
    if video is None:
        sys.exit(f"{folder.name} has no video/script.toml")
    timings_file = folder / "video" / "timings.json"
    timings = json.loads(timings_file.read_text()) if timings_file.exists() else {}
    model = None
    for scene in video["scenes"]:
        sid, text = scene["id"], scene["narration"]
        mp3 = folder / "video" / "audio" / f"{sid}.mp3"
        sha = scene_sha(text)
        if not args.force and mp3.exists() and timings.get(sid, {}).get("sha256") == sha:
            print(f"  {sid}: unchanged")
            continue
        model = model or load_kokoro()
        seconds = speak(model, text, mp3)
        timings[sid] = {"seconds": seconds, "sha256": sha}
        print(f"  {sid}: {seconds:.1f}s")
    ids = {s["id"] for s in video["scenes"]}
    timings = {k: v for k, v in timings.items() if k in ids}
    for stale in (folder / "video" / "audio").glob("*.mp3"):
        if stale.stem not in ids:
            stale.unlink()
    timings_file.write_text(json.dumps(timings, indent=2) + "\n")
    total = sum(t["seconds"] for t in timings.values())
    print(f"{folder.name}: {len(timings)} scenes, {total // 60:.0f}:{total % 60:04.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
