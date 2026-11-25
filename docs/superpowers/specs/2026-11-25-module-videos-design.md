# Module videos (narrated animated explainers) — design

Approved in chat on 2026-11-25 (format, Kokoro voice, the `media` dependency group, the Kokoro-82M
download). First module: M5.

## Why
The learner wants every module to open with a short animated, narrated video that explains the idea
before the lesson, exercises and run: "first people can go through the video and then rest of the
content". It must make the module easier for anyone, so it is plain-spoken, visual, and accurate.

## Approaches
1. **Animated page in the browser + Kokoro narration as committed MP3s (chosen).** Plays like a video
   (play/pause, scrubber, captions, voice), a few KB of code plus ~1.5 MB of audio per module, no
   install for learners, every number checkable against real runs, easy to edit.
2. MP4 rendered with manim + ffmpeg: shareable anywhere, but heavy installs, 10–30 MB per module, and
   every edit is a re-render.
3. Both (record the page to MP4 later): kept as a possible follow-up.

## Design
- **Per module:** `video/script.toml` (`title`, `[[scene]]` with `id`, `visual`, `narration`),
  `video/scenes.js` (one draw function per `visual`, a pure function of time so scrubbing works),
  `video/audio/<id>.mp3`, `video/timings.json` (`{id: {seconds, sha256}}`, sha of voice + text).
- **Narration:** `make narration M=05` runs `scripts/narrate.py` with Kokoro-82M (`af_heart`) via
  `mlx-audio` (dependency group `media`, not installed by default), MP3 48 kbps mono. Unchanged
  scenes are skipped. Needs Homebrew `espeak-ng` (misaki's bundled copy has a broken data path).
- **Player:** `dashboard/video.html` + `video.js` + `video.css`, served at `/video/<module id>/`.
  Plays the scenes in order, each with its clip; captions show the current sentence; a scrubber
  over the whole video; keyboard: space, arrows. When the last scene ends it POSTs
  `/api/watched` and the step turns done.
- **Server:** `GET /video/<mid>/` (player), `/video/<mid>/script.json` (script + timings),
  `/video/<mid>/scenes.js`, `/video/<mid>/audio/<file>.mp3` (resolved inside the module's
  `video/audio`, 404 otherwise); `POST /api/watched` `{"module": "m05"}` appends to
  `.progress/watched.jsonl` (known module ids only).
- **Guide:** new step kind `video` (status done once watched); the dashboard shows a "Play the
  video" button instead of a command for it. It is each module's first step.
- **Tests:** every scene has an MP3 and a timing whose sha matches its text; every `visual` is drawn
  by `scenes.js`; the video is step 1; server routes and path safety; the watched record.
- **M5 scenes (10, ~4 min):** what fine-tuning changes · the original model's formal rewrite · prompt
  + answer as one sequence · loss masking · padding a batch · where the memory goes · training (loss
  falling) · before/after rewrites · forgetting · your turn. Numbers from the measured runs.
