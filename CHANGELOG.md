# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Module videos: a narrated, animated explainer as a module's first step, played from the dashboard (`/video/<module>/`) with captions, scene-by-scene scrubbing and links to a moment (`#scene=3&t=10`). The voice is Kokoro-82M, made once by course authors with `make narration M=NN` (new `media` dependency group); learners need nothing extra. M1 to M5 each have one (3:44 to 4:28).
- M5 · Your first real fine-tune: a full fine-tune of Qwen2.5-0.5B-Instruct (every weight) on 500 original jargon → plain English pairs, with your own loop and with HF `Trainer`. The learner writes loss masking, padding/collation, the shifted masked loss, an optimizer-memory meter, chrF and a buzzword counter. The checkpoint needs the held-out loss to fall by 0.4 without losing meaning (chrF) or adding jargon; a forgetting check on 10 ordinary questions is reported. Peaks at about 24 GB.
- `accelerate` in the `course` dependency group (needed by `Trainer`).
- M4 · The training loop, from scratch: a 4.8M-parameter character-level GPT trained from random weights on Tiny Shakespeare (bundled, public domain). The learner writes the loop: batches, the causal LM loss, warmup + cosine schedule, gradient clipping, gradient accumulation and evaluation. The checkpoint needs validation loss below 1.70 and a sample that is at least 80% real words; the run takes about 2 minutes on Apple Silicon. Tests use a tiny model on the CPU.
- M3 · Tokenizers & chat templates: bytes and UTF-8, a byte-level BPE tokenizer built from scratch, a look inside SmolLM2-Instruct's tokenizer and special tokens, and a hand-written ChatML formatter that must match the real chat template on five tricky conversations. Tests are plain Python and need no download.
- Optional Part G (beyond text: vision-language, speech-to-text, image generation) in the course plan.
- M2 · How an LLM actually works: load SmolLM2-135M, look inside it, attention by hand, and a hand-written generation loop (greedy, temperature, top-p) that must match `model.generate()` token for token. Tests use a tiny random model, so they need no download.
- `transformers` in the `course` dependency group.
- Step-by-step module guide on the dashboard (and in `make progress TEXT=1`): each step's state is detected from your files and your last check, with per-exercise results, failure messages and hint commands.
- Checkpoint runs record which tests failed and why.
- Understanding checks: each module's `answers.toml` is checked against stored hashes.
- `make hint` and `make solution` for one exercise at a time, logged and shown on the dashboard.
- `make reset` / `make reset M=NN` to start over, with a typed confirmation.
- M1 · Tensors, autograd & PyTorch essentials: lesson notebook, six exercises, a two-moons training script and a checkpoint at 95% accuracy.
- Course dependencies (PyTorch, NumPy, matplotlib, JupyterLab) in a `course` dependency group.
- Course structure for 17 modules (`COURSE.md`) with a manifest per module.
- Progress dashboard (`make progress`): next-step guidance, the course route, and results charts over time. Updates live as you pass checkpoints.
- Terminal progress view (`make progress TEXT=1`).
- Strict checkpoint recording through `make check M=NN`. Runs against the reference solutions never count.
- M0 · Environment & Apple Silicon setup (`make doctor`).
- Open-source project setup: licences, contributing guide, code of conduct, issue and PR templates, and CI.

### Changed
- M1 rewritten for complete beginners: concept-first lesson with a "What each line does" box after every code cell, a worked example, step-by-step comments in every exercise, a sixth question on logits, and a glossary and syntax table in the README.
- `make reset` now also restores a module's lesson notebook to its unrun state, for modules that ship `starter/lesson.ipynb` (M1 to M5).
