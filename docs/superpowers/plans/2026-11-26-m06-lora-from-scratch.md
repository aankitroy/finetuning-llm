# M6 LoRA from scratch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build module M6: the learner writes LoRA by hand and tunes Qwen2.5-0.5B with under 1% of its weights.

**Architecture:** Same module layout as M5 (module.toml, hints.toml, starter/, solutions/, tests/, lesson, README, video/). `train.py` loads M5's train.py and exercises as libraries.

**Tech Stack:** PyTorch 2.15, transformers 5.23, nbformat/nbclient, Kokoro via mlx-audio.

**Spec:** docs/superpowers/specs/2026-11-26-m06-lora-from-scratch-design.md

## Global Constraints
- Never touch the learner's real `modules/*/exercises.py`, `answers.toml`, `lesson.ipynb`, `outputs/` or `.progress`. Real-size measurements run with `FTCOURSE_OUTPUTS` set to a temp dir.
- Beginner style: plain words, numbered steps, "What each line does" after every code cell.
- Checkpoint thresholds are constants in test_checkpoint.py and stated in the README.

## Review Focus
1. A learner's LoRALinear that creates A/B on the CPU while the base is on MPS: train.py must say so plainly.
2. inject_lora called twice (LoRA inside LoRA): the second call must not wrap a LoRALinear's `base`.
3. The learner's M5 run missing: fall back to the reference M5 loss, and say which number was used.
4. merge_lora returning a layer that still shares storage with the trainable B/A: outputs must still match after merging.
5. `--targets all` on the tiny model and the real one: names match Qwen2's (gate_proj, up_proj, down_proj).

---

### Task 1: Solutions, tests and train.py
- Create `modules/m06_lora_from_scratch/solutions/exercises.py` (5 exercises per spec), `tests/conftest.py`, `tests/test_checkpoint.py`, `train.py`.
- Exercise tests fail against the starter's NotImplementedError and pass with `FTCOURSE_SOLUTIONS=1`.
- `train.py --tiny` runs in seconds; add `("m06_lora_from_scratch", ["train.py", "--tiny"])` to `tests/test_module_solutions.py`.

### Task 2: Measure on the real model
- Run train.py full size with solutions into a temp outputs dir for lr ∈ {1e-4, 2e-4, 5e-4}, then r ∈ {2, 8, 32} and `--targets all`. Pick defaults; set `MARGIN` from the numbers. Record in the spec.

### Task 3: Starter, answers, hints, module.toml, README
- Starter exercises with "In plain words" + numbered steps; six questions (hashes via `make answer-hashes M=06`); hints; module.toml steps (video, lesson, exercises, answers, train, checkpoint); README with glossary and syntax table.
- `tests/test_m06_content.py` modelled on test_m05_content.py.

### Task 4: Lesson notebook
- Builder script in the scratchpad; sections 0–8; outputs from real runs drive every claim; runs top to bottom (content test).

### Task 5: Video
- `video/script.toml`, `scenes.js`, narration with `make narration M=06`; reviewed with final-frame screenshots.

### Task 6: Full suite, lint, review, PR
- `uv run pytest`, ruff; fresh reviewer; push `feat/m06-lora` and open PR #11 based on `feat/videos-m1-m4`.
