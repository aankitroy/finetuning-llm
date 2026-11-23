# M4 · Training loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build module M4 so the learner writes a full training loop and trains a ~4.8M-parameter character GPT on Tiny Shakespeare.

**Architecture:** Same layout as M3 (`module.toml`, `hints.toml`, `starter/`, `solutions/`, `tests/`, a run script). The model is given in `model.py`; the six exercises are the loop's pieces; `train.py` wires them together and writes `outputs/m04/`.

**Tech Stack:** Python 3.11+, PyTorch (MPS), matplotlib, nbformat/nbclient, pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-11-24-m04-training-loop-design.md`

## Global Constraints
- Beginner-first copy: plain words, "In plain words:" docstrings, `# 1.` steps, a lesson-section pointer per exercise.
- Every code cell is followed by a `**What each line does**` markdown cell; terms bold where defined.
- `exercises.py` == `starter/exercises.py`, `answers.toml` == `starter/answers.toml`, `lesson.ipynb` == `starter/lesson.ipynb` (unrun).
- Tests and CI never download anything; `--tiny` runs on CPU in seconds.
- Targets: val loss < 1.70, real-word rate ≥ 0.60.

## Review Focus
- A learner's `train_step` that forgets to divide by the number of micro-batches: the accumulation test must catch it.
- `clip_grad_norm` scaling up gradients that are already small: must leave them untouched.
- `estimate_loss` leaving the model in eval mode (dropout off for the rest of training): must be caught.
- `lr_at` off by one at the warmup boundary or past `total_steps`: pinned by exact values.
- `train.py` given an exercise that raises: a plain message, not a traceback (as M3).

---

### Task 1: Data and model
**Files:** Create `modules/m04_training_loop/shakespeare.txt`, `model.py`; modify `.pre-commit-config.yaml` (exclude the text).
- [ ] Copy the downloaded file; check sha256 `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`.
- [ ] `TinyGPT` as in the spec; parameter count printed ≈ 4.8M.

### Task 2: Exercises (TDD against the solutions)
**Files:** `tests/conftest.py`, `tests/test_checkpoint.py`, `solutions/exercises.py`, `starter/exercises.py`, `exercises.py`.
- [ ] Write the six exercise tests (exact values for `lr_at`; accumulation equals one full batch; clipping matches `torch.nn.utils.clip_grad_norm_`; train mode restored).
- [ ] Run with `FTCOURSE_SOLUTIONS=1` against an empty solutions file: FAIL.
- [ ] Write the solutions: PASS. Starter stubs raise `NotImplementedError("Exercise N: name")`.

### Task 3: The run
**Files:** `train.py`, run tests in `test_checkpoint.py`, `tests/test_module_solutions.py`, `module.toml`.
- [ ] Run tests: target loss, real-word rate, full-size run, own code; `report_metric("final_loss")`, `report_metric("peak_mem_gb")`.
- [ ] `train.py --tiny` with solutions passes the checkpoint; the full run on MPS meets both targets.

### Task 4: Questions and hints
**Files:** `starter/answers.toml`, `answers.toml`, `solutions/answers.toml`, `tests/answer_hashes.json`, `hints.toml`.

### Task 5: Lesson
**Files:** `lesson.ipynb`, `starter/lesson.ipynb` (built by a scratchpad script).
- [ ] Runs top to bottom in a few minutes on CPU/MPS.

### Task 6: README, content tests, shared files
**Files:** `README.md`, `tests/test_m04_content.py`, `pyproject.toml` (if needed), `CHANGELOG.md`.
- [ ] Full suite, ruff, pre-commit green.
