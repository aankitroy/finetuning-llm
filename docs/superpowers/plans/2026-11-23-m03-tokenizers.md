# M3 · Tokenizers & chat templates Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans. Steps use `- [ ]`.

**Goal:** build M3 in the M1/M2 beginner style: toy byte-level BPE + ChatML by hand, checked offline.
**Spec:** `docs/superpowers/specs/2026-11-23-m03-tokenizers-design.md`

## Global Constraints
- Same layout as M2: `module.toml` steps, `hints.toml` with `test`, `starter/`, `solutions/`,
  `tests/answer_hashes.json`, `starter/lesson.ipynb`.
- Tests and CI need no download; pure Python except the run script and lesson.
- Every stub has "In plain words:", "# 1. " steps and points to a lesson section.

## Review Focus
- BPE tie-breaking: learner and reference must agree (most frequent, ties → first seen).
- encode must apply merges by rank, not left-to-right greedily.
- decode of a partial UTF-8 sequence must not crash (`errors="replace"`).
- default system prompt: only when the first message isn't a system message.
- run summary written even when an exercise is unfinished? No: run needs all six; it must fail
  with a clear message naming the missing exercise.

### Task 1: Checkpoint tests (RED against the starter)
`modules/m03_tokenizers/tests/{conftest.py,test_checkpoint.py}`; exercise tests with fixed values,
run-summary tests, `test_understanding`. Verify: `make check` on starter fails every exercise.

### Task 2: Solutions, starter, hints, answers, hashes (GREEN)
`solutions/exercises.py`, `starter/exercises.py` + `exercises.py`, `answers.toml` (starter/solutions),
`hints.toml`, `tests/answer_hashes.json` via `make answer-hashes M=03`.

### Task 3: Run script
`tokenize_check.py` (+ `--offline`), `corpus.txt`, `chat_template.jinja` copy; add to
`tests/test_module_solutions.py`.

### Task 4: Lesson notebook
Builder in scratchpad writes `lesson.ipynb` and `starter/lesson.ipynb`; executes top to bottom.

### Task 5: README, module.toml, content tests, changelog, docs
`tests/test_m03_content.py` (as M2), README glossary + syntax table, module steps + metric,
CHANGELOG (M3 + Part G line), est_minutes.
