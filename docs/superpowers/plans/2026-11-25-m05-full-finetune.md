# M5 · Full fine-tune Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build M5 so the learner fully fine-tunes Qwen2.5-0.5B-Instruct on a jargon → plain English style, with their own loop and with HF `Trainer`.

**Spec:** `docs/superpowers/specs/2026-11-25-m05-full-finetune-design.md`

## Global Constraints
- Same module layout and beginner-first copy as M3/M4; starter copies identical to the working copies.
- Tests and CI never download: a tiny random Qwen2 model (`tiny_model.py`) and `--tiny`.
- Targets come from measurement on the reference Mac (see the spec's ruling).

## Review Focus
- Masking off by one at the prompt/answer boundary, or padding graded in the loss.
- A learner's `collate` that pads on the left (breaks the shift for training).
- chrF without clipping, or counting spaces; buzzword counting that matches inside words.
- train.py blaming the learner for errors raised by the course's own code.
- Memory: lesson and train.py run in the same session (24 GB each).

### Task 1: Data (500 pairs, buzzwords, general questions) and the tiny model
### Task 2: Six exercises, TDD against the solutions; mutation check
### Task 3: train.py (--loop / --trainer / --tiny), measured; checkpoint gates from measurement
### Task 4: Questions, hints, manifest steps
### Task 5: Lesson (built by a scratchpad script), run top to bottom on the real model
### Task 6: README, content tests, CHANGELOG, shared solutions test supports several runs
