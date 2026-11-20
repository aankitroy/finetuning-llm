# M2 · How an LLM actually works — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build M2 in M1's beginner-first style, so the learner writes LLM text generation by hand and matches `model.generate()`.

**Architecture:** A new module folder, `modules/m02_how_llms_work/`, with the same shape as M1: lesson.ipynb + starter copy, exercises.py + starter, answers.toml + starter, solutions/, hints.toml, generate.py (the run step), tests/ (conftest, test_checkpoint, answer_hashes.json). Structural content tests go in `tests/test_m02_content.py`.

**Tech Stack:** PyTorch 2.15, transformers 5.23 (`LlamaForCausalLM`, `AutoModelForCausalLM`, `AutoTokenizer`), nbformat/nbclient, pytest.

**Spec:** `docs/superpowers/specs/2026-11-20-m02-how-llms-work-design.md`

## Global Constraints

- Real model: `HuggingFaceTB/SmolLM2-135M`. Tests use a tiny random `LlamaForCausalLM` (vocab 128, hidden 32, 2 layers, 4 heads, 2 KV heads, seed 0) and never download.
- The greedy comparison runs on CPU in float32, so it's exact.
- Every learner-facing rule from M1 applies: plain words first, a **What each line does** box after every code cell, terms bold where defined, step comments without code, the starter notebook unrun.
- Never touch the learner's `.progress/`, `outputs/`, or M1 `exercises.py` / `answers.toml` / `lesson.ipynb`.
- `est_minutes` stays 180.

## Review Focus

1. Sampling tests must be deterministic: pass a seeded `torch.Generator` and use `torch.multinomial(..., generator=generator)`.
2. The top-p boundary: keep the smallest set whose probabilities sum to ≥ p, *including* the token that crosses p, and always keep at least one token.
3. `generate()` must stop at `eos_token_id` and include that token, as `model.generate` does.
4. The checkpoint must fail when the run used `--tiny` (not the real model), except in the solutions self-test.
5. The lesson notebook test must skip (not fail) when SmolLM2 isn't cached, e.g. in CI.

---

### Task 1: M1 review leftovers
**Files:** `.gitignore`, `modules/m01_pytorch_essentials/hints.toml`, `modules/m01_pytorch_essentials/solutions/exercises.py`, `tests/test_m01_content.py`
- [ ] Test `test_hints_are_nudges_not_solutions`: no M1 exercise hint contains `p -= lr * p.grad` or `torch.randperm(len(X), device=X.device)`; the q5 hint mentions "section 5". Expect FAIL; then reword the hints. Expect PASS.
- [ ] Drop `model.train()` from the solution (the suite stays green). Add `.ipynb_checkpoints/` to `.gitignore`. The `test_lesson_matches_starter` skip happens only when `returncode == 1`.
- [ ] Commit: `chore(m01): tidy review leftovers`

### Task 2: M2 exercises, reference solutions and checkpoint tests
**Files:** create `modules/m02_how_llms_work/{exercises.py,starter/exercises.py,solutions/exercises.py,hints.toml,tests/conftest.py,tests/test_checkpoint.py}`
**Interfaces (produces):**
- `next_token_logits(model, input_ids: LongTensor(1, seq)) -> Tensor(vocab,)`
- `greedy(logits: Tensor(vocab,)) -> int`
- `softmax_with_temperature(logits, temperature: float) -> Tensor(vocab,)` (sums to 1)
- `top_p_filter(probs, p: float) -> Tensor(vocab,)`: renormalised, zeros outside the kept set
- `causal_attention(q, k, v: Tensor(seq, d)) -> Tensor(seq, d)`
- `generate(model, input_ids, max_new_tokens, temperature=0.0, top_p=1.0, eos_token_id=None, generator=None) -> list[int]`: new tokens only; `temperature == 0` is greedy
- [ ] Tests (each carries a beginner-readable failure message):
  - `test_next_token_logits`: shape `(128,)`, equal to `model(ids).logits[0, -1]`
  - `test_greedy`: on `[0.1, 2.0, -1.0]` the result is `1`, and it's a Python int
  - `test_softmax_with_temperature`: sums to 1; T=1 equals softmax; T=0.5 has a larger max than T=2
  - `test_top_p_filter`: `[0.5, 0.3, 0.15, 0.05]` with p=0.75 gives `[0.625, 0.375, 0, 0]`; p=0.1 keeps only the top token
  - `test_causal_attention`: matches `F.scaled_dot_product_attention(q, k, v, is_causal=True)` on seeded (6, 8) inputs; the first row equals `v[0]`
  - `test_generate_greedy_matches_library`: 20 tokens on the tiny model equal `model.generate(do_sample=False)`
  - `test_generate_stops_at_eos`: with `eos_token_id` set to the library's 3rd token, it returns 3 tokens
  - `test_generate_sampling_respects_top_p`: with a seeded generator and top_p 0.5, each sampled token is inside that step's top-p set (checked by recomputing)
- [ ] With the starter, expect FAIL. Then with `FTCOURSE_SOLUTIONS=1`, expect PASS.
- [ ] Commit: `feat(m02): exercises, reference solutions and checkpoint tests`

### Task 3: Questions, the run script and the manifest
**Files:** `answers.toml` + starter + solutions, `tests/answer_hashes.json`, `generate.py`, `module.toml` (`[[steps]]` and metrics), the `tests/test_module_solutions.py` entry, the checkpoint tests for the run
- [ ] Tests:
  - `test_run_matched_the_library` (reports the `tokens_per_sec` metric)
  - `test_run_used_the_real_model` (skipped when `FTCOURSE_SOLUTIONS` is set)
  - `test_run_with_your_code`
  - `test_understanding[q1..q6]`
  - add `("m02_how_llms_work", ["generate.py", "--tiny"])` to `MODULES`
- [ ] `generate.py`:
  - flags `--tiny`, `--prompt` (default "The capital of France is"), `--tokens` (default 20)
  - prints both texts and a sampled continuation
  - writes the `run_summary.json` keys `matched`, `model`, `prompt`, `your_text`, `library_text`, `tokens_compared`, `your_tokens_per_sec`, `library_tokens_per_sec`, `used_solutions`
- [ ] Manifest steps: lesson / exercises / answers / train (`generate.py`, metric `your_tokens_per_sec`) / checkpoint. Metrics: `tokens_per_sec`.
- [ ] Full suite PASS. Commit: `feat(m02): questions, generate.py run step and manifest`

### Task 4: The lesson notebook
**Files:** `lesson.ipynb` + `starter/lesson.ipynb` (built by a scratchpad builder script), `tests/test_m02_content.py`
- [ ] Tests:
  - starter unrun
  - sections 0–8 in order
  - every code cell explained
  - terms bold: language model, token, tokenizer, vocabulary, embedding, attention, causal mask, query, key, value, layer, logits, greedy, sampling, temperature, top-p, KV cache
  - step comments + "In plain words" + "section" pointers in every stub
  - question ids line up
  - the notebook runs (skipped if SmolLM2 isn't cached)
  - README glossary + syntax table
- [ ] Build the notebook, run it, and check every printed number against its box. Commit: `feat(m02): beginner-first lesson`

### Task 5: README and changelog
- [ ] README (steps, glossary, syntax at a glance, stuck?) and a CHANGELOG line. The full suite and ruff pass. Commit: `docs(m02): README, glossary and syntax table`
