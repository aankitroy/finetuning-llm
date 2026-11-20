# M1 for complete beginners — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite M1 so someone who has never seen a neural network or PyTorch can finish it with only the repo open.

**Architecture:** Content change inside `modules/m01_pytorch_essentials/`: a concept-first notebook, step-commented exercise stubs, a new question, a glossary README. One tiny code change (reset message) plus a clean `starter/lesson.ipynb` so reset restores the notebook. Content is pinned by structural tests in `tests/test_m01_content.py`.

**Tech Stack:** Python 3.11+, PyTorch, Jupyter (`nbformat`, `nbclient` 0.11 already installed in the `course` group), pytest.

**Spec:** `docs/superpowers/specs/2026-11-16-m01-beginner-rewrite-design.md`

## Global Constraints

- Function names, signatures and `tests/test_checkpoint.py` assertions for exercises 1–6 do not change. The 95% bar and strict pass rules do not change.
- Step comments say *what* a line must do, never the code that does it.
- The section 7 worked example must not use two moons and must not be copy-paste-able into `MLP`/`train_one_epoch` (use a regression problem: fit `y = sin(x)` with `nn.Tanh`, MSE, full batch).
- Committed notebooks have no outputs (`execution_count` null, `outputs` empty).
- Never edit the learner's `.progress/` or run `make check`/`make doctor` against the repo root. `exercises.py`/`answers.toml` are overwritten only because they currently equal their starters (checked 2026-11-16).
- Copy: plain English, sentence case, no unexplained term. Each new term **bold** where first defined.
- `est_minutes = 240`.

## Review Focus

1. A learner who edited/ran `lesson.ipynb` runs `make test`: content tests must check the *starter* notebook and the committed structure, not fail because the learner ran cells.
2. The notebook must execute top to bottom on a machine without MPS (CPU fallback) — covered by the nbclient test.
3. `make reset M=01` after running the lesson restores the unrun notebook and the lesson step returns to "todo".
4. `make solution M=01 EX=q6` and `make hint M=01 EX=q6` work (q6 present in hints and solutions).
5. The guide still shows exercises as "Not written yet" with step comments present (comments aren't AST nodes, so `_is_stub` is unaffected — pin with a test).

---

### Task 1: Reset restores the lesson notebook

**Files:**
- Modify: `scripts/course.py:217` (reset message)
- Test: `tests/test_course_cli.py`

- [ ] **Step 1: Write tests**
  - `test_reset_restores_a_starter_notebook(root)`: write `starter/lesson.ipynb` (`'{"cells": []}'`) and a run `lesson.ipynb` (different bytes) into `m05_x`; `reset_plan(root, "05")` contains `"modules/m05_x/lesson.ipynb"`; after `do_reset(root, "05")` the bytes match the starter. (Pins existing behaviour; expected to pass on first run — a regression pin, not new logic.)
  - `test_main_reset_says_notebooks_are_reset_too(root, monkeypatch, capsys)`: input `"no"`; output does not contain `"Lesson notebooks"`, does contain `"Reference solutions are kept."`.
- [ ] **Step 2: Run** `uv run pytest tests/test_course_cli.py -q` — Expected: second test FAILS on the old message.
- [ ] **Step 3:** Change the message to `"This can't be undone. Reference solutions are kept."`
- [ ] **Step 4: Run** same command — Expected: PASS.
- [ ] **Step 5: Commit** `fix: reset message no longer says notebooks are kept`

### Task 2: Concept-first lesson notebook

**Files:**
- Create: `scratchpad/build_m01_lesson.py` (not committed; writes the notebook with `nbformat.v4`)
- Replace: `modules/m01_pytorch_essentials/lesson.ipynb`
- Create: `modules/m01_pytorch_essentials/starter/lesson.ipynb` (byte-identical copy)
- Test: `tests/test_m01_content.py`

**Interfaces:** Produces section headings that `README.md` (Task 5) links to by name.

- [ ] **Step 1: Write tests** (module constant `M = REPO_ROOT/"modules/m01_pytorch_essentials"`; read `M/"starter/lesson.ipynb"`):
  - `test_starter_lesson_is_unrun`: every code cell has `execution_count is None` and `outputs == []`.
  - `test_lesson_sections_in_order`: the `## ` headings of markdown cells, in order, start with these prefixes: `"## 0. What is a model"`, `"## 1. Tensors"`, `"## 2. Gradients and autograd"`, `"## 3. Linear regression, by hand"`, `"## 4. nn.Module"`, `"## 5. Classification"`, `"## 6. Batches and epochs"`, `"## 7. A worked example"`, `"## 8. The Mac's GPU"`, `"## 9. Your task"`.
  - `test_every_code_cell_is_explained`: every code cell is immediately followed by a markdown cell whose source starts with `"**What each line does**"`, except the final two-moons plot cell, which may be followed by the "What to do now" cell.
  - `test_key_terms_are_defined`: the joined markdown contains `**tensor**`, `**shape**`, `**dtype**`, `**device**`, `**matrix multiply**`, `**broadcasting**`, `**gradient**`, `**autograd**`, `**loss**`, `**logits**`, `**softmax**`, `**cross-entropy**`, `**argmax**`, `**ReLU**`, `**batch**`, `**epoch**`, `**learning rate**`, `**optimizer**` (case-insensitive).
  - `test_lesson_runs_top_to_bottom(tmp_path)`: `pytest.importorskip("torch")`; copy `M` contents to `tmp_path`; execute the starter notebook with `nbclient.NotebookClient(nb, timeout=120, kernel_name="python3", resources={"metadata": {"path": str(tmp_path)}})`; no exception.
  - `test_lesson_matches_starter`: `json.loads` of `lesson.ipynb` and `starter/lesson.ipynb` have equal cell *sources* — skipped (`pytest.skip("you've edited the lesson")`) when `git diff --quiet` on `lesson.ipynb` reports a change, so a learner's run doesn't fail `make test`.
- [ ] **Step 2: Run** `uv run pytest tests/test_m01_content.py -q` — Expected: FAIL (no starter notebook).
- [ ] **Step 3: Write the notebook** via the builder script, following the spec's section list. Requirements beyond the tests:
  - §0: a matplotlib or text diagram of guess → measure → nudge → repeat; no PyTorch yet.
  - §1: 0-d/1-d/2-d tensor examples; a worked 2×3 @ 3×2 multiply written out by hand in markdown, then checked in code; memory maths; broadcasting.
  - §2: plot (w−5)² with the slope at w=3; then `requires_grad`, `.backward()`, `.grad`, `torch.no_grad()`; a cell that calls `backward()` twice to show accumulation; the lr=1.1 Predict.
  - §5: three logits → softmax by hand → cross-entropy by hand → compare with `F.cross_entropy`; argmax picks the class.
  - §6: batch/epoch in words, a cell printing slices of a shuffled index list.
  - §7: `class SineNet(nn.Module)` with `nn.Tanh`, trained on `sin(x)` with MSE using full-batch steps, plotted fit.
  - Each "What each line does" box is a bullet per line/call used, in plain words.
  - Keep the Predict prompts and the Stretch.
- [ ] **Step 4:** Write it to both `lesson.ipynb` and `starter/lesson.ipynb`. Run the tests — Expected: PASS.
- [ ] **Step 5: Commit** `feat(m01): concept-first lesson for complete beginners` (stage `lesson.ipynb`, `starter/lesson.ipynb`, `tests/test_m01_content.py`; never `.ipynb_checkpoints/` or `.vincent/`).

### Task 3: Step-commented exercises and hints

**Files:**
- Modify: `modules/m01_pytorch_essentials/starter/exercises.py`, `exercises.py` (copy of starter), `hints.toml` (exercises 1–6)
- Test: `tests/test_m01_content.py`

- [ ] **Step 1: Write tests**
  - `test_each_exercise_has_step_comments`: parse `starter/exercises.py` source; for each of the six names, the source lines of that def (for `MLP`, both methods) contain a line matching `^\s+# 1\. ` and a `raise NotImplementedError`.
  - `test_each_exercise_says_what_it_is`: each def's docstring contains `"In plain words:"`.
  - `test_step_comments_dont_count_as_written`: `common.guide._written` over the parsed starter is False for all six (Review Focus 5).
  - `test_harder_exercises_point_to_the_lesson`: `MLP` and `train_one_epoch` docstrings contain `"section 7"`.
- [ ] **Step 2: Run** — Expected: FAIL.
- [ ] **Step 3:** Rewrite the starter. Example shape for `train_one_epoch`'s comments: `# 1. Shuffle the order of the rows (a random permutation of 0..len(X)-1, on X's device)`, `# 2. Loop over that order in slices of batch_size`, `# 3. Forward: get the logits for this batch`, `# 4. Loss: cross-entropy between logits and the batch's labels`, `# 5. Clear old gradients`, `# 6. Backward`, `# 7. Step the optimizer`, `# 8. Remember this batch's loss as a Python number`, `# 9. Return the average`. Copy starter → `exercises.py`. Reword hints 1–6 to refer to step numbers.
- [ ] **Step 4: Run** `uv run pytest tests/test_m01_content.py tests/test_module_solutions.py tests/test_guide.py -q` — Expected: PASS.
- [ ] **Step 5: Commit** `feat(m01): step-by-step exercise stubs and matching hints`

### Task 4: Question 6 on logits and cross-entropy

**Files:**
- Modify: `modules/m01_pytorch_essentials/starter/answers.toml`, `answers.toml` (copy), `solutions/answers.toml`, `hints.toml`, `tests/answer_hashes.json` (regenerate with `make answer-hashes M=01`)
- Test: `tests/test_m01_content.py`

- [ ] **Step 1: Write test** `test_question_ids_line_up`: ids in starter answers == ids in solutions == question ids (`q*`) in hints == keys of `answer_hashes.json` == `{"q1",…,"q6"}`.
- [ ] **Step 2: Run** — Expected: FAIL (no q6).
- [ ] **Step 3:** Add q6: `"A model outputs logits [2.0, 0.5, -1.0] for classes 0, 1, 2. Which class does it predict? (a number)"`, solution `kind = "number"`, `answer = "0"`, why explaining argmax and that softmax keeps the order. Reword q5's question to say "(see section 0 and the ReLU box in the lesson)". Add q6 hint. Regenerate hashes.
- [ ] **Step 4: Run** `uv run pytest -q` — Expected: all PASS (including `test_committed_answer_hashes_are_up_to_date` and the solutions run).
- [ ] **Step 5: Commit** `feat(m01): question on logits and cross-entropy`

### Task 5: README glossary, syntax table, manifest

**Files:**
- Modify: `modules/m01_pytorch_essentials/README.md`, `module.toml` (`est_minutes = 240`, step details mention "four hours" nowhere — keep times out of details), `CHANGELOG.md`
- Test: `tests/test_m01_content.py`

- [ ] **Step 1: Write tests**
  - `test_readme_glossary_covers_terms`: README has a `## Glossary` section containing a table row starting `| **<term>** |` for each term in the Task 2 list plus `MSE`, `layer`, `parameter`, `nn.Module`, `MPS`.
  - `test_readme_syntax_table`: README has `## Syntax at a glance` with rows for `` `requires_grad=True` ``, `` `.backward()` ``, `` `.grad` ``, `` `torch.no_grad()` ``, `` `nn.Module` ``, `` `nn.Linear` ``, `` `optimizer.zero_grad()` ``, `` `optimizer.step()` ``, `` `F.cross_entropy` ``, `` `.argmax(dim=1)` ``, `` `.to(device)` ``, `` `.item()` ``.
  - `test_m01_estimate`: `load_modules` gives m01 `est_minutes == 240`.
- [ ] **Step 2: Run** — Expected: FAIL.
- [ ] **Step 3:** Write the README: a "No experience needed" intro, steps (updated to six questions), glossary (term | plain meaning | where in the lesson), syntax table (call | what it does | when you use it), Stuck?. Update manifest and CHANGELOG (Unreleased: "M1 rewritten for complete beginners; reset now restores lesson notebooks").
- [ ] **Step 4: Run** `uv run pytest -q && uv run ruff check . && uv run ruff format --check .` — Expected: all PASS, clean.
- [ ] **Step 5: Commit** `docs(m01): glossary and syntax table for beginners`
