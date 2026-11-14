# Progress Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A local, auto-refreshing progress dashboard for the CLI-first fine-tuning course. Progress is detected from `make check` results and from artifacts on disk, and the page shows a next-step card and results over time.

**Architecture:** Each module has a `module.toml` manifest describing it. A pytest plugin, active only under `make check`, appends strict pass/fail records to `.progress/history.jsonl`. A pure function in `common/progress.py` combines the manifests, the history and an artifact scan into one state dict, which a stdlib HTTP server (`/api/state`) and a `--text` terminal mode both serve. A static page polls the state every 3 s and renders it with a vendored Chart.js.

**Tech Stack:** Python 3.11+ standard library (`tomllib`, `http.server`, `json`, `glob`), pytest (dev only), `uv`, plain HTML/CSS/JS, Chart.js 4.4.1 (vendored UMD build).

**Spec:** `docs/superpowers/specs/2026-11-14-progress-dashboard-design.md`

## Global Constraints

- Python `>=3.11`, managed with `uv`. The **server and state code import only the standard library.** The dev dependencies are pytest, ruff and pre-commit (added in Task 8).
- The repo is a public open-source project: no personal email addresses in any file, and all code must pass `make lint` and CI.
- The server binds `127.0.0.1` only. Port search starts at `8765`.
- The frontend has no CDN references at runtime. Chart.js is vendored at `dashboard/vendor/chart.umd.min.js`, version `4.4.1`.
- `.progress/` and `outputs/` are gitignored.
- History records use schema `v: 1` with keys `v, ts, module, passed, tests{passed,failed}, metrics, run_label` (spec 5.2). `ts` is ISO 8601 with a local offset.
- The plugin writes **only** when env `FTCOURSE_RECORD=<NN>` is set. Strict pass rule as in spec 5.3.
- Reserved cross-module metric keys: `peak_mem_gb`, `tokens_per_sec`, `trainable_pct`, `regression_score`.
- Status values: `"done"`, `"in_progress"`, `"not_started"`. The page polls every `3000` ms.
- Light and dark mode follow `prefers-color-scheme`. The layout must be usable at about 640 px wide.

## Review Focus

1. **No `.progress/` dir or history file yet** (a brand-new learner): state returns the empty-state next step ("run `make doctor`") with no crash. Covered in Task 3.
2. **Truncated or malformed history line** (Ctrl-C mid-write, hand edits): the line is skipped, a warning is added, and other records still count. Covered in Task 2.
3. **A broken or missing `module.toml` in one module**: that module is skipped with a warning, and the rest of the course still loads. Covered in Task 1.
4. **A record for a module id with no manifest** (a renamed or deleted module): it is ignored for status and progress counts, a warning is added, and it is not charted. Covered in Task 3.
5. **Port 8765 already in use** (a second `make progress`): the server uses the next free port and prints the real URL. Covered in Task 6.

---

## File Structure

```
pyproject.toml                      # uv project, pytest dev dep, pytest config
.gitignore
Makefile                            # doctor, check, progress, test
common/__init__.py
common/course.py                    # manifest loading (Module dataclass)
common/progress.py                  # history I/O, report_metric, compute_state, render_text
common/progress_plugin.py           # pytest plugin (strict records)
scripts/doctor.py                   # environment checks → outputs/m00/doctor.json
scripts/progress_server.py          # stdlib server + --text mode
dashboard/index.html, app.js, style.css
dashboard/vendor/chart.umd.min.js
modules/m00_setup/ … modules/m16_capstone/module.toml   # 17 manifests
modules/m00_setup/tests/test_checkpoint.py
tests/                              # the dashboard's own tests (never write records)
tests/fixtures/                     # sample courses + histories
```

---

### Task 1: Project skeleton + manifest loader + 17 manifests

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `common/__init__.py`, `common/course.py`, `modules/mNN_<slug>/module.toml` ×17, `tests/test_course.py`, `tests/fixtures/course_basic/…`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) class Module: id: str; title: str; part: str; order: int; summary: str; est_minutes: int; next_start: str; next_finish: str; next_why: str; artifacts: list[str]; metrics: list[dict]; path: Path`. Here `metrics` items are `{"key", "label", "better"}` and `better ∈ {"lower","higher"}`.
  - `load_modules(root: Path) -> tuple[list[Module], list[str]]`. It returns the modules sorted by `order` plus warnings. It globs `root/"modules"/"m*"/"module.toml"`.

- [ ] **Step 1: Write `pyproject.toml` and `.gitignore`.** Set the project name to `finetuning-llm`, `requires-python = ">=3.11"`, the dev group to `pytest>=8`, and `[tool.pytest.ini_options] testpaths = ["tests"]`. The gitignore covers `.progress/`, `outputs/`, `.venv/` and `__pycache__/`. Run `uv sync`. Expected: it succeeds.

- [ ] **Step 2: Write the failing tests** in `tests/test_course.py`:

```python
def test_loads_real_course():
    mods, warnings = load_modules(REPO_ROOT)
    assert warnings == []
    assert [m.id for m in mods] == [f"m{i:02d}" for i in range(17)]
    assert [m.order for m in mods] == list(range(17))
    assert all(m.next_finish.startswith("make ") for m in mods)

def test_broken_manifest_is_skipped_with_warning(tmp_path):
    # fixture: m00 valid, m01 invalid TOML, m02 missing "title"
    mods, warnings = load_modules(copy_fixture("course_basic", tmp_path))
    assert [m.id for m in mods] == ["m00"]
    assert len(warnings) == 2 and "m01" in warnings[0]

def test_metric_better_validated(tmp_path):
    # better = "sideways" → module skipped, warning mentions "better"
```

- [ ] **Step 3: Run** `uv run pytest tests/test_course.py -v`. Expected: FAIL (import error).

- [ ] **Step 4: Implement `load_modules`** in `common/course.py` using `tomllib`. It catches `TOMLDecodeError`, `KeyError` and `ValueError` per file and turns each into a warning string. `[next]` keys map to `next_start`, `next_finish` and `next_why`. A missing `[artifacts]` becomes `[]`, and missing `[[metrics]]` becomes `[]`.

- [ ] **Step 5: Write the 17 manifests.** Copy each module's id, title, Part label, order, one-line summary and estimate from `COURSE.md` §4. Folder slugs are: `m00_setup, m01_pytorch_essentials, m02_how_llms_work, m03_tokenizers, m04_training_loop, m05_full_finetune, m06_lora_from_scratch, m07_lora_tools, m08_qlora, m09_data, m10_sft, m11_evaluation, m12_preference_tuning, m13_grpo, m14_export, m15_serve, m16_capstone`.
  - `next_finish` is `make check M=NN`, except m00, which uses `make doctor`.
  - `next_why` paraphrases that module's checkpoint line.
  - Modules that train a model (m05, m07, m08, m10, m12, m13) declare `final_loss`, `peak_mem_gb`, `tokens_per_sec` and `trainable_pct`. m11 declares `regression_score`.
  - The artifacts glob is `outputs/mNN/*` for every module.

- [ ] **Step 6: Run** `uv run pytest tests/test_course.py -v`. Expected: PASS.

- [ ] **Step 7: Commit:** `git add -A && git commit -m "feat: project skeleton, module manifests and loader"`

---

### Task 2: History I/O and metric reporting

**Files:**
- Modify: `common/progress.py` (create)
- Test: `tests/test_history.py`

**Interfaces:**
- Produces:
  - `HISTORY_PATH = Path(".progress/history.jsonl")`, always resolved against a `root`.
  - `read_history(root: Path) -> tuple[list[dict], list[str]]`. It returns records in file order plus warnings. A missing file gives `([], [])`.
  - `append_record(root: Path, record: dict) -> None`. It creates `.progress/`, writes one JSON line plus `\n`, and uses `open("a")` followed by `flush()` and `os.fsync`.
  - `report_metric(key: str, value: float) -> None` and `collected_metrics() -> dict[str, float]` / `reset_metrics() -> None`. These are a module-level store that the plugin drains.

- [ ] **Step 1: Write the failing tests:**

```python
def test_missing_file_is_empty(tmp_path):
    assert read_history(tmp_path) == ([], [])

def test_roundtrip(tmp_path):
    append_record(tmp_path, {"v": 1, "module": "m00", "passed": True})
    append_record(tmp_path, {"v": 1, "module": "m01", "passed": False})
    recs, w = read_history(tmp_path)
    assert [r["module"] for r in recs] == ["m00", "m01"] and w == []

def test_malformed_and_truncated_lines_skipped(tmp_path):
    # line 2 = "not json", line 3 = '{"v": 1, "module": "m0'  (truncated, no newline)
    recs, w = read_history(tmp_path)
    assert len(recs) == 1 and len(w) == 2 and "line 2" in w[0]

def test_record_missing_required_keys_skipped(tmp_path):
    # {"v": 1} with no module/passed → skipped, warning

def test_report_metric_collects_and_resets():
    reset_metrics(); report_metric("final_loss", 1.5)
    assert collected_metrics() == {"final_loss": 1.5}
    reset_metrics(); assert collected_metrics() == {}
```

- [ ] **Step 2: Run** `uv run pytest tests/test_history.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** the functions above. The required keys checked on read are `module` (str) and `passed` (bool).
- [ ] **Step 4: Run** the same command. Expected: PASS.
- [ ] **Step 5: Commit:** `git commit -am "feat: progress history I/O and metric reporting"`

---

### Task 3: State computation (status, next step, charts data)

**Files:**
- Modify: `common/progress.py`
- Test: `tests/test_state.py`, `tests/fixtures/`

**Interfaces:**
- Consumes: `load_modules`, `read_history` (Tasks 1–2).
- Produces: `compute_state(root: Path) -> dict` with this exact shape:

```python
{
  "course":   {"total": int, "done": int, "est_minutes_done": int},
  "modules":  [{"id", "title", "part", "order", "summary", "status",
                "latest_metrics": dict, "attempts": [record, ...],
                "metrics": [ {key,label,better}, ... ]}],
  "next":     {"kind": "finish" | "retry" | "start" | "empty" | "complete",
               "module": str | None, "title": str, "instruction": str,
               "command": str | None, "why": str},
  "history":  [record, ...],   # only records whose module is known
  "warnings": [str, ...],
}
```

- [ ] **Step 1: Write the failing tests.** Use a fixture course of three modules (m00, m01, m02) and build histories in `tmp_path`:

```python
def test_empty_state():            # no history, no artifacts
    s = compute_state(root)
    assert s["next"]["kind"] == "empty" and s["next"]["command"] == "make doctor"
    assert all(m["status"] == "not_started" for m in s["modules"])

def test_done_needs_a_passing_record():
    # m00: failed then passed → done; m01: failed only → in_progress
    assert status(s, "m00") == "done" and status(s, "m01") == "in_progress"

def test_artifact_marks_in_progress():
    # create root/outputs/m01/adapter.safetensors, no records
    assert status(s, "m01") == "in_progress"

def test_next_prefers_lowest_in_progress_and_retry_wording():
    # m00 done, m01 failed (tests 2 passed / 1 failed), m02 artifact
    n = s["next"]
    assert n["module"] == "m01" and n["kind"] == "retry"
    assert "1 of 3" in n["instruction"] and n["command"] == "make check M=01"

def test_next_start_when_nothing_in_progress():
    # m00 done only → next kind "start", module m01, instruction == m01.next_start

def test_complete():
    # all passed → kind "complete", command None

def test_unknown_module_record_ignored_with_warning():
    # record module "m99" → not in history, warning mentions m99, done count unaffected

def test_latest_metrics_and_counts():
    # m00 passed twice with metrics {"peak_mem_gb": 2.0} then {"peak_mem_gb": 1.5}
    assert module(s, "m00")["latest_metrics"] == {"peak_mem_gb": 1.5}
    assert s["course"]["done"] == 1 and s["course"]["est_minutes_done"] == m00_est
```

- [ ] **Step 2: Run** `uv run pytest tests/test_state.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement `compute_state`.** Apply spec 5.5's rules exactly. For "empty", the condition is that no records exist and no artifacts match. Its command is `make doctor`, its title is "Start with M0 · Setup" and its why is m00's `next_why`. Expand artifact globs with `glob.glob(str(root/pattern))`. `latest_metrics` comes from the module's last record, pass or fail. Warnings from the loader and the history are merged in.
- [ ] **Step 4: Run** the same command. Expected: PASS.
- [ ] **Step 5: Commit:** `git commit -am "feat: compute dashboard state from manifests, history and artifacts"`

---

### Task 4: Strict pytest plugin

**Files:**
- Create: `common/progress_plugin.py`, `tests/test_plugin.py`

**Interfaces:**
- Consumes: `append_record`, `collected_metrics`, `reset_metrics`.
- Produces: the plugin is loaded with `pytest -p common.progress_plugin`. It is inert unless `FTCOURSE_RECORD` is set. The record root is `FTCOURSE_ROOT` (defaulting to cwd), and the label comes from `FTCOURSE_LABEL` (optional, empty becomes `null`).

- [ ] **Step 1: Write the failing tests** using pytest's `pytester` fixture (`pytest_plugins = ["pytester"]`). Each test creates a fake `modules/m05_x/tests/` and runs `pytester.runpytest_subprocess("-p", "common.progress_plugin", "modules/m05_x/tests", ...)` with env set:

```python
def test_inert_without_env():            # no FTCOURSE_RECORD → no history file
def test_all_pass_records_passed_true(): # record: module "m05", passed True, tests {"passed":2,"failed":0}
def test_failure_records_passed_false():
def test_skip_makes_it_fail():           # one plain skip → passed False
def test_optional_checkpoint_skip_ok():  # skip marked @pytest.mark.optional_checkpoint → passed True
def test_keyword_filter_never_passes():  # extra "-k", "one" → passed False
def test_nodeid_narrowing_never_passes():# path ".../test_a.py::test_one" → passed False
def test_zero_tests_never_passes():
def test_collection_error_records_false():
def test_metrics_and_label_recorded():   # test calls report_metric("final_loss", 1.2); FTCOURSE_LABEL="r=8"
    assert rec["metrics"] == {"final_loss": 1.2} and rec["run_label"] == "r=8"
```

- [ ] **Step 2: Run** `uv run pytest tests/test_plugin.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement the hooks:**
  - `pytest_configure`: register the `optional_checkpoint` marker and call `reset_metrics()`.
  - `pytest_runtest_logreport`: tally passed, failed and errors, and record skips that aren't marked optional.
  - `pytest_collectreport`: catch collection errors.
  - `pytest_keyboard_interrupt`: set an interrupted flag.
  - `pytest_sessionfinish`: write exactly one record.

  "Full suite" means `config.option.keyword` and `config.option.markexpr` are empty and every arg in `config.args` is a directory, with no `::` and no file paths. Build `ts` with `datetime.now().astimezone().isoformat(timespec="seconds")`.
- [ ] **Step 4: Run** the same command. Expected: PASS.
- [ ] **Step 5: Commit:** `git commit -am "feat: strict pytest plugin that records checkpoint attempts"`

---

### Task 5: Makefile, `make doctor`, M0 checkpoint

**Files:**
- Create: `Makefile`, `scripts/doctor.py`, `modules/m00_setup/README.md`, `modules/m00_setup/tests/test_checkpoint.py`, `tests/test_make.py`

**Interfaces:**
- Consumes: the plugin (Task 4).
- Produces:
  - `make check M=NN [LABEL=...]` runs `FTCOURSE_RECORD=NN FTCOURSE_ROOT=$(CURDIR) FTCOURSE_LABEL="$(LABEL)" uv run pytest -p common.progress_plugin modules/mNN_*/tests`, globbing the module dir from `M`. A missing `M` or no matching dir prints a usage message and exits 2.
  - `make doctor` runs `uv run python scripts/doctor.py && $(MAKE) check M=00`.
  - `make progress [TEXT=1]` and `make test` (which runs `uv run pytest`).
  - `scripts/doctor.py` writes `outputs/m00/doctor.json`: `{"python": "3.x.y", "python_ok": bool, "platform": str, "apple_silicon": bool, "total_mem_gb": float, "mps_available": bool | null, "mlx_available": bool | null}`. Here `null` means the package isn't installed yet. It prints a readable checklist and exits non-zero only if `python_ok` is false. It reads memory via `sysctl -n hw.memsize`.

- [ ] **Step 1: Write the M0 checkpoint test.** It asserts `outputs/m00/doctor.json` exists, `python_ok` is true, and `total_mem_gb > 0`. It also calls `report_metric("total_mem_gb", ...)`. MPS and MLX checks are `optional_checkpoint` tests that skip if the value is `null`, because torch and MLX are installed in later modules.
- [ ] **Step 2: Write `tests/test_make.py`:** `test_check_without_m_exits_2`, which runs `make check` via subprocess with `FTCOURSE_ROOT` pointed at `tmp_path`, so the real history is never touched.
- [ ] **Step 3: Implement** `Makefile` and `scripts/doctor.py`.
- [ ] **Step 4: Verify** with `make doctor`. Expected: the checklist prints, then pytest passes, and `.progress/history.jsonl` gains a line with `"module": "m00", "passed": true`. Then run `uv run pytest`. Expected: all pass, with no new history line, because the dashboard's own tests never record.
- [ ] **Step 5: Commit:** `git commit -am "feat: make doctor/check targets and M0 checkpoint"`

---

### Task 6: Server and text mode

**Files:**
- Create: `scripts/progress_server.py`, `tests/test_server.py`
- Modify: `common/progress.py` (add `render_text`)

**Interfaces:**
- Consumes: `compute_state`.
- Produces:
  - `render_text(state: dict) -> str`. It renders the header, progress bar and Part-grouped checklist using the icons `✔ ◐ ○`, then a NEXT box with the instruction, command and why.
  - `find_free_port(start: int = 8765, tries: int = 20) -> int`.
  - `make_server(root: Path, port: int) -> ThreadingHTTPServer`.
  - CLI: `python scripts/progress_server.py [--text] [--no-open] [--root PATH]`.

- [ ] **Step 1: Write the failing tests:**

```python
def test_api_state_shape(tmp_path):      # server on find_free_port(); GET /api/state → 200, keys == {"course","modules","next","history","warnings"}
def test_index_and_static_served():      # GET / → 200 text/html; GET /static/app.js → 200
def test_path_traversal_blocked():       # GET /static/../common/progress.py → 404
def test_port_in_use_falls_forward():    # bind a socket on p, find_free_port(p) == p+1 (or higher)
def test_render_text_contains_next():    # "NEXT" and "make doctor" in render_text(empty state)
```

- [ ] **Step 2: Run** `uv run pytest tests/test_server.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement it.** Recompute the state on each `/api/state` request and set `Cache-Control: no-store`. Serve static files only from `dashboard/`, resolving the path and requiring it to stay under that directory. Bind `127.0.0.1`. Print `Dashboard: http://127.0.0.1:<port>`. Open the browser unless `--no-open` is given. Ctrl-C exits cleanly. Point `make progress` at the script, and `TEXT=1` at `--text`.
- [ ] **Step 4: Run** the same command. Expected: PASS.
- [ ] **Step 5: Commit:** `git commit -am "feat: local progress server and text mode"`

---

### Task 7: Dashboard page

**Files:**
- Create: `dashboard/index.html`, `dashboard/app.js`, `dashboard/style.css`, `dashboard/vendor/chart.umd.min.js`, `tests/fixtures/histories/{partial,complete}.jsonl`

**Interfaces:**
- Consumes: `/api/state` (Task 3 shape).

- [ ] **Step 1: Vendor Chart.js:** `curl -fsSL https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js -o dashboard/vendor/chart.umd.min.js`. Expected: the file is non-empty and starts with `/*!`.
- [ ] **Step 2: Build the page** following spec 5.7, sections 1–5, in order:
  - header with progress bar and estimated time
  - prominent next-step card with a copy button (`navigator.clipboard`)
  - course map grouped by Part, with expandable attempt history per module
  - results: a cumulative-passes timeline, per-module charts for each declared metric across attempts (x axis is the attempt, with `run_label` as the tick, and the title notes "lower/higher is better"), and a comparison bar chart of `peak_mem_gb`, `tokens_per_sec` and `trainable_pct` (plus `regression_score` when present) for m05, m07, m08, m10 and m12, using each module's latest passing record
  - empty state

  It polls `/api/state` every 3000 ms, diffs the done set, and adds a 2 s highlight class to newly done modules. Charts are updated in place, not recreated. The Results section is hidden when no metrics exist. Warnings show as a small dismissible footer note. The style direction is a calm course companion: CSS variables for colours, a `prefers-color-scheme` dark palette, a single column below 720 px, and no horizontal scroll at 640 px.
- [ ] **Step 3: Verify manually.** Run `python scripts/progress_server.py --root <fixture-root> --no-open` for the empty, partial and complete fixtures, and open each in a browser. Check that the next card matches `make progress TEXT=1`, the charts render offline (with the network disabled), dark mode works, and the layout holds at 640 px. Then run `make check M=00` in the real repo with the page open. Expected: M0 flips to ✔ within about 3 s and is highlighted.
- [ ] **Step 4: Commit:** `git add dashboard tests/fixtures && git commit -m "feat: dashboard page with next step, course map and results charts"`

---

### Task 8: Open-source project setup

**Files:**
- Create:
  - `README.md`, `LICENSE`, `LICENSE-CONTENT`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, `CHANGELOG.md`
  - `.editorconfig`, `.pre-commit-config.yaml`
  - `.github/workflows/ci.yml`, `.github/PULL_REQUEST_TEMPLATE.md`
  - `.github/ISSUE_TEMPLATE/{bug_report.yml,lesson_feedback.yml,module_proposal.yml,config.yml}`
  - `modules/_template/{module.toml,README.md,tests/test_checkpoint.py}`
- Modify:
  - `pyproject.toml`: add `ruff>=0.6` to dev, with `[tool.ruff] line-length = 100, target-version = "py311"` and lint `select = ["E","F","I","UP","B"]`
  - `Makefile`: add `lint` (`uv run ruff check . && uv run ruff format --check .`) and `fmt`
  - `COURSE.md` §5: make the "Progress tracker" bullet describe the dashboard

**Decisions (from the conversation; licences are proposed defaults the owner can change):**
- **Code licence:** MIT, copyright "2026 Aankit Roy".
- **Course content licence:** CC BY 4.0 for `modules/**/README.md`, notebooks and `COURSE.md`, declared in `LICENSE-CONTENT` and in the README licence section.
- **No personal email anywhere:**
  - Security reports go through GitHub private vulnerability reporting (`SECURITY.md`).
  - Code of conduct reports go through a GitHub issue or private advisory to the maintainers.
- **Commits** follow Conventional Commits, as the earlier tasks already do.
- **`modules/_template/`** isn't loaded as a module, because `load_modules` globs `m*`. Add a test that confirms this.

- [ ] **Step 1: Write the failing test** in `tests/test_course.py`, `test_template_dir_not_loaded`: `"_template"` must not be among the loaded module paths. Then run `uv run pytest tests/test_course.py -v`. Expected: PASS already, because of the glob, which pins the behaviour. If it fails, fix the glob.
- [ ] **Step 2: Write the licences.**
  - `LICENSE`: the standard MIT text.
  - `LICENSE-CONTENT`: a CC BY 4.0 notice with its scope (which files) and the link `https://creativecommons.org/licenses/by/4.0/`.
  - `CODE_OF_CONDUCT.md`: Contributor Covenant 2.1, fetched with `curl -fsSL https://www.contributor-covenant.org/version/2/1/code_of_conduct/code_of_conduct.md`. Replace its `[INSERT CONTACT METHOD]` with the GitHub-based reporting route above.
- [ ] **Step 3: Write `README.md`** with these sections:
  - title and a one-line pitch
  - CI and licence badges
  - "What you'll learn" (the Part A–F list from COURSE.md)
  - hardware notes (the memory table summary, and that it's tested on M-series Macs)
  - Quick start: `git clone`, `uv sync`, `make doctor`, `make progress`
  - "How progress works": only `make check` records, with the strict rule
  - repo layout
  - Contributing link
  - Licence section
- [ ] **Step 4: Write `CONTRIBUTING.md`:**
  - dev setup (`uv sync`, `pre-commit install`)
  - `make test` and `make lint`
  - **adding a module:** copy `modules/_template` to `modules/mNN_slug`, fill in `module.toml` (the field list from Task 1), write checkpoint tests that read `outputs/mNN/run_summary.json` and call `report_metric`, use the reserved metric keys for training modules, and keep `--tiny` runs under 2 minutes
  - commit style and the PR checklist
  - the rule that **checkpoints verify and do not retrain**
- [ ] **Step 5: Write the remaining files:**
  - `SECURITY.md`, `CHANGELOG.md` (Keep a Changelog format, with an `Unreleased` entry listing the dashboard and M0), `.editorconfig` (4-space Python, 2-space YAML, JS, CSS and TOML; LF; final newline)
  - `.pre-commit-config.yaml`: ruff and ruff-format hooks, plus `end-of-file-fixer` and `trailing-whitespace`. Pin the versions current at the time of writing.
  - Issue forms:
    - bug: required fields for macOS version, chip and memory, module, the pasted `make doctor` output, and the command and error
    - lesson feedback
    - module proposal
  - `config.yml` sets `blank_issues_enabled: false`.
  - PR template: a checklist covering tests, lint, the CHANGELOG, and that a new module has a manifest and a checkpoint.
- [ ] **Step 6: Write `.github/workflows/ci.yml`:**
  - triggers: `push` to `main` and `pull_request`
  - job `test` with matrix `os: [macos-14, ubuntu-latest]` × `python: ["3.11", "3.12"]`
  - steps: `actions/checkout@v4`, `astral-sh/setup-uv@v3`, `uv sync --python ${{ matrix.python }}`, `make lint`, `uv run pytest`
  - the dashboard tests use only the standard library, so Ubuntu works too, and macOS covers the target platform
- [ ] **Step 7: Verify** with `make lint && uv run pytest`. Expected: both pass. Then `uv run pre-commit run --all-files`. Expected: passes. If it auto-fixes anything, re-run until it's clean.
- [ ] **Step 8: Commit:** `git add -A && git commit -m "chore: open-source project setup (licences, contributing, CI, templates)"`
