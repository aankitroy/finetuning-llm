# Progress Dashboard — Design

**Date:** 2026-11-14
**Status:** Draft, pending approval
**Related:** `COURSE.md` (course structure, modules M0–M16)

## 1. Intent

The course in `COURSE.md` is learned **entirely from the command line**: learners read a module's README, work through notebooks and scripts, and pass a checkpoint test with `make check M=NN`.

The dashboard is a **"feeler"**: a quick glance at where you are, what you've achieved and what to do next. It never runs training or lessons. It **works out progress automatically** from what the learner has actually done (no manual ticking), gives **guidance for the next step**, and shows **results over time** (loss, memory, eval scores, across modules and across repeated attempts).

Success looks like this: a learner runs `make progress` once, leaves the tab open, and as they pass checkpoints in the terminal the page updates on its own, showing a tick, a new data point and a fresh "next step" card.

## 2. Goals, non-goals, anti-goals

**Goals**
- Progress is detected automatically from checkpoint results and artifacts on disk.
- A single "Next step" card with the exact command to run and a one-line reason.
- Charts of results over time, keeping every attempt rather than only the latest.
- Opened with one command, works offline, adds no extra Python dependencies.
- Live refresh while the learner works in the terminal.

**Non-goals**
- Running checks, training, or notebooks from the UI. The terminal is the only place work happens.
- Accounts, multiple users, cloud sync, or hosting.
- Editing progress by hand from the UI.

**Anti-goals** (failure even if it technically works)
- Progress that lies: a module shown as done when its checkpoint never passed.
- A page that feels like a generic admin panel. It should feel like a calm, encouraging course companion.
- Making the course depend on the dashboard. Everything must still work if the learner never opens it.

## 3. Constraints

- Runs on macOS (Apple Silicon), Python 3.11+, managed with `uv` (as proposed in `COURSE.md`).
- The server uses **only the Python standard library** (`http.server`, `json`, `pathlib`, `webbrowser`).
- The frontend is **one static HTML page** with plain JS and CSS. The chart library is **vendored into the repo** (no CDN), so the page works offline.
- Binds to `127.0.0.1` only.

## 4. Approaches considered

| | Approach | Trade-off |
|---|---|---|
| **A (chosen)** | Local web page served by a stdlib Python server, with a vendored chart library and polling refresh | We write the layout ourselves, but we fully control the look and add zero dependencies |
| B | Streamlit dashboard | Fastest to write, but a heavy dependency, a generic look, and its own run and reload quirks |
| C | Terminal dashboard (Textual/Rich) | Matches the command line, but charts over time are crude as terminal characters |

The learner picked **A**.

## 5. Design

### 5.1 Data flow

```
 terminal                         disk                              browser
 ────────                         ────                              ───────
 make check M=06 ──pytest──► .progress/history.jsonl  ◄──GET /api/state── dashboard page
                              (append-only)                  (polls every 3s)
 training scripts ─────────► outputs/m06/...         ◄── artifact scan
                              modules/*/module.toml  ◄── course metadata
```

### 5.2 Progress history (`.progress/history.jsonl`)

- **Append-only**, one JSON record per checkpoint run, **pass or fail**. Failed attempts are kept as part of the learning story but never count as "done".
- `.progress/` is gitignored, so it belongs to the learner.
- Record schema (v1):

```json
{
  "v": 1,
  "ts": "2026-11-14T15:02:11+05:30",
  "module": "m06",
  "passed": true,
  "tests": {"passed": 4, "failed": 0},
  "metrics": {"final_loss": 1.83, "peak_mem_gb": 9.4, "trainable_pct": 0.42, "eval_score": 0.71},
  "run_label": "r=8"
}
```

- `metrics` is free-form, but each module declares which keys it reports (5.4), so charts know what to plot.
- `run_label` is optional. The learner can set it with `make check M=06 LABEL="r=8"` to tell attempts apart on charts.

### 5.3 How checks write records

- `make check M=NN` runs pytest on `modules/mNN_*/tests/`.
- **Only `make check` writes records.** The Makefile sets `FTCOURSE_RECORD=NN`. The plugin does nothing without it, so running pytest directly, the dashboard's own unit tests, or `-k` filtering never write progress. *(Builder's call, made in the builder check.)*
- **Strict pass rule:** a record has `passed: true` only if all of these hold. Anything else is written as `passed: false`, so the attempt is still visible. *(Builder's call.)*
  - the module's full checkpoint suite was collected (no `-k` or path narrowing, and the collected count matches the number of tests in the module's `tests/` directory)
  - at least one test ran
  - zero failures, errors or collection errors
  - the session wasn't interrupted
  - no skips, unless a test is marked `@pytest.mark.optional_checkpoint`
- **M0:** `make doctor` runs `scripts/doctor.py`, and when that succeeds it runs `make check M=00`, whose tests assert the doctor's results. The single command the page tells a new learner to run therefore records M0 as done. *(Builder's call.)*
- A shared pytest plugin, `common/progress_plugin.py`, collects the results and writes one record at the end of the session.
- Tests report numbers through a shared helper, `common.progress.report_metric("final_loss", value)`. The plugin gathers these into `metrics`.
- Where do numbers like `final_loss` come from? Training scripts save a small `outputs/mNN/run_summary.json` (written by the shared memory meter and training helpers in `common/`), and checkpoint tests read it, assert on it, and report the metrics from it. This keeps checks fast: **checks verify, they don't retrain**, except in modules where the training is the exercise and runs in the `--tiny` budget.

### 5.4 Course metadata (`modules/mNN_*/module.toml`)

Each module has a small manifest. The dashboard, `make check` and the next-step guidance all read it, so it is the single source of truth.

```toml
id = "m06"
title = "LoRA from scratch"
part = "C · Efficient tuning"
order = 6
summary = "Build a LoRA layer by hand and inject it into a real model."
est_minutes = 180

[next]                       # guidance when this module is the next one
start = "Read modules/m06_lora_from_scratch/README.md, then open lesson.ipynb"
finish = "make check M=06"
why = "Confirms your LoRA layer matches the base at init and trains < 1% of params"

[artifacts]                  # files that signal "in progress"
any = ["outputs/m06/adapter*.safetensors", "outputs/m06/run_summary.json"]

[[metrics]]                  # what to chart for this module
key = "final_loss"
label = "Final val loss"
better = "lower"
[[metrics]]
key = "peak_mem_gb"
label = "Peak memory (GB)"
better = "lower"
```

Unlike YAML, TOML can be read with the standard library (`tomllib`), which keeps the server dependency-free.

### 5.5 Module status rules

| Status | Rule |
|---|---|
| **Done** ✔ | At least one record with `passed: true` |
| **In progress** ◐ | Not done, and either a failed record exists or any `[artifacts]` glob matches |
| **Not started** ○ | Neither |

**Next step:**
1. If there is a module in progress, pick the lowest `order` among them. If it has a failed attempt, show "Your last check failed: N of M tests" plus `finish`. Otherwise show `finish`.
2. Otherwise, pick the lowest-`order` module that isn't started and show `start`, with `why`.
3. If everything is done, show a completion card pointing to the capstone tracks.

Modules are **not locked**. The learner may jump ahead. "Next" is a suggestion, not a gate.

### 5.6 Server (`scripts/progress_server.py`)

- `make progress` runs it. It picks a free port starting at 8765, prints the URL, and opens the browser.
- Routes:
  - `GET /` → `dashboard/index.html`
  - `GET /static/*` → the dashboard's JS, CSS and vendored chart library
  - `GET /api/state` → computed JSON: course (from the manifests), per-module status, the next step, the full history, and the metric declarations
- State is **recomputed on each request** from disk. The data is small (dozens to hundreds of records), so no caching or file watching is needed.
- A malformed line in `history.jsonl` is skipped and listed under `warnings` in the response. It never crashes the server.
- `make progress` also has a text fallback, `make progress TEXT=1`, which prints the same checklist and next step to the terminal using the same state logic. It's useful when the learner doesn't want a browser.

### 5.7 Page

One page, top to bottom:

1. **Header:** course title, an overall progress bar ("6 / 17 modules"), and total time invested (the sum of `est_minutes` for completed modules, labelled as an estimate).
2. **Next step card:** the most prominent element. It shows the module title, the instruction, a copy-to-clipboard command, and the "why" line.
3. **Course map:** modules grouped by Part (A–F plus Capstone), each with a status icon and its latest key metric. Clicking a module expands it to show its attempt history (time, pass/fail, metrics, label).
4. **Results over time:**
   - **Timeline:** checks passed, cumulative, by date.
   - **Per-module metric charts:** each declared metric plotted across attempts, labelled with `run_label`, and annotated with the "better" direction.
   - **Comparison chart:** compares the techniques that train models (M5 full fine-tune, M7 LoRA, M8 QLoRA, M10 SFT, M12 DPO) on **metrics that mean the same thing in every module**: `peak_mem_gb`, `tokens_per_sec` and `trainable_pct`, all written by the shared training helpers in `common/`. Per-module task scores are **not** compared, because each module has a different task. When the shared general-capability regression suite from M11 has been run on a module's model, its `regression_score` (same eval set, same scale) is added to the chart. *(Builder's call.)* These four keys are reserved, cross-module metric names. Any module that trains a model must report the first three.
5. **Empty state:** before any record exists, a friendly "Start with M0: run `make doctor`" card.

It polls `/api/state` every 3 seconds. When a newly passed module appears, it briefly highlights it.

**Look:** calm and encouraging, closer to a course companion than a monitoring tool. It supports light and dark mode (following the system setting) and should still read well when the browser window is half the screen width beside a terminal.

### 5.8 Files

```
common/progress.py            # report_metric(), state computation (shared by server + text mode)
common/progress_plugin.py     # pytest plugin writing history records
scripts/progress_server.py    # stdlib HTTP server
dashboard/index.html
dashboard/app.js
dashboard/style.css
dashboard/vendor/<chart lib>  # vendored, pinned version
modules/mNN_*/module.toml     # one per module
.progress/history.jsonl       # gitignored, created on first check
Makefile                      # check, progress targets
```

## 6. Testing

- Unit tests for `common/progress.py`: status rules, next-step selection, malformed-line handling, and metric aggregation, run against fixture history files.
- A pytest plugin test that runs a fake module's tests and asserts that the right record is appended for both pass and fail.
- A server smoke test that starts it on a random port and checks that `/api/state` returns a valid shape.
- A manual check of the page across the empty, partial and complete states, using fixture histories.

## 7. Scope of first build

The dashboard is useless without modules. The first build therefore includes:
- all the dashboard pieces above, and
- `module.toml` manifests for **all 17 modules**, filled from `COURSE.md`, so the course map is complete from day one, and
- one real module wired end to end, **M0 · Setup** (`make doctor` + its checkpoint), to prove the flow.

The other modules' lessons and checkpoints are built later. Their manifests already exist, so they appear as "not started".

## 8. Left to the builder

- The choice of chart library (small and offline-friendly, such as Chart.js or uPlot), pinned and vendored.
- The visual details, within the "calm course companion" direction and with light and dark mode.
- Exact port fallback behaviour and wording of the text output.
- Internal code structure within the listed files.
