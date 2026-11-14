"""Progress history, metric reporting and dashboard state."""

from __future__ import annotations

import glob
import json
import math
import os
from pathlib import Path

from common.course import Module, load_modules

HISTORY_PATH = Path(".progress/history.jsonl")

_metrics: dict[str, float] = {}


# --- metric reporting (used by checkpoint tests, drained by the pytest plugin) ---


def report_metric(key: str, value: float) -> None:
    """Record a number for the current checkpoint run (shown on the dashboard)."""
    _metrics[str(key)] = float(value)


def collected_metrics() -> dict[str, float]:
    return dict(_metrics)


def reset_metrics() -> None:
    _metrics.clear()


# --- history I/O ---


def _is_num(v) -> bool:
    return isinstance(v, int | float) and not isinstance(v, bool)


def _finite_metrics(metrics: dict) -> dict:
    """NaN/Infinity (e.g. a diverged loss) become None so the history stays strict JSON."""
    return {k: (v if not _is_num(v) or math.isfinite(v) else None) for k, v in metrics.items()}


def append_record(root: Path, record: dict) -> None:
    path = Path(root) / HISTORY_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    record = dict(record)
    if isinstance(record.get("metrics"), dict):
        record["metrics"] = _finite_metrics(record["metrics"])
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _shape_problem(rec) -> str | None:
    if not isinstance(rec, dict):
        return "not a JSON object"
    if not isinstance(rec.get("module"), str) or not isinstance(rec.get("passed"), bool):
        return "missing 'module' or 'passed'"
    tests = rec.get("tests", {})
    if not isinstance(tests, dict) or not all(_is_num(v) for v in tests.values()):
        return "'tests' must be an object of numbers"
    metrics = rec.get("metrics", {})
    if metrics is None:
        metrics = {}
    if not isinstance(metrics, dict) or not all(v is None or _is_num(v) for v in metrics.values()):
        return "'metrics' must be an object of numbers"
    return None


def read_history(root: Path) -> tuple[list[dict], list[str]]:
    """Return valid records in file order plus warnings for lines that were skipped."""
    path = Path(root) / HISTORY_PATH
    if not path.exists():
        return [], []
    records: list[dict] = []
    warnings: list[str] = []
    for n, raw in enumerate(path.read_bytes().split(b"\n"), start=1):
        if not raw.strip():
            continue
        try:
            rec = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            warnings.append(f"history line {n}: not valid JSON, skipped")
            continue
        problem = _shape_problem(rec)
        if problem:
            warnings.append(f"history line {n}: {problem}, skipped")
            continue
        if isinstance(rec.get("metrics"), dict):
            rec["metrics"] = _finite_metrics(rec["metrics"])
        records.append(rec)
    return records, warnings


# --- dashboard state ---


def _has_artifact(root: Path, patterns: list[str]) -> bool:
    return any(glob.glob(str(Path(root) / p)) for p in patterns)


def _next_step(modules: list[Module], status: dict[str, str], attempts: dict, any_activity: bool):
    def step(kind, m, instruction, command):
        return {
            "kind": kind,
            "module": m.id if m else None,
            "title": f"{m.id.upper()} · {m.title}" if m else "",
            "instruction": instruction,
            "command": command,
            "why": m.next_why if m else "",
        }

    if not any_activity:
        first = modules[0] if modules else None
        s = step(
            "empty",
            first,
            "Check that your Mac and Python are ready for the course.",
            "make doctor",
        )
        s["title"] = "Start with M0 · Setup"
        return s

    in_progress = [m for m in modules if status[m.id] == "in_progress"]
    if in_progress:
        m = in_progress[0]
        tries = attempts[m.id]
        if tries:
            last = tries[-1].get("tests") or {}
            failed = int(last.get("failed", 0))
            total = failed + int(last.get("passed", 0))
            msg = f"Your last check failed: {failed} of {total} tests. Fix them and re-run."
            return step("retry", m, msg, m.next_finish)
        return step(
            "finish", m, "You've started this module. Finish it with its checkpoint.", m.next_finish
        )

    todo = [m for m in modules if status[m.id] == "not_started"]
    if todo:
        m = todo[0]
        return step("start", m, m.next_start, m.next_finish)

    s = step("complete", None, "Every module is done. Pick a capstone track and ship it.", None)
    s["title"] = "Course complete 🎉"
    return s


def compute_state(root: Path) -> dict:
    """Combine manifests, history and artifacts into the dashboard state."""
    root = Path(root)
    modules, warnings = load_modules(root)
    records, hist_warnings = read_history(root)
    warnings += hist_warnings

    known = {m.id for m in modules}
    unknown = sorted({r["module"] for r in records} - known)
    for mid in unknown:
        warnings.append(f"history has records for unknown module {mid!r}, ignored")
    history = [r for r in records if r["module"] in known]

    attempts: dict[str, list[dict]] = {m.id: [] for m in modules}
    for r in history:
        attempts[r["module"]].append(r)

    status: dict[str, str] = {}
    artifact_seen = False
    for m in modules:
        has_art = _has_artifact(root, m.artifacts)
        artifact_seen = artifact_seen or has_art
        if any(r["passed"] for r in attempts[m.id]):
            status[m.id] = "done"
        elif attempts[m.id] or has_art:
            status[m.id] = "in_progress"
        else:
            status[m.id] = "not_started"

    done = [m for m in modules if status[m.id] == "done"]
    return {
        "course": {
            "total": len(modules),
            "done": len(done),
            "est_minutes_done": sum(m.est_minutes for m in done),
        },
        "modules": [
            {
                "id": m.id,
                "title": m.title,
                "part": m.part,
                "order": m.order,
                "summary": m.summary,
                "status": status[m.id],
                "latest_metrics": dict(attempts[m.id][-1].get("metrics") or {})
                if attempts[m.id]
                else {},
                "attempts": attempts[m.id],
                "metrics": m.metrics,
            }
            for m in modules
        ],
        "next": _next_step(modules, status, attempts, bool(history) or artifact_seen),
        "history": history,
        "warnings": warnings,
    }


# --- terminal rendering (`make progress TEXT=1`) ---

_ICON = {"done": "✔", "in_progress": "◐", "not_started": "○"}


def render_text(state: dict) -> str:
    c = state["course"]
    total = c["total"] or 1
    filled = round(20 * c["done"] / total)
    lines = [
        f"LLM Fine-Tuning · progress   {'█' * filled}{'░' * (20 - filled)}  "
        f"{c['done']} / {c['total']} modules",
        "",
    ]
    part = None
    for m in state["modules"]:
        if m["part"] != part:
            part = m["part"]
            lines.append(f" {part}")
        lines.append(f"   {_ICON[m['status']]} {m['id'].upper()} {m['title']}")
    n = state["next"]
    lines += ["", " ┌ NEXT " + "─" * 50, f" │ {n['title']}", f" │ {n['instruction']}"]
    if n["command"]:
        lines.append(f" │ $ {n['command']}")
    if n["why"]:
        lines.append(f" │ Why: {n['why']}")
    lines.append(" └" + "─" * 56)
    for w in state["warnings"]:
        lines.append(f" ! {w}")
    return "\n".join(lines)
