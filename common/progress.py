"""Progress history, metric reporting and dashboard state."""

from __future__ import annotations

import glob
import json
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


def append_record(root: Path, record: dict) -> None:
    path = Path(root) / HISTORY_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_history(root: Path) -> tuple[list[dict], list[str]]:
    """Return valid records in file order plus warnings for lines that were skipped."""
    path = Path(root) / HISTORY_PATH
    if not path.exists():
        return [], []
    records: list[dict] = []
    warnings: list[str] = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            warnings.append(f"history line {n}: not valid JSON, skipped")
            continue
        if (
            not isinstance(rec, dict)
            or not isinstance(rec.get("module"), str)
            or not isinstance(rec.get("passed"), bool)
        ):
            warnings.append(f"history line {n}: missing 'module' or 'passed', skipped")
            continue
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
        s = step("empty", first, "Start with M0 · Setup: check your environment.", "make doctor")
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
        return step("finish", m, "You've started this module. Finish it with its checkpoint.", m.next_finish)

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
