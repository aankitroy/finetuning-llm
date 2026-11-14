"""Progress history, metric reporting and dashboard state."""

from __future__ import annotations

import json
import os
from pathlib import Path

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
