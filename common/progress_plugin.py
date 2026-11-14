"""pytest plugin that records one checkpoint attempt per `make check` run.

Inert unless ``FTCOURSE_RECORD=<NN>`` is set (the Makefile sets it). A run counts
as passed only when the module's full checkpoint suite ran with at least one
test, nothing failed/errored/was skipped (except ``optional_checkpoint`` tests),
there were no collection errors, and the session wasn't interrupted.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from common.progress import append_record, collected_metrics, reset_metrics

_state: dict = {}


def _active() -> bool:
    return bool(os.environ.get("FTCOURSE_RECORD"))


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "optional_checkpoint: a skip here doesn't fail the checkpoint"
    )
    reset_metrics()
    _state.clear()
    _state.update(passed=set(), failed=set(), bad_skips=set(), collect_errors=0, interrupted=False)


def pytest_collectreport(report):
    if report.failed:
        _state["collect_errors"] += 1


def pytest_runtest_logreport(report):
    nid = report.nodeid
    if report.failed:
        _state["failed"].add(nid)
    elif report.skipped and "optional_checkpoint" not in report.keywords:
        _state["bad_skips"].add(nid)
    elif report.when == "call" and report.passed:
        _state["passed"].add(nid)


def pytest_keyboard_interrupt(excinfo):
    _state["interrupted"] = True


def _full_suite(config) -> bool:
    if config.option.keyword or config.option.markexpr:
        return False
    for arg in config.args:
        if "::" in arg or not Path(arg).is_dir():
            return False
    return True


def pytest_sessionfinish(session, exitstatus):
    if not _active():
        return
    passed = _state["passed"] - _state["failed"]
    failed = _state["failed"]
    ok = (
        _full_suite(session.config)
        and len(passed) > 0
        and not failed
        and not _state["bad_skips"]
        and _state["collect_errors"] == 0
        and not _state["interrupted"]
        and int(exitstatus) == 0
    )
    record = {
        "v": 1,
        "ts": datetime.now().astimezone().isoformat(timespec="seconds"),
        "module": "m" + os.environ["FTCOURSE_RECORD"],
        "passed": ok,
        "tests": {"passed": len(passed), "failed": len(failed)},
        "metrics": collected_metrics(),
        "run_label": os.environ.get("FTCOURSE_LABEL") or None,
    }
    append_record(Path(os.environ.get("FTCOURSE_ROOT") or os.getcwd()), record)
