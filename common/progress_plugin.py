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
    _state.update(
        passed=set(),
        failed=set(),
        bad_skips=set(),
        collect_errors=0,
        interrupted=False,
        deselected=False,
        skipped=set(),
        failures={},
    )


def pytest_collectreport(report):
    if report.failed:
        _state["collect_errors"] += 1


def pytest_runtest_logreport(report):
    nid = report.nodeid
    if report.failed:
        _state["failed"].add(nid)
        if nid not in _state["failures"]:
            crash = getattr(report.longrepr, "reprcrash", None)
            msg = crash.message if crash else str(report.longrepr)
            _state["failures"][nid] = " ".join(msg.split())[:300]
    elif report.skipped:
        _state["skipped"].add(nid)
        if "optional_checkpoint" not in report.keywords:
            _state["bad_skips"].add(nid)
    elif report.when == "call" and report.passed:
        _state["passed"].add(nid)


def pytest_collection_finish(session):
    _state["collected"] = {item.nodeid for item in session.items}


def pytest_deselected(items):
    # -k, -m, --deselect, --lf and --sw all report here: part of the suite didn't run.
    if items:
        _state["deselected"] = True


def pytest_keyboard_interrupt(excinfo):
    _state["interrupted"] = True


def _full_suite(config) -> bool:
    if config.option.keyword or config.option.markexpr:
        return False
    if config.getoption("ignore", None) or config.getoption("ignore_glob", None):
        return False
    # --lf / --sw rerun only part of the suite (--lf does so without reporting deselection).
    if config.getoption("lf", False) or config.getoption("stepwise", False):
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
    used_solutions = bool(os.environ.get("FTCOURSE_SOLUTIONS"))
    ok = (
        not used_solutions  # the reference solutions never count as your progress
        and _full_suite(session.config)
        and not _state["deselected"]
        and _state.get("collected", set()) <= (passed | failed | _state["skipped"])
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
        "used_solutions": used_solutions,
        "failures": [
            {"test": nid.split("::")[-1], "message": msg}
            for nid, msg in list(_state["failures"].items())[:50]
        ],
    }
    append_record(Path(os.environ.get("FTCOURSE_ROOT") or os.getcwd()), record)
