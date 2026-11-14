import json

import pytest

from tests.conftest import REPO_ROOT

pytest_plugins = ["pytester"]

TESTS_DIR = "modules/m05_x/tests"


@pytest.fixture
def run(pytester, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", str(REPO_ROOT))
    monkeypatch.delenv("FTCOURSE_LABEL", raising=False)

    def _run(files: dict[str, str], *extra, env=True, target=TESTS_DIR):
        d = pytester.path / TESTS_DIR
        d.mkdir(parents=True, exist_ok=True)
        for name, body in files.items():
            (d / name).write_text(body)
        if env:
            monkeypatch.setenv("FTCOURSE_RECORD", "05")
            monkeypatch.setenv("FTCOURSE_ROOT", str(pytester.path))
        else:
            monkeypatch.delenv("FTCOURSE_RECORD", raising=False)
        pytester.runpytest_subprocess("-p", "common.progress_plugin", target, *extra)
        hist = pytester.path / ".progress" / "history.jsonl"
        if not hist.exists():
            return None
        lines = hist.read_text().splitlines()
        assert len(lines) == 1, "exactly one record per session"
        return json.loads(lines[0])

    return _run


TWO_PASS = {"test_a.py": "def test_one():\n    pass\n\ndef test_two():\n    pass\n"}


def test_inert_without_env(run):
    assert run(TWO_PASS, env=False) is None


def test_all_pass_records_passed_true(run):
    rec = run(TWO_PASS)
    assert rec["v"] == 1 and rec["module"] == "m05" and rec["passed"] is True
    assert rec["tests"] == {"passed": 2, "failed": 0}
    assert rec["run_label"] is None and rec["metrics"] == {}
    assert "T" in rec["ts"] and ("+" in rec["ts"] or "-" in rec["ts"][10:])


def test_failure_records_passed_false(run):
    rec = run({"test_a.py": "def test_one():\n    pass\n\ndef test_two():\n    assert False\n"})
    assert rec["passed"] is False and rec["tests"] == {"passed": 1, "failed": 1}


def test_error_in_fixture_counts_as_failed(run):
    body = (
        "import pytest\n@pytest.fixture\ndef boom():\n    raise RuntimeError\n"
        "def test_one(boom):\n    pass\n"
    )
    rec = run({"test_a.py": body})
    assert rec["passed"] is False and rec["tests"]["failed"] == 1


def test_skip_makes_it_fail(run):
    body = (
        "import pytest\ndef test_one():\n    pass\n@pytest.mark.skip\ndef test_two():\n    pass\n"
    )
    assert run({"test_a.py": body})["passed"] is False


def test_optional_checkpoint_skip_ok(run):
    body = (
        "import pytest\ndef test_one():\n    pass\n"
        "@pytest.mark.optional_checkpoint\ndef test_two():\n    pytest.skip('not installed')\n"
    )
    rec = run({"test_a.py": body})
    assert rec["passed"] is True and rec["tests"] == {"passed": 1, "failed": 0}


def test_keyword_filter_never_passes(run):
    assert run(TWO_PASS, "-k", "one")["passed"] is False


def test_marker_filter_never_passes(run):
    assert run(TWO_PASS, "-m", "slow")["passed"] is False


def test_nodeid_narrowing_never_passes(run):
    assert run(TWO_PASS, target=f"{TESTS_DIR}/test_a.py::test_one")["passed"] is False


def test_file_narrowing_never_passes(run):
    files = {**TWO_PASS, "test_b.py": "def test_three():\n    pass\n"}
    assert run(files, target=f"{TESTS_DIR}/test_a.py")["passed"] is False


def test_zero_tests_never_passes(run):
    rec = run({"test_a.py": "x = 1\n"})
    assert rec["passed"] is False and rec["tests"] == {"passed": 0, "failed": 0}


def test_collection_error_records_false(run):
    rec = run({**TWO_PASS, "test_b.py": "import does_not_exist\n"})
    assert rec["passed"] is False


def test_interrupt_records_false(run):
    rec = run({"test_a.py": "def test_one():\n    raise KeyboardInterrupt\n"})
    assert rec["passed"] is False


def test_metrics_and_label_recorded(run, monkeypatch):
    monkeypatch.setenv("FTCOURSE_LABEL", "r=8")
    body = (
        "from common.progress import report_metric\n"
        "def test_one():\n    report_metric('final_loss', 1.2)\n"
    )
    rec = run({"test_a.py": body})
    assert rec["metrics"] == {"final_loss": 1.2} and rec["run_label"] == "r=8"


def test_deselect_never_passes(run, monkeypatch):
    monkeypatch.setenv("PYTEST_ADDOPTS", f"--deselect {TESTS_DIR}/test_a.py::test_two")
    assert run(TWO_PASS)["passed"] is False


def test_ignore_never_passes(run):
    files = {**TWO_PASS, "test_b.py": "def test_three():\n    pass\n"}
    assert run(files, f"--ignore={TESTS_DIR}/test_b.py")["passed"] is False


def test_last_failed_rerun_never_passes(run, pytester):
    run({"test_a.py": "def test_one():\n    pass\n\ndef test_two():\n    assert False\n"})
    (pytester.path / ".progress" / "history.jsonl").unlink()
    rec = run({"test_a.py": "def test_one():\n    pass\n\ndef test_two():\n    pass\n"}, "--lf")
    assert rec["passed"] is False


def test_stepwise_never_passes(run):
    assert run(TWO_PASS, "--sw")["passed"] is False


def test_solutions_mode_never_passes(run, monkeypatch):
    monkeypatch.setenv("FTCOURSE_SOLUTIONS", "1")
    rec = run(TWO_PASS)
    assert rec["passed"] is False and rec["used_solutions"] is True


def test_normal_run_records_used_solutions_false(run, monkeypatch):
    monkeypatch.delenv("FTCOURSE_SOLUTIONS", raising=False)
    assert run(TWO_PASS)["used_solutions"] is False
