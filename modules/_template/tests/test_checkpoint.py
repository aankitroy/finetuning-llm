"""MNN checkpoint. Checks verify results; they don't retrain."""

import json
import os
from pathlib import Path

import pytest

from common.checkpoint import check_answer, load_exercise
from common.progress import report_metric

MODULE_DIR = Path(__file__).resolve().parent.parent
ROOT = MODULE_DIR.parent.parent
OUTPUTS = Path(os.environ.get("FTCOURSE_OUTPUTS") or ROOT / "outputs") / "mNN"
ex = load_exercise(MODULE_DIR)


def test_example():
    assert ex.example(3) == 3, "example(x) should return x"


@pytest.mark.parametrize(
    "qid", sorted(json.loads((MODULE_DIR / "tests" / "answer_hashes.json").read_text()))
)
def test_understanding(qid):
    ok, message = check_answer(MODULE_DIR, "mNN", qid)
    assert ok, message


def test_run_summary():
    summary = OUTPUTS / "run_summary.json"
    assert summary.exists(), "Run the training step first — it writes outputs/mNN/run_summary.json"
    s = json.loads(summary.read_text())
    report_metric("final_loss", s["final_loss"])
    assert s["final_loss"] < 2.0, f"Final loss {s['final_loss']:.2f} should be below 2.0"
