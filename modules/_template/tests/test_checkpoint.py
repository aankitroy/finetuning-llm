"""MNN checkpoint. Checks verify results; they don't retrain."""

import json
from pathlib import Path

from common.progress import report_metric

ROOT = Path(__file__).resolve().parents[3]
SUMMARY = ROOT / "outputs" / "mNN" / "run_summary.json"


def test_run_summary_exists():
    assert SUMMARY.exists(), "Run the training step first — it writes outputs/mNN/run_summary.json"


def test_loss_below_target():
    s = json.loads(SUMMARY.read_text())
    report_metric("final_loss", s["final_loss"])
    assert s["final_loss"] < 2.0, f"Final loss {s['final_loss']:.2f} should be below 2.0"
