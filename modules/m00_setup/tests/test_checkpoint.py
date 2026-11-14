"""M0 checkpoint: your environment is ready. Run with `make doctor`."""

import json
from pathlib import Path

import pytest

from common.progress import report_metric

ROOT = Path(__file__).resolve().parents[3]
DOCTOR = ROOT / "outputs" / "m00" / "doctor.json"


@pytest.fixture(scope="module")
def doctor():
    assert DOCTOR.exists(), "Run `make doctor` first — it writes outputs/m00/doctor.json"
    return json.loads(DOCTOR.read_text())


def test_python_version_ok(doctor):
    assert doctor["python_ok"], f"Python 3.11+ is required, you have {doctor['python']}"


def test_memory_detected(doctor):
    assert doctor["total_mem_gb"] > 0, "Couldn't read total memory"
    report_metric("total_mem_gb", doctor["total_mem_gb"])


@pytest.mark.optional_checkpoint
def test_mps_available(doctor):
    if doctor["mps_available"] is None:
        pytest.skip("PyTorch not installed yet — you'll install it in M1")
    assert doctor["mps_available"], "PyTorch is installed but MPS isn't available"


@pytest.mark.optional_checkpoint
def test_mlx_available(doctor):
    if doctor["mlx_available"] is None:
        pytest.skip("MLX not installed yet — you'll install it in M7")
    assert doctor["mlx_available"]
