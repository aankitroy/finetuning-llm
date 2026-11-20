"""Each built module's checkpoint must pass with its reference solutions.

Runs in a temp outputs dir, so the learner's real outputs/ and progress are untouched.
Skipped when the course dependencies (PyTorch) aren't installed.
"""

import os
import subprocess
import sys

import pytest

from tests.conftest import REPO_ROOT

MODULES = [
    ("m01_pytorch_essentials", ["train.py", "--tiny"]),
    ("m02_how_llms_work", ["generate.py", "--tiny"]),
]


@pytest.mark.parametrize("slug,train", MODULES)
def test_checkpoint_passes_with_solutions(slug, train, tmp_path):
    pytest.importorskip("torch")
    env = {
        **os.environ,
        "FTCOURSE_SOLUTIONS": "1",
        "FTCOURSE_OUTPUTS": str(tmp_path),
        "PYTHONPATH": str(REPO_ROOT),
    }
    env.pop("FTCOURSE_RECORD", None)
    module_dir = REPO_ROOT / "modules" / slug
    run = subprocess.run(
        [sys.executable, str(module_dir / train[0]), *train[1:]],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    check = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(module_dir / "tests")],
        env=env,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=120,
    )
    assert check.returncode == 0, check.stdout + check.stderr
