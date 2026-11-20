import os
import sys
from pathlib import Path

import pytest

from common.checkpoint import load_exercise

MODULE_DIR = Path(__file__).resolve().parent.parent
ROOT = MODULE_DIR.parent.parent
sys.path.insert(0, str(MODULE_DIR))  # for tiny_model.py


@pytest.fixture(scope="session")
def ex():
    return load_exercise(MODULE_DIR)


@pytest.fixture(scope="session")
def outputs():
    return Path(os.environ.get("FTCOURSE_OUTPUTS") or ROOT / "outputs") / "m02"


@pytest.fixture(scope="session")
def tiny():
    from tiny_model import make_tiny_model

    return make_tiny_model()
