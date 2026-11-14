import os
import sys
from pathlib import Path

import pytest

from common.checkpoint import load_exercise

MODULE_DIR = Path(__file__).resolve().parent.parent
ROOT = MODULE_DIR.parent.parent
sys.path.insert(0, str(MODULE_DIR))  # for data.py


@pytest.fixture(scope="session")
def ex():
    return load_exercise(MODULE_DIR)


@pytest.fixture(scope="session")
def outputs():
    return Path(os.environ.get("FTCOURSE_OUTPUTS") or ROOT / "outputs") / "m01"
