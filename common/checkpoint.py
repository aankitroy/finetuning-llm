"""Helpers shared by module checkpoint tests."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType


def load_exercise(module_dir: Path, name: str = "exercises") -> ModuleType:
    """Import the learner's ``<name>.py`` from a module folder.

    With ``FTCOURSE_SOLUTIONS=1`` the reference ``solutions/<name>.py`` is loaded
    instead (used by the project's own tests to prove each checkpoint is passable).
    """
    module_dir = Path(module_dir).resolve()
    use_solutions = bool(os.environ.get("FTCOURSE_SOLUTIONS"))
    path = module_dir / ("solutions" if use_solutions else "") / f"{name}.py"
    # A unique import name per file, so modules with the same filename never clash.
    unique = "ftcourse_" + hashlib.sha1(str(path).encode()).hexdigest()[:12]
    spec = importlib.util.spec_from_file_location(unique, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[unique] = mod
    spec.loader.exec_module(mod)
    return mod
