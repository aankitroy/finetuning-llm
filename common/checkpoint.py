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


# --- understanding checks (answers.toml) ---


def normalize(raw: str, kind: str) -> str:
    """Canonical form of an answer, so "(32, 4)" and "32x4" count as the same."""
    s = str(raw).strip().lower()
    if kind == "shape":
        s = s.replace("torch.size", "")
        for ch in "()[] ":
            s = s.replace(ch, "")
        s = s.replace(",", "x").replace("×", "x").replace("*", "x")
        return s.strip("x")
    if kind == "number":
        s = s.replace(",", "").replace("_", "").replace(" ", "")
        try:
            v = float(s)
        except ValueError:
            return s
        return str(int(v)) if v.is_integer() else repr(v)
    if kind == "choice":
        s = s.lstrip("( ")
        return s[:1]
    return " ".join(s.split())


def answer_hash(module_id: str, qid: str, normalized: str) -> str:
    return hashlib.sha256(f"ftcourse:{module_id}:{qid}:{normalized}".encode()).hexdigest()


def check_answer(module_dir: Path, module_id: str, qid: str) -> tuple[bool, str]:
    """Compare the learner's answer in answers.toml with the stored hash."""
    import json
    import tomllib

    module_dir = Path(module_dir)
    use_solutions = bool(os.environ.get("FTCOURSE_SOLUTIONS"))
    answers_file = module_dir / ("solutions" if use_solutions else "") / "answers.toml"
    expected = json.loads((module_dir / "tests" / "answer_hashes.json").read_text())[qid]
    given = str(tomllib.loads(answers_file.read_text()).get(qid, {}).get("answer", "")).strip()
    if not given:
        return False, f"{qid}: no answer yet. Open answers.toml and fill in [{qid}] answer."
    if answer_hash(module_id, qid, normalize(given, expected["kind"])) == expected["sha256"]:
        return True, ""
    nn = module_id[1:]
    return False, (
        f"{qid}: not quite. Re-read the question, try it in the lesson notebook, "
        f"or run `make hint M={nn} EX={qid}`."
    )
