import json

import pytest

from common.course import load_modules
from common.guide import module_guide
from common.progress import append_record, compute_state

MANIFEST = """id = "m05"
title = "T"
part = "B"
order = 5
summary = "s"
est_minutes = 1
[next]
start = "a"
finish = "make check M=05"
why = "w"
[[steps]]
kind = "lesson"
title = "Run the lesson"
detail = "d"
command = "uv run jupyter lab modules/m05_x/lesson.ipynb"
file = "lesson.ipynb"
[[steps]]
kind = "exercises"
title = "Write the exercises"
detail = "d"
file = "exercises.py"
[[steps]]
kind = "answers"
title = "Answer the questions"
detail = "d"
file = "answers.toml"
[[steps]]
kind = "train"
title = "Train"
detail = "d"
command = "uv run python modules/m05_x/train.py"
file = "outputs/m05/run_summary.json"
metric = "val_accuracy"
[[steps]]
kind = "checkpoint"
title = "Pass the checkpoint"
detail = "d"
command = "make check M=05"
"""
HINTS = """
[[exercise]]
id = "1"
name = "add"
test = "test_add"
hint = "h"
[[exercise]]
id = "2"
name = "Model"
test = "test_model"
hint = "h"
[[exercise]]
id = "q1"
test = "test_understanding[q1]"
hint = "h"
[[exercise]]
id = "q2"
test = "test_understanding[q2]"
hint = "h"
"""
STUBS = '''"""Exercises."""


def add(a, b):
    """Add."""
    raise NotImplementedError("Exercise 1: add")


class Model:
    def __init__(self):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError("Exercise 2")
'''


def nb(exec_counts):
    cells = [{"cell_type": "markdown", "source": "# hi", "metadata": {}}]
    cells += [
        {"cell_type": "code", "source": "1", "metadata": {}, "outputs": [], "execution_count": c}
        for c in exec_counts
    ]
    return json.dumps({"cells": cells, "metadata": {}, "nbformat": 4, "nbformat_minor": 5})


@pytest.fixture
def root(tmp_path):
    d = tmp_path / "modules" / "m05_x"
    d.mkdir(parents=True)
    (d / "module.toml").write_text(MANIFEST)
    (d / "hints.toml").write_text(HINTS)
    (d / "lesson.ipynb").write_text(nb([None, None, None]))
    (d / "exercises.py").write_text(STUBS)
    (d / "answers.toml").write_text(
        '[q1]\nquestion = "First?\\n  a) x"\nanswer = ""\n[q2]\nquestion = "Second?"\nanswer = ""\n'
    )
    return tmp_path


def guide(root):
    (m,), _ = load_modules(root)
    s = compute_state(root)
    attempts = next(x for x in s["modules"] if x["id"] == "m05")["attempts"]
    return module_guide(root, m, attempts)


def step(g, kind):
    return next(s for s in g["steps"] if s["kind"] == kind)


def test_fresh_module_all_todo(root):
    g = guide(root)
    assert [s["status"] for s in g["steps"]] == ["todo"] * 5
    assert g["done"] == 0 and g["total"] == 5
    assert step(g, "lesson")["progress"] == "0 of 3 cells run"
    assert step(g, "exercises")["progress"] == "0 of 2 written"
    assert step(g, "answers")["progress"] == "0 of 2 answered"
    assert step(g, "lesson")["command"].startswith("uv run jupyter lab")


def test_lesson_progress(root):
    (root / "modules/m05_x/lesson.ipynb").write_text(nb([1, None, 3]))
    s = step(guide(root), "lesson")
    assert s["status"] == "in_progress" and s["progress"] == "2 of 3 cells run"
    (root / "modules/m05_x/lesson.ipynb").write_text(nb([1, 2, 3]))
    assert step(guide(root), "lesson")["status"] == "done"


def test_exercise_written_detection(root):
    f = root / "modules/m05_x/exercises.py"
    f.write_text(
        STUBS.replace('    raise NotImplementedError("Exercise 1: add")', "    return a + b")
    )
    s = step(guide(root), "exercises")
    assert s["status"] == "in_progress" and s["progress"] == "1 of 2 written"
    items = {i["id"]: i for i in s["items"]}
    assert items["1"]["written"] is True and items["2"]["written"] is False
    assert items["2"]["name"] == "Model" and items["2"]["hint"] == "make hint M=05 EX=2"


def test_class_counts_as_written_only_when_no_method_is_a_stub(root):
    f = root / "modules/m05_x/exercises.py"
    f.write_text(
        STUBS.replace(
            "    def __init__(self):\n        raise NotImplementedError\n",
            "    def __init__(self):\n        self.w = 1\n",
        )
    )
    items = {i["id"]: i for i in step(guide(root), "exercises")["items"]}
    assert items["2"]["written"] is False


def test_syntax_error_is_reported_not_crashed(root):
    (root / "modules/m05_x/exercises.py").write_text("def add(:\n")
    s = step(guide(root), "exercises")
    assert "syntax error" in s["note"] and "line 1" in s["note"]
    assert s["status"] == "in_progress"


def test_last_check_marks_items_pass_and_fail(root):
    f = root / "modules/m05_x/exercises.py"
    f.write_text(
        "def add(a, b):\n    return a + b\n\nclass Model:\n    def forward(self, x):\n        return x\n"
    )
    append_record(
        root,
        {
            "v": 1,
            "module": "m05",
            "passed": False,
            "failures": [
                {"test": "test_model", "message": "AssertionError: two layers"},
                {"test": "test_understanding[q2]", "message": "q2: not quite"},
            ],
        },
    )
    g = guide(root)
    ex = {i["id"]: i for i in step(g, "exercises")["items"]}
    assert ex["1"]["last"] == "pass" and ex["2"]["last"] == "fail"
    assert ex["2"]["message"] == "AssertionError: two layers"
    assert step(g, "exercises")["status"] == "in_progress"
    assert step(g, "exercises")["progress"] == "2 of 2 written, 1 passing at your last check"
    qs = {i["id"]: i for i in step(g, "answers")["items"]}
    assert qs["q1"]["last"] == "pass" and qs["q2"]["last"] == "fail"
    assert qs["q1"]["question"] == "First?"


def test_exercises_done_when_written_and_passing(root):
    (root / "modules/m05_x/exercises.py").write_text(
        "def add(a, b):\n    return a + b\n\nclass Model:\n    def forward(self, x):\n        return x\n"
    )
    append_record(
        root,
        {
            "v": 1,
            "module": "m05",
            "passed": False,
            "failures": [{"test": "test_trained", "message": "no run"}],
        },
    )
    assert step(guide(root), "exercises")["status"] == "done"


def test_answers_progress(root):
    (root / "modules/m05_x/answers.toml").write_text(
        '[q1]\nquestion = "First?"\nanswer = "b"\n[q2]\nquestion = "Second?"\nanswer = " "\n'
    )
    s = step(guide(root), "answers")
    assert s["status"] == "in_progress" and s["progress"] == "1 of 2 answered"


def test_train_step(root):
    out = root / "outputs/m05"
    out.mkdir(parents=True)
    (out / "run_summary.json").write_text('{"val_accuracy": 0.97}')
    s = step(guide(root), "train")
    assert s["status"] == "done" and s["value"] == 0.97 and s["metric"] == "val_accuracy"
    (out / "run_summary.json").write_text('{"val_accuracy": 0.97, "used_solutions": true}')
    s = step(guide(root), "train")
    assert s["status"] == "in_progress" and "reference solutions" in s["note"]


def test_checkpoint_step(root):
    append_record(
        root, {"v": 1, "module": "m05", "passed": False, "tests": {"passed": 9, "failed": 5}}
    )
    s = step(guide(root), "checkpoint")
    assert s["status"] == "in_progress" and s["progress"] == "Last check: 9 of 14 tests passed"
    append_record(
        root, {"v": 1, "module": "m05", "passed": True, "tests": {"passed": 14, "failed": 0}}
    )
    g = guide(root)
    assert step(g, "checkpoint")["status"] == "done"


def test_state_includes_guide_only_for_modules_with_steps(root, tmp_path):
    s = compute_state(root)
    assert s["modules"][0]["guide"]["total"] == 5
    d = root / "modules" / "m06_y"
    d.mkdir()
    (d / "module.toml").write_text(
        MANIFEST.split("[[steps]]")[0].replace('"m05"', '"m06"').replace("order = 5", "order = 6")
    )
    assert next(m for m in compute_state(root)["modules"] if m["id"] == "m06")["guide"] is None


def test_missing_files_dont_crash(root):
    for f in ("lesson.ipynb", "exercises.py", "answers.toml", "hints.toml"):
        (root / "modules/m05_x" / f).unlink()
    g = guide(root)
    assert step(g, "lesson")["status"] == "todo" and "not found" in step(g, "lesson")["note"]


def test_hint_test_names_exist_in_checkpoints():
    import ast as _ast
    import tomllib

    from tests.conftest import REPO_ROOT

    for hints in sorted(REPO_ROOT.glob("modules/m*/hints.toml")):
        names = set()
        for f in (hints.parent / "tests").glob("test_*.py"):
            names |= {
                n.name for n in _ast.parse(f.read_text()).body if isinstance(n, _ast.FunctionDef)
            }
        for e in tomllib.loads(hints.read_text())["exercise"]:
            assert "test" in e, f"{hints}: exercise {e['id']} has no test"
            assert e["test"].split("[")[0] in names, f"{hints}: {e['test']} not found"


def test_real_m01_guide_has_five_steps():
    from tests.conftest import REPO_ROOT

    mods, _ = load_modules(REPO_ROOT)
    m01 = next(m for m in mods if m.id == "m01")
    assert [s["kind"] for s in m01.steps] == [
        "lesson",
        "exercises",
        "answers",
        "train",
        "checkpoint",
    ]


def test_render_text_lists_steps_for_the_next_module(root):
    from common.progress import render_text

    out = render_text(compute_state(root))
    assert "Steps for M05" in out
    assert "1. ○ Run the lesson" in out and "0 of 3 cells run" in out
    assert "$ uv run jupyter lab" in out
