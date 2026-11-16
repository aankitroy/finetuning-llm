import json
import sys

import pytest

from common.progress import append_record, compute_state, read_history, read_reveals
from tests.conftest import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "scripts"))
import course  # noqa: E402

MANIFEST = (
    'id = "m{nn}"\ntitle = "T{nn}"\npart = "A"\norder = {order}\nsummary = "s"\nest_minutes = 1\n'
    '[next]\nstart = "a"\nfinish = "make check M={nn}"\nwhy = "w"\n'
)
STUB = "def add(a, b):\n    raise NotImplementedError\n"
HINTS = """
[[exercise]]
id = "1"
name = "add"
hint = "Use the + operator."

[[exercise]]
id = "q1"
hint = "Think about shapes."
"""
SOLUTION = '''"""Solutions."""


def helper():
    return 0


def add(a, b):
    return a + b
'''
ANSWERS_STARTER = '[q1]\nquestion = "2 + 2?"\nanswer = ""\n'
ANSWERS_SOLUTION = '[q1]\nquestion = "2 + 2?"\nanswer = "4"\n'


@pytest.fixture
def root(tmp_path):
    for order, nn in enumerate(["05", "06"]):
        d = tmp_path / "modules" / f"m{nn}_x"
        (d / "starter").mkdir(parents=True)
        (d / "solutions").mkdir()
        (d / "module.toml").write_text(MANIFEST.format(nn=nn, order=order))
        (d / "hints.toml").write_text(HINTS)
        (d / "starter" / "exercises.py").write_text(STUB)
        (d / "starter" / "answers.toml").write_text(ANSWERS_STARTER)
        (d / "exercises.py").write_text("def add(a, b):\n    return a + b  # my work\n")
        (d / "answers.toml").write_text(ANSWERS_SOLUTION)
        (d / "solutions" / "exercises.py").write_text(SOLUTION)
        (d / "solutions" / "answers.toml").write_text(ANSWERS_SOLUTION)
        (tmp_path / "outputs" / f"m{nn}").mkdir(parents=True)
        (tmp_path / "outputs" / f"m{nn}" / "run_summary.json").write_text("{}")
        append_record(tmp_path, {"v": 1, "module": f"m{nn}", "passed": True})
    return tmp_path


# --- hints and solutions ---


def test_hint_shows_text_and_is_recorded(root):
    out = course.show_hint(root, "05", "1")
    assert "Use the + operator." in out
    assert [(r["module"], r["exercise"], r["kind"]) for r in read_reveals(root)[0]] == [
        ("m05", "1", "hint")
    ]


def test_solution_for_exercise_shows_only_that_function(root):
    out = course.show_solution(root, "05", "1")
    assert "def add(a, b):" in out and "return a + b" in out
    assert "def helper" not in out
    assert read_reveals(root)[0][-1]["kind"] == "solution"


def test_solution_for_question_shows_answer(root):
    out = course.show_solution(root, "05", "q1")
    assert "4" in out
    assert read_reveals(root)[0][-1]["exercise"] == "q1"


def test_unknown_exercise_is_an_error_and_not_recorded(root):
    with pytest.raises(course.CourseError) as e:
        course.show_hint(root, "05", "9")
    assert "1, q1" in str(e.value)
    assert read_reveals(root)[0] == []


def test_unknown_module_is_an_error(root):
    with pytest.raises(course.CourseError):
        course.show_hint(root, "99", "1")


def test_dashboard_counts_reveals(root):
    course.show_hint(root, "05", "1")
    course.show_hint(root, "05", "q1")
    course.show_solution(root, "05", "1")
    m = next(m for m in compute_state(root)["modules"] if m["id"] == "m05")
    assert m["reveals"] == {"hints": 2, "solutions": 1}


# --- reset ---


def test_reset_plan_lists_what_will_go(root):
    plan = "\n".join(course.reset_plan(root, None))
    assert "2 progress records" in plan
    assert "outputs/" in plan
    assert "modules/m05_x/exercises.py" in plan and "modules/m05_x/answers.toml" in plan


def test_reset_everything(root):
    course.show_hint(root, "05", "1")
    course.do_reset(root, None)
    assert not (root / ".progress").exists()
    assert not (root / "outputs").exists()
    for nn in ("05", "06"):
        d = root / "modules" / f"m{nn}_x"
        assert (d / "exercises.py").read_text() == STUB
        assert (d / "answers.toml").read_text() == ANSWERS_STARTER
        assert (d / "solutions" / "exercises.py").exists(), "solutions are never touched"
    s = compute_state(root)
    assert s["course"]["done"] == 0 and s["next"]["kind"] == "empty"


def test_reset_one_module_keeps_the_others(root):
    course.show_hint(root, "05", "1")
    course.show_hint(root, "06", "1")
    course.do_reset(root, "05")
    assert [r["module"] for r in read_history(root)[0]] == ["m06"]
    assert [r["module"] for r in read_reveals(root)[0]] == ["m06"]
    assert not (root / "outputs" / "m05").exists() and (root / "outputs" / "m06").exists()
    assert (root / "modules" / "m05_x" / "exercises.py").read_text() == STUB
    assert "my work" in (root / "modules" / "m06_x" / "exercises.py").read_text()


def test_reset_on_a_fresh_course_is_harmless(tmp_path):
    assert course.reset_plan(tmp_path, None) == []
    course.do_reset(tmp_path, None)


def test_main_reset_requires_typing_reset(root, monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda prompt="": "yes")
    assert course.main(["--root", str(root), "reset"]) == 1
    assert (root / ".progress").exists(), "anything but 'reset' cancels"
    monkeypatch.setattr("builtins.input", lambda prompt="": "reset")
    assert course.main(["--root", str(root), "reset"]) == 0
    assert not (root / ".progress").exists()


def test_reset_restores_a_starter_notebook(root):
    d = root / "modules" / "m05_x"
    (d / "starter" / "lesson.ipynb").write_text('{"cells": []}')
    (d / "lesson.ipynb").write_text('{"cells": [{"execution_count": 3}]}')
    assert any("modules/m05_x/lesson.ipynb" in line for line in course.reset_plan(root, "05"))
    course.do_reset(root, "05")
    assert (d / "lesson.ipynb").read_text() == '{"cells": []}'


def test_main_reset_says_notebooks_are_reset_too(root, monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda prompt="": "no")
    course.main(["--root", str(root), "reset"])
    out = capsys.readouterr().out
    assert "Lesson notebooks" not in out
    assert "Reference solutions are kept." in out


def test_main_hint_unknown_exercise_exits_2(root, capsys):
    assert course.main(["--root", str(root), "hint", "05", "9"]) == 2
    assert "1, q1" in capsys.readouterr().err


def test_reveal_record_shape(root):
    course.show_hint(root, "05", "1")
    raw = (root / ".progress" / "reveals.jsonl").read_text().splitlines()
    rec = json.loads(raw[0])
    assert set(rec) == {"v", "ts", "module", "exercise", "kind"}


def test_build_answer_hashes_matches_checker(root):
    from common.checkpoint import answer_hash

    d = root / "modules" / "m05_x"
    (d / "solutions" / "answers.toml").write_text('[q1]\nkind = "number"\nanswer = "4,000"\n')
    hashes = course.build_answer_hashes(root, "05")
    assert hashes == {"q1": {"kind": "number", "sha256": answer_hash("m05", "q1", "4000")}}


def test_committed_answer_hashes_are_up_to_date():
    for f in sorted(REPO_ROOT.glob("modules/m*/tests/answer_hashes.json")):
        nn = f.parent.parent.name[1:3]
        assert json.loads(f.read_text()) == course.build_answer_hashes(REPO_ROOT, nn), (
            f"{f} is stale: run make answer-hashes M={nn}"
        )
