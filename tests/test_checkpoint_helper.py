from common.checkpoint import load_exercise


def _module(tmp_path):
    (tmp_path / "solutions").mkdir()
    (tmp_path / "exercises.py").write_text("WHO = 'learner'\n")
    (tmp_path / "solutions" / "exercises.py").write_text("WHO = 'solution'\n")
    return tmp_path


def test_loads_learner_file_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("FTCOURSE_SOLUTIONS", raising=False)
    assert load_exercise(_module(tmp_path)).WHO == "learner"


def test_loads_solution_when_env_set(tmp_path, monkeypatch):
    monkeypatch.setenv("FTCOURSE_SOLUTIONS", "1")
    assert load_exercise(_module(tmp_path)).WHO == "solution"


def test_same_filename_in_two_modules_does_not_clash(tmp_path, monkeypatch):
    monkeypatch.delenv("FTCOURSE_SOLUTIONS", raising=False)
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "exercises.py").write_text("WHO = 'a'\n")
    (b / "exercises.py").write_text("WHO = 'b'\n")
    assert load_exercise(a).WHO == "a" and load_exercise(b).WHO == "b"


import json  # noqa: E402

import pytest  # noqa: E402

from common.checkpoint import answer_hash, check_answer, normalize  # noqa: E402


@pytest.mark.parametrize(
    "kind,raw,expected",
    [
        ("shape", "(32, 4)", "32x4"),
        ("shape", "torch.Size([32, 4])", "32x4"),
        ("shape", " 32 × 4 ", "32x4"),
        ("number", "4,000,000", "4000000"),
        ("number", "-4.0", "-4"),
        ("number", "4e6", "4000000"),
        ("number", "0.25", "0.25"),
        ("choice", "B", "b"),
        ("choice", "b) because...", "b"),
        ("choice", " (c) ", "c"),
    ],
)
def test_normalize(kind, raw, expected):
    assert normalize(raw, kind) == expected


def _qmodule(tmp_path, my_answer: str):
    (tmp_path / "tests").mkdir()
    (tmp_path / "solutions").mkdir()
    (tmp_path / "answers.toml").write_text(f'[q1]\nquestion = "?"\nanswer = "{my_answer}"\n')
    (tmp_path / "solutions" / "answers.toml").write_text('[q1]\nquestion = "?"\nanswer = "b"\n')
    hashes = {"q1": {"kind": "choice", "sha256": answer_hash("m09", "q1", "b")}}
    (tmp_path / "tests" / "answer_hashes.json").write_text(json.dumps(hashes))
    return tmp_path


def test_check_answer_right(tmp_path, monkeypatch):
    monkeypatch.delenv("FTCOURSE_SOLUTIONS", raising=False)
    assert check_answer(_qmodule(tmp_path, "B"), "m09", "q1") == (True, "")


def test_check_answer_wrong_does_not_reveal(tmp_path, monkeypatch):
    monkeypatch.delenv("FTCOURSE_SOLUTIONS", raising=False)
    ok, msg = check_answer(_qmodule(tmp_path, "c"), "m09", "q1")
    assert not ok and "q1" in msg and "make hint" in msg and "'b'" not in msg


def test_check_answer_blank(tmp_path, monkeypatch):
    monkeypatch.delenv("FTCOURSE_SOLUTIONS", raising=False)
    ok, msg = check_answer(_qmodule(tmp_path, ""), "m09", "q1")
    assert not ok and "answers.toml" in msg


def test_check_answer_uses_solutions_in_solutions_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("FTCOURSE_SOLUTIONS", "1")
    assert check_answer(_qmodule(tmp_path, "c"), "m09", "q1")[0] is True


def test_hash_is_module_and_question_specific():
    assert answer_hash("m01", "q1", "b") != answer_hash("m01", "q2", "b")
    assert answer_hash("m01", "q1", "b") != answer_hash("m02", "q1", "b")
