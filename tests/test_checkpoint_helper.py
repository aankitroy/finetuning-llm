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
