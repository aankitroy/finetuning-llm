import os
import subprocess

from tests.conftest import REPO_ROOT


def _make(tmp_path, *args):
    env = {**os.environ, "FTCOURSE_ROOT": str(tmp_path)}
    return subprocess.run(
        ["make", "-s", *args], cwd=REPO_ROOT, env=env, capture_output=True, text=True
    )


def test_check_without_m_exits_2(tmp_path):
    r = _make(tmp_path, "check")
    assert r.returncode == 2
    assert "make check M=" in r.stdout + r.stderr
    assert not (tmp_path / ".progress").exists()


def test_check_unknown_module_exits_2(tmp_path):
    r = _make(tmp_path, "check", "M=99")
    assert r.returncode == 2
    assert "m99" in r.stdout + r.stderr


def test_reset_on_empty_root_says_nothing_to_reset(tmp_path):
    r = _make(tmp_path, "reset")
    assert r.returncode == 0 and "Nothing to reset" in r.stdout


def test_hint_without_ex_exits_2(tmp_path):
    r = _make(tmp_path, "hint", "M=01")
    assert r.returncode == 2 and "make hint M=NN EX=" in r.stdout + r.stderr


def test_solution_without_m_exits_2(tmp_path):
    r = _make(tmp_path, "solution", "EX=1")
    assert r.returncode == 2
