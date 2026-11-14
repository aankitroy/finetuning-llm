import pytest

from common.progress import append_record, compute_state

EST = {"m00": 60, "m01": 120, "m02": 90}


def _manifest(i: int) -> str:
    nn = f"{i:02d}"
    finish = "make doctor" if i == 0 else f"make check M={nn}"
    return (
        f'id = "m{nn}"\ntitle = "Module {nn}"\npart = "A"\norder = {i}\nsummary = "s"\n'
        f"est_minutes = {EST[f'm{nn}']}\n"
        f'[next]\nstart = "start m{nn}"\nfinish = "{finish}"\nwhy = "why m{nn}"\n'
        f'[artifacts]\nany = ["outputs/m{nn}/*"]\n'
        f'[[metrics]]\nkey = "peak_mem_gb"\nlabel = "Peak"\nbetter = "lower"\n'
    )


@pytest.fixture
def root(tmp_path):
    for i in range(3):
        d = tmp_path / "modules" / f"m{i:02d}_x"
        d.mkdir(parents=True)
        (d / "module.toml").write_text(_manifest(i))
    return tmp_path


def rec(module, passed, tp=1, tf=0, **metrics):
    return {
        "v": 1,
        "ts": "2026-11-14T10:00:00+05:30",
        "module": module,
        "passed": passed,
        "tests": {"passed": tp, "failed": tf},
        "metrics": metrics,
        "run_label": None,
    }


def status(s, mid):
    return next(m for m in s["modules"] if m["id"] == mid)["status"]


def module(s, mid):
    return next(m for m in s["modules"] if m["id"] == mid)


def test_state_keys(root):
    assert set(compute_state(root)) == {"course", "modules", "next", "history", "warnings"}


def test_empty_state(root):
    s = compute_state(root)
    assert s["next"]["kind"] == "empty"
    assert s["next"]["command"] == "make doctor"
    assert s["next"]["module"] == "m00"
    assert all(m["status"] == "not_started" for m in s["modules"])
    assert s["course"] == {"total": 3, "done": 0, "est_minutes_done": 0}


def test_empty_state_without_any_modules(tmp_path):
    s = compute_state(tmp_path)
    assert s["next"]["kind"] == "empty" and s["modules"] == []


def test_done_needs_a_passing_record(root):
    append_record(root, rec("m00", False))
    append_record(root, rec("m00", True))
    append_record(root, rec("m01", False))
    s = compute_state(root)
    assert status(s, "m00") == "done"
    assert status(s, "m01") == "in_progress"
    assert status(s, "m02") == "not_started"


def test_artifact_marks_in_progress(root):
    (root / "outputs" / "m01").mkdir(parents=True)
    (root / "outputs" / "m01" / "adapter.safetensors").write_text("x")
    assert status(compute_state(root), "m01") == "in_progress"


def test_next_prefers_lowest_in_progress_and_retry_wording(root):
    append_record(root, rec("m00", True))
    append_record(root, rec("m01", False, tp=2, tf=1))
    (root / "outputs" / "m02").mkdir(parents=True)
    (root / "outputs" / "m02" / "a.json").write_text("{}")
    n = compute_state(root)["next"]
    assert n["module"] == "m01" and n["kind"] == "retry"
    assert "1 of 3" in n["instruction"]
    assert n["command"] == "make check M=01"


def test_next_finish_for_artifact_only(root):
    append_record(root, rec("m00", True))
    (root / "outputs" / "m01").mkdir(parents=True)
    (root / "outputs" / "m01" / "a.json").write_text("{}")
    n = compute_state(root)["next"]
    assert n["kind"] == "finish" and n["module"] == "m01" and n["command"] == "make check M=01"


def test_next_start_when_nothing_in_progress(root):
    append_record(root, rec("m00", True))
    n = compute_state(root)["next"]
    assert n["kind"] == "start" and n["module"] == "m01"
    assert n["instruction"] == "start m01" and n["why"] == "why m01"
    assert n["command"] == "make check M=01"


def test_complete(root):
    for m in ("m00", "m01", "m02"):
        append_record(root, rec(m, True))
    n = compute_state(root)["next"]
    assert n["kind"] == "complete" and n["command"] is None and n["module"] is None


def test_unknown_module_record_ignored_with_warning(root):
    append_record(root, rec("m99", True))
    s = compute_state(root)
    assert s["history"] == []
    assert any("m99" in w for w in s["warnings"])
    assert s["course"]["done"] == 0


def test_latest_metrics_and_counts(root):
    append_record(root, rec("m00", True, peak_mem_gb=2.0))
    append_record(root, rec("m00", True, peak_mem_gb=1.5))
    s = compute_state(root)
    m = module(s, "m00")
    assert m["latest_metrics"] == {"peak_mem_gb": 1.5}
    assert len(m["attempts"]) == 2
    assert m["metrics"] == [{"key": "peak_mem_gb", "label": "Peak", "better": "lower"}]
    assert s["course"] == {"total": 3, "done": 1, "est_minutes_done": 60}


def test_history_warnings_merged(root):
    (root / ".progress").mkdir()
    (root / ".progress" / "history.jsonl").write_text("garbage\n")
    assert any("line 1" in w for w in compute_state(root)["warnings"])
