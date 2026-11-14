import json

from common.progress import (
    append_record,
    collected_metrics,
    read_history,
    report_metric,
    reset_metrics,
)


def test_missing_file_is_empty(tmp_path):
    assert read_history(tmp_path) == ([], [])


def test_roundtrip(tmp_path):
    append_record(tmp_path, {"v": 1, "module": "m00", "passed": True})
    append_record(tmp_path, {"v": 1, "module": "m01", "passed": False})
    recs, w = read_history(tmp_path)
    assert [r["module"] for r in recs] == ["m00", "m01"]
    assert w == []


def test_malformed_and_truncated_lines_skipped(tmp_path):
    p = tmp_path / ".progress" / "history.jsonl"
    p.parent.mkdir()
    good = json.dumps({"v": 1, "module": "m00", "passed": True})
    p.write_text(good + "\nnot json\n" + '{"v": 1, "module": "m0')
    recs, w = read_history(tmp_path)
    assert len(recs) == 1
    assert len(w) == 2
    assert "line 2" in w[0] and "line 3" in w[1]


def test_record_missing_required_keys_skipped(tmp_path):
    p = tmp_path / ".progress" / "history.jsonl"
    p.parent.mkdir()
    p.write_text('{"v": 1}\n{"v": 1, "module": "m00", "passed": "yes"}\n')
    recs, w = read_history(tmp_path)
    assert recs == [] and len(w) == 2


def test_blank_lines_ignored(tmp_path):
    p = tmp_path / ".progress" / "history.jsonl"
    p.parent.mkdir()
    p.write_text('\n{"v": 1, "module": "m00", "passed": true}\n\n')
    assert read_history(tmp_path) == ([{"v": 1, "module": "m00", "passed": True}], [])


def test_report_metric_collects_and_resets():
    reset_metrics()
    report_metric("final_loss", 1.5)
    assert collected_metrics() == {"final_loss": 1.5}
    reset_metrics()
    assert collected_metrics() == {}
