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


def _write(tmp_path, data: bytes):
    p = tmp_path / ".progress" / "history.jsonl"
    p.parent.mkdir(exist_ok=True)
    p.write_bytes(data)


GOOD = b'{"v": 1, "module": "m00", "passed": true}\n'


def test_nan_metric_written_as_null(tmp_path):
    append_record(
        tmp_path,
        {
            "v": 1,
            "module": "m00",
            "passed": False,
            "metrics": {"final_loss": float("nan"), "x": float("inf")},
        },
    )
    raw = (tmp_path / ".progress" / "history.jsonl").read_text()
    assert "NaN" not in raw and "Infinity" not in raw
    recs, _ = read_history(tmp_path)
    assert recs[0]["metrics"] == {"final_loss": None, "x": None}


def test_nan_token_in_file_is_sanitized(tmp_path):
    _write(tmp_path, b'{"v": 1, "module": "m00", "passed": true, "metrics": {"l": NaN}}\n')
    recs, w = read_history(tmp_path)
    assert recs[0]["metrics"] == {"l": None} and w == []


def test_invalid_utf8_line_skipped(tmp_path):
    _write(
        tmp_path,
        GOOD
        + b'{"v": 1, "module": "m00", "passed": true, "run_label": "caf\xe9"}\n'
        + GOOD
        + '{"v": 1, "module": "m0", "run_label": "α'.encode()[:-1],
    )
    recs, w = read_history(tmp_path)
    assert len(recs) == 2 and len(w) == 2 and "line 2" in w[0]


def test_bad_tests_and_metrics_shapes_skipped(tmp_path):
    _write(
        tmp_path,
        GOOD
        + b'{"v": 1, "module": "m00", "passed": false, "tests": "oops"}\n'
        + b'{"v": 1, "module": "m00", "passed": false, "tests": {"failed": "x"}}\n'
        + b'{"v": 1, "module": "m00", "passed": false, "metrics": [1, 2]}\n'
        + b'{"v": 1, "module": "m00", "passed": false, "metrics": {"a": "high"}}\n',
    )
    recs, w = read_history(tmp_path)
    assert len(recs) == 1 and len(w) == 4
