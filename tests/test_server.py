import json
import socket
import sys
import threading
import urllib.error
import urllib.request

import pytest

from common.progress import compute_state, render_text
from tests.conftest import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from progress_server import find_free_port, make_server  # noqa: E402


@pytest.fixture
def server(tmp_path):
    port = find_free_port()
    srv = make_server(tmp_path, port)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{port}"
    srv.shutdown()
    srv.server_close()


def get(url):
    with urllib.request.urlopen(url, timeout=5) as r:
        return r.status, r.headers, r.read()


def test_api_state_shape(server):
    status, headers, body = get(server + "/api/state")
    assert status == 200
    assert headers["Content-Type"].startswith("application/json")
    assert headers["Cache-Control"] == "no-store"
    assert set(json.loads(body)) == {"course", "modules", "next", "history", "warnings"}


def test_index_and_static_served(server):
    status, headers, _ = get(server + "/")
    assert status == 200 and headers["Content-Type"].startswith("text/html")
    status, headers, _ = get(server + "/static/app.js")
    assert status == 200 and "javascript" in headers["Content-Type"]


@pytest.mark.parametrize(
    "path", ["/static/../common/progress.py", "/static/%2e%2e/common/progress.py", "/nope"]
)
def test_path_traversal_and_unknown_blocked(server, path):
    with pytest.raises(urllib.error.HTTPError) as e:
        get(server + path)
    assert e.value.code == 404


def test_binds_localhost_only(tmp_path):
    srv = make_server(tmp_path, find_free_port())
    try:
        assert srv.server_address[0] == "127.0.0.1"
    finally:
        srv.server_close()


def test_port_in_use_falls_forward():
    p = find_free_port()
    with socket.socket() as s:
        s.bind(("127.0.0.1", p))
        s.listen()
        assert find_free_port(p) > p


def test_render_text_contains_next(tmp_path):
    out = render_text(compute_state(tmp_path))
    assert "NEXT" in out and "make doctor" in out


def test_render_text_lists_modules():
    out = render_text(compute_state(REPO_ROOT))
    assert "M06" in out and "LoRA from scratch" in out
    assert "/ 17" in out


def test_api_state_is_strict_json(server, tmp_path):
    from common.progress import append_record

    d = tmp_path / "modules" / "m00_x"
    d.mkdir(parents=True)
    (d / "module.toml").write_text(
        'id = "m00"\ntitle = "T"\npart = "A"\norder = 0\nsummary = "s"\nest_minutes = 1\n'
        '[next]\nstart = "a"\nfinish = "make doctor"\nwhy = "w"\n'
    )
    (tmp_path / ".progress").mkdir()
    (tmp_path / ".progress" / "history.jsonl").write_text(
        '{"v": 1, "module": "m00", "passed": false, "metrics": {"final_loss": NaN}}\n'
    )
    append_record(
        tmp_path,
        {"v": 1, "module": "m00", "passed": False, "metrics": {"final_loss": float("nan")}},
    )
    _, _, body = get(server + "/api/state")

    def reject(token):
        raise ValueError(token)

    json.loads(body, parse_constant=reject)


def test_state_error_returns_500_json(server, monkeypatch):
    import progress_server

    def boom(root):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(progress_server, "compute_state", boom)
    with pytest.raises(urllib.error.HTTPError) as e:
        get(server + "/api/state")
    assert e.value.code == 500
    assert "kaboom" in json.loads(e.value.read())["error"]
