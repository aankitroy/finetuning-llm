"""Module videos: the script, the narration audio, the watched record, the guide and the server."""

import json
import sys
import threading
import urllib.error
import urllib.request

import pytest

from common.course import load_modules
from common.guide import module_guide
from common.video import load_video, mark_watched, scene_sha, watched_modules
from tests.conftest import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from progress_server import find_free_port, make_server  # noqa: E402

MANIFEST = """
id = "m09"
title = "Demo"
part = "A"
order = 9
summary = "s"
est_minutes = 10

[next]
start = "a"
finish = "b"
why = "c"

[[steps]]
kind = "video"
title = "Watch the video"
detail = "d"

[[steps]]
kind = "checkpoint"
title = "Pass"
command = "make check M=09"
"""

SCRIPT = """
title = "Demo video"

[[scene]]
id = "s01"
visual = "intro"
narration = "Hello there. This is the first scene."

[[scene]]
id = "s02"
visual = "end"
narration = "And this is the end."
"""


@pytest.fixture
def course(tmp_path):
    m = tmp_path / "modules" / "m09_demo"
    (m / "video" / "audio").mkdir(parents=True)
    (m / "module.toml").write_text(MANIFEST)
    (m / "video" / "script.toml").write_text(SCRIPT)
    (m / "video" / "scenes.js").write_text("// scenes")
    timings = {}
    for sid, text, secs in [
        ("s01", "Hello there. This is the first scene.", 3.5),
        ("s02", "And this is the end.", 2.0),
    ]:
        (m / "video" / "audio" / f"{sid}.mp3").write_bytes(b"ID3fake")
        timings[sid] = {"seconds": secs, "sha256": scene_sha(text)}
    (m / "video" / "timings.json").write_text(json.dumps(timings))
    return tmp_path


def test_load_video_joins_script_and_timings(course):
    video, problems = load_video(course / "modules" / "m09_demo")
    assert problems == []
    assert video["title"] == "Demo video"
    assert [s["id"] for s in video["scenes"]] == ["s01", "s02"]
    assert video["scenes"][0] == {
        "id": "s01",
        "visual": "intro",
        "narration": "Hello there. This is the first scene.",
        "seconds": 3.5,
        "audio": "audio/s01.mp3",
    }
    assert video["total_seconds"] == 5.5


def test_load_video_reports_stale_or_missing_audio(course):
    m = course / "modules" / "m09_demo"
    (m / "video" / "script.toml").write_text(SCRIPT.replace("the end.", "the very end."))
    (m / "video" / "audio" / "s01.mp3").unlink()
    _, problems = load_video(m)
    assert any("s01" in p and "audio" in p for p in problems)
    assert any("s02" in p and "narration changed" in p for p in problems)


def test_no_video_folder_means_no_video(tmp_path):
    assert load_video(tmp_path) == (None, [])


def test_scene_sha_depends_on_voice_and_text():
    assert scene_sha("a") != scene_sha("b")
    assert scene_sha("a") != scene_sha("a", voice="am_adam")
    assert scene_sha("a") == scene_sha("a")


def test_watched_record(tmp_path):
    assert watched_modules(tmp_path) == set()
    mark_watched(tmp_path, "m09")
    mark_watched(tmp_path, "m09")
    assert watched_modules(tmp_path) == {"m09"}
    (tmp_path / ".progress" / "watched.jsonl").open("a").write("not json\n")
    assert watched_modules(tmp_path) == {"m09"}


def test_video_step_in_the_guide(course):
    modules, warnings = load_modules(course)
    assert warnings == []
    (m,) = modules
    step = module_guide(course, m, [])["steps"][0]
    assert step["kind"] == "video" and step["status"] == "todo"
    assert step["url"] == "/video/m09/"
    assert step["progress"] == "0:06 long"
    mark_watched(course, "m09")
    assert module_guide(course, m, [])["steps"][0]["status"] == "done"


@pytest.fixture
def server(course):
    port = find_free_port()
    srv = make_server(course, port)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{port}"
    srv.shutdown()
    srv.server_close()


def _get(url):
    with urllib.request.urlopen(url, timeout=5) as r:
        return r.status, r.headers["Content-Type"], r.read()


def test_server_serves_the_player_script_and_audio(server):
    status, ctype, body = _get(server + "/video/m09/")
    assert status == 200 and ctype.startswith("text/html") and b"video.js" in body
    status, ctype, body = _get(server + "/video/m09/script.json")
    assert status == 200 and json.loads(body)["scenes"][1]["id"] == "s02"
    status, ctype, body = _get(server + "/video/m09/audio/s01.mp3")
    assert status == 200 and ctype == "audio/mpeg" and body == b"ID3fake"
    status, ctype, _ = _get(server + "/video/m09/scenes.js")
    assert status == 200 and "javascript" in ctype


@pytest.mark.parametrize(
    "path",
    [
        "/video/m09/audio/../../module.toml",
        "/video/m09/audio/%2e%2e/script.toml",
        "/video/m77/",
        "/video/m09/nope.txt",
    ],
)
def test_server_refuses_anything_else(server, path):
    with pytest.raises(urllib.error.HTTPError) as e:
        _get(server + path)
    assert e.value.code == 404


def _post(url, body: bytes):
    req = urllib.request.Request(
        url, data=body, method="POST", headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=5) as r:
        return r.status


def test_server_records_a_watched_video(server, course):
    assert _post(server + "/api/watched", b'{"module": "m09"}') == 204
    assert watched_modules(course) == {"m09"}
    for bad in [b'{"module": "m77"}', b"not json", b'{"module": 5}']:
        with pytest.raises(urllib.error.HTTPError) as e:
            _post(server + "/api/watched", bad)
        assert e.value.code == 400
    assert watched_modules(course) == {"m09"}


# --- every real module that has a video ---

VIDEO_MODULES = sorted(p.parent.parent for p in REPO_ROOT.glob("modules/m*/video/script.toml"))


@pytest.mark.parametrize("folder", VIDEO_MODULES, ids=lambda p: p.name)
def test_module_video_is_complete(folder):
    import re

    video, problems = load_video(folder)
    assert problems == [], problems
    assert 5 <= len(video["scenes"]) <= 14
    assert 60 <= video["total_seconds"] <= 360, "aim for a video of a few minutes"
    scenes_js = (folder / "video" / "scenes.js").read_text()
    drawn = set(re.findall(r"\bS\.(\w+)\s*=", scenes_js))
    missing = {s["visual"] for s in video["scenes"]} - drawn
    assert not missing, f"scenes.js draws nothing for {missing}"
    modules, _ = load_modules(REPO_ROOT)
    (module,) = [m for m in modules if m.path == folder]
    assert module.steps[0]["kind"] == "video", "the video should be the module's first step"
