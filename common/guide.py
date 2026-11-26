"""Step-by-step module guide: works out each step's state from what's on disk.

Nothing here is ticked by hand. A step is done only when the files show it:
notebook cells run, exercises written (and passing at the last check),
questions answered (and right at the last check), training run with your
own code, and the checkpoint passed.
"""

from __future__ import annotations

import ast
import json
import tomllib
from pathlib import Path

from common.course import Module


def _status(done: bool, started: bool) -> str:
    return "done" if done else "in_progress" if started else "todo"


def _hints(module: Module) -> list[dict]:
    f = module.path / "hints.toml"
    if not f.exists():
        return []
    try:
        return [e for e in tomllib.loads(f.read_text()).get("exercise", []) if isinstance(e, dict)]
    except (tomllib.TOMLDecodeError, UnicodeDecodeError):
        return []


def _last_results(attempts: list[dict]) -> dict[str, str] | None:
    """{test name: failure message} from the latest check, or None if unknown."""
    if not attempts or not isinstance(attempts[-1].get("failures"), list):
        return None
    return {
        str(f.get("test")): str(f.get("message", ""))
        for f in attempts[-1]["failures"]
        if isinstance(f, dict)
    }


def _verdict(test: str | None, last: dict[str, str] | None) -> tuple[str | None, str]:
    if last is None or not test:
        return None, ""
    if test in last:
        return "fail", last[test]
    return "pass", ""


def _is_stub(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    body = list(node.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]  # docstring
    if len(body) != 1 or not isinstance(body[0], ast.Raise) or body[0].exc is None:
        return False
    exc = body[0].exc
    target = exc.func if isinstance(exc, ast.Call) else exc
    return isinstance(target, ast.Name) and target.id == "NotImplementedError"


def _written(defs: dict[str, ast.AST], name: str) -> bool:
    node = defs.get(name)
    if node is None:
        return False
    if isinstance(node, ast.ClassDef):
        methods = [n for n in node.body if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)]
        return bool(methods) and not any(_is_stub(m) for m in methods)
    return not _is_stub(node)


# --- one function per step kind; each returns extra fields for the step ---


def _video(root: Path, module: Module, step: dict, attempts) -> dict:
    from common.video import load_video, watched_modules

    video, problems = load_video(module.path)
    if video is None:
        return {"status": "todo", "progress": "", "note": "This module's video isn't made yet."}
    secs = round(video["total_seconds"])
    return {
        "status": "done" if module.id in watched_modules(root) else "todo",
        "progress": f"{secs // 60}:{secs % 60:02d} long",
        "note": "; ".join(problems),
        "url": f"/video/{module.id}/",
    }


def _lesson(root: Path, module: Module, step: dict, attempts) -> dict:
    f = module.path / (step["file"] or "lesson.ipynb")
    if not f.exists():
        return {"status": "todo", "progress": "", "note": f"{f.name} not found"}
    try:
        cells = json.loads(f.read_text()).get("cells", [])
    except (json.JSONDecodeError, UnicodeDecodeError, AttributeError):
        return {"status": "todo", "progress": "", "note": f"couldn't read {f.name}"}
    code = [c for c in cells if isinstance(c, dict) and c.get("cell_type") == "code"]
    ran = sum(1 for c in code if c.get("execution_count") is not None)
    return {
        "status": _status(bool(code) and ran == len(code), ran > 0),
        "progress": f"{ran} of {len(code)} cells run",
        "note": "Save the notebook in Jupyter so your progress shows here." if ran == 0 else "",
    }


def _exercises(root: Path, module: Module, step: dict, attempts) -> dict:
    f = module.path / (step["file"] or "exercises.py")
    nn = module.id[1:]
    exercises = [e for e in _hints(module) if e.get("name")]
    last = _last_results(attempts)
    note = ""
    defs: dict[str, ast.AST] = {}
    if not f.exists():
        note = f"{f.name} not found"
    else:
        try:
            tree = ast.parse(f.read_text())
            defs = {
                n.name: n
                for n in tree.body
                if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
            }
        except SyntaxError as e:
            note = f"{f.name} has a syntax error (line {e.lineno}): {e.msg}"
    items = []
    for e in exercises:
        verdict, message = _verdict(e.get("test"), last)
        items.append(
            {
                "id": str(e["id"]),
                "name": e["name"],
                "written": _written(defs, e["name"]),
                "last": verdict,
                "message": message,
                "hint": f"make hint M={nn} EX={e['id']}",
            }
        )
    written = sum(i["written"] for i in items)
    passing = sum(i["last"] == "pass" for i in items)
    progress = f"{written} of {len(items)} written"
    if last is not None and items:
        progress += f", {passing} passing at your last check"
    all_done = bool(items) and written == len(items) and passing == len(items)
    return {
        "status": _status(all_done, written > 0 or bool(note and f.exists())),
        "progress": progress,
        "note": note,
        "items": items,
    }


def _answers(root: Path, module: Module, step: dict, attempts) -> dict:
    f = module.path / (step["file"] or "answers.toml")
    nn = module.id[1:]
    if not f.exists():
        return {"status": "todo", "progress": "", "note": f"{f.name} not found", "items": []}
    try:
        data = tomllib.loads(f.read_text())
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as e:
        return {
            "status": "in_progress",
            "progress": "",
            "note": f"{f.name} can't be read: {e}. Check the quotes around your answers.",
            "items": [],
        }
    tests = {str(e.get("id")): e.get("test") for e in _hints(module)}
    last = _last_results(attempts)
    items = []
    for qid, q in data.items():
        if not isinstance(q, dict):
            continue
        verdict, message = _verdict(tests.get(qid) or f"test_understanding[{qid}]", last)
        items.append(
            {
                "id": qid,
                "question": str(q.get("question", "")).strip().splitlines()[0]
                if q.get("question")
                else "",
                "answered": bool(str(q.get("answer", "")).strip()),
                "last": verdict,
                "message": message,
                "hint": f"make hint M={nn} EX={qid}",
            }
        )
    answered = sum(i["answered"] for i in items)
    right = sum(i["last"] == "pass" for i in items)
    progress = f"{answered} of {len(items)} answered"
    if last is not None and items:
        progress += f", {right} right at your last check"
    all_done = bool(items) and answered == len(items) and right == len(items)
    return {
        "status": _status(all_done, answered > 0),
        "progress": progress,
        "note": "",
        "items": items,
    }


def _train(root: Path, module: Module, step: dict, attempts) -> dict:
    f = Path(root) / (step["file"] or f"outputs/{module.id}/run_summary.json")
    if not f.exists():
        return {"status": "todo", "progress": "Not run yet", "note": ""}
    try:
        summary = json.loads(f.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {"status": "in_progress", "progress": "", "note": f"couldn't read {f.name}"}
    out = {"status": "done", "progress": "Trained", "note": ""}
    if step.get("metric") and isinstance(summary.get(step["metric"]), int | float):
        out["metric"] = step["metric"]
        out["value"] = summary[step["metric"]]
    if summary.get("used_solutions"):
        out["status"] = "in_progress"
        out["note"] = "This run used the reference solutions. Train again with your own code."
    return out


def _checkpoint(root: Path, module: Module, step: dict, attempts) -> dict:
    if any(a.get("passed") for a in attempts):
        return {"status": "done", "progress": "Passed", "note": ""}
    if not attempts:
        return {"status": "todo", "progress": "Not run yet", "note": ""}
    t = attempts[-1].get("tests") or {}
    passed, failed = int(t.get("passed", 0)), int(t.get("failed", 0))
    return {
        "status": "in_progress",
        "progress": f"Last check: {passed} of {passed + failed} tests passed",
        "note": "",
    }


_KINDS = {
    "video": _video,
    "lesson": _lesson,
    "exercises": _exercises,
    "answers": _answers,
    "train": _train,
    "checkpoint": _checkpoint,
}


def module_guide(root: Path, module: Module, attempts: list[dict]) -> dict | None:
    if not module.steps:
        return None
    steps = []
    for st in module.steps:
        try:
            extra = _KINDS[st["kind"]](Path(root), module, st, attempts)
        except Exception as e:  # a guide problem must never take the dashboard down
            extra = {"status": "todo", "progress": "", "note": f"couldn't check this step: {e}"}
        steps.append(
            {
                "kind": st["kind"],
                "title": st["title"],
                "detail": st["detail"],
                "command": st["command"],
                **extra,
            }
        )
    return {
        "steps": steps,
        "done": sum(s["status"] == "done" for s in steps),
        "total": len(steps),
    }
