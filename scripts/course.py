"""Course helper: hints, solutions and reset.

make hint M=01 EX=3        a nudge for exercise 3 (logged)
make solution M=01 EX=3    the reference answer for exercise 3 (logged)
make reset                 start the whole course over
make reset M=01            start one module over
"""

from __future__ import annotations

import argparse
import ast
import json
import shutil
import sys
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from common.progress import (  # noqa: E402
    HISTORY_PATH,
    REVEALS_PATH,
    append_reveal,
    read_history,
    read_reveals,
)


class CourseError(Exception):
    """A problem worth a one-line message, not a traceback."""


# --- lookups ---


def _module_dir(root: Path, nn: str) -> Path:
    matches = sorted((Path(root) / "modules").glob(f"m{nn}_*"))
    if not matches:
        raise CourseError(f"No module m{nn} (looked for modules/m{nn}_*)")
    return matches[0]


def _exercise(module_dir: Path, ex: str) -> dict:
    hints_file = module_dir / "hints.toml"
    if not hints_file.exists():
        raise CourseError(f"{module_dir.name} has no hints yet")
    exercises = tomllib.loads(hints_file.read_text()).get("exercise", [])
    for e in exercises:
        if str(e.get("id")) == str(ex):
            return e
    ids = ", ".join(str(e.get("id")) for e in exercises)
    raise CourseError(f"No exercise {ex!r} in {module_dir.name}. Choose one of: {ids}")


def _definition_source(path: Path, name: str) -> str:
    source = path.read_text()
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef | ast.ClassDef) and node.name == name:
            start = min([node.lineno] + [d.lineno for d in node.decorator_list])
            return "\n".join(source.splitlines()[start - 1 : node.end_lineno])
    raise CourseError(f"{name} not found in {path}")


# --- hints and solutions ---


def show_hint(root: Path, nn: str, ex: str) -> str:
    d = _module_dir(root, nn)
    e = _exercise(d, ex)
    append_reveal(root, f"m{nn}", str(ex), "hint")
    return f"Hint for m{nn} exercise {ex}:\n\n  {e['hint']}\n"


def show_solution(root: Path, nn: str, ex: str) -> str:
    d = _module_dir(root, nn)
    e = _exercise(d, ex)
    if str(ex).startswith("q"):
        answers = tomllib.loads((d / "solutions" / "answers.toml").read_text())
        body = f"Answer: {answers[str(ex)]['answer']}"
        if answers[str(ex)].get("why"):
            body += f"\n\nWhy: {answers[str(ex)]['why']}"
    else:
        body = _definition_source(d / "solutions" / "exercises.py", e["name"])
    append_reveal(root, f"m{nn}", str(ex), "solution")
    return (
        f"Solution for m{nn} exercise {ex} (logged on your dashboard):\n\n{body}\n\n"
        "Type it out yourself rather than pasting. That's where it sticks.\n"
    )


# --- answer hashes (for module authors) ---


def build_answer_hashes(root: Path, nn: str) -> dict:
    """Hashes of the reference answers, so checkpoints can check answers without storing them."""
    from common.checkpoint import answer_hash, normalize

    d = _module_dir(root, nn)
    answers = tomllib.loads((d / "solutions" / "answers.toml").read_text())
    out = {}
    for qid, entry in answers.items():
        kind = entry.get("kind", "text")
        digest = answer_hash(f"m{nn}", qid, normalize(entry["answer"], kind))
        out[qid] = {"kind": kind, "sha256": digest}
    return out


# --- reset ---


def _starter_files(root: Path, nn: str | None) -> list[tuple[Path, Path]]:
    """(starter copy, learner file) pairs for one module or all."""
    pattern = f"m{nn}_*" if nn else "m*"
    pairs = []
    for d in sorted((Path(root) / "modules").glob(pattern)):
        starter = d / "starter"
        if starter.is_dir():
            pairs += [(f, d / f.name) for f in sorted(starter.iterdir()) if f.is_file()]
    return pairs


def _rel(root: Path, p: Path) -> str:
    return str(p.relative_to(root))


def reset_plan(root: Path, nn: str | None) -> list[str]:
    root = Path(root)
    if nn:
        _module_dir(root, nn)
    mid = f"m{nn}" if nn else None
    plan = []
    records = [r for r in read_history(root)[0] if mid is None or r["module"] == mid]
    if records:
        plan.append(f"delete {len(records)} progress records" + (f" for {mid}" if mid else ""))
    reveals = [r for r in read_reveals(root)[0] if mid is None or r.get("module") == mid]
    if reveals:
        plan.append(f"delete {len(reveals)} hint/solution log entries")
    outputs = root / "outputs" / (mid or "")
    if outputs.exists():
        plan.append(f"delete {_rel(root, outputs)}/ (your training results)")
    for starter, mine in _starter_files(root, nn):
        if not mine.exists() or mine.read_bytes() != starter.read_bytes():
            plan.append(f"restore {_rel(root, mine)} to its empty starter (your code is lost)")
    return plan


def _rewrite_without(path: Path, mid: str) -> None:
    if not path.exists():
        return
    keep = []
    for raw in path.read_bytes().split(b"\n"):
        if not raw.strip():
            continue
        try:
            if json.loads(raw.decode("utf-8")).get("module") == mid:
                continue
        except (UnicodeDecodeError, json.JSONDecodeError, AttributeError):
            pass  # leave lines we can't read alone
        keep.append(raw + b"\n")
    path.write_bytes(b"".join(keep))


def do_reset(root: Path, nn: str | None) -> None:
    root = Path(root)
    if nn:
        mid = f"m{nn}"
        _module_dir(root, nn)
        _rewrite_without(root / HISTORY_PATH, mid)
        _rewrite_without(root / REVEALS_PATH, mid)
        shutil.rmtree(root / "outputs" / mid, ignore_errors=True)
    else:
        shutil.rmtree(root / ".progress", ignore_errors=True)
        shutil.rmtree(root / "outputs", ignore_errors=True)
    for starter, mine in _starter_files(root, nn):
        shutil.copyfile(starter, mine)


# --- CLI ---


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Course hints, solutions and reset")
    ap.add_argument("--root", type=Path, default=REPO)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("hint", "solution"):
        p = sub.add_parser(name)
        p.add_argument("module", help="module number, e.g. 01")
        p.add_argument("exercise", help="exercise id, e.g. 3 or q2")
    h = sub.add_parser("answer-hashes", help="regenerate tests/answer_hashes.json (authors)")
    h.add_argument("module")
    r = sub.add_parser("reset")
    r.add_argument("module", nargs="?", help="module number; omit to reset everything")
    args = ap.parse_args(argv)

    try:
        if args.cmd == "hint":
            print(show_hint(args.root, args.module, args.exercise))
            return 0
        if args.cmd == "solution":
            print(show_solution(args.root, args.module, args.exercise))
            return 0
        if args.cmd == "answer-hashes":
            out = _module_dir(args.root, args.module) / "tests" / "answer_hashes.json"
            out.write_text(json.dumps(build_answer_hashes(args.root, args.module), indent=2) + "\n")
            print(f"Wrote {out}")
            return 0
        plan = reset_plan(args.root, args.module)
        if not plan:
            print("Nothing to reset.")
            return 0
        scope = f"module m{args.module}" if args.module else "the whole course"
        print(f"This resets {scope}. It will:")
        for line in plan:
            print(f"  - {line}")
        print("This can't be undone. Reference solutions are kept.")
        if input("Type 'reset' to continue: ").strip() != "reset":
            print("Cancelled. Nothing was changed.")
            return 1
        do_reset(args.root, args.module)
        print("Done. Run `make progress` to start again.")
        return 0
    except CourseError as e:
        print(e, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
