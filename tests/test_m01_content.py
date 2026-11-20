"""M1 is written for complete beginners: these pin the structure that makes it so.

They read the *starter* copies, so a learner who has run the lesson or written
exercises doesn't break `make test`.
"""

import json
import re
import shutil
import subprocess

import pytest

from tests.conftest import REPO_ROOT

M = REPO_ROOT / "modules" / "m01_pytorch_essentials"
STARTER_NB = M / "starter" / "lesson.ipynb"

SECTIONS = [
    "## 0. What is a model",
    "## 1. Tensors",
    "## 2. Gradients and autograd",
    "## 3. Linear regression, by hand",
    "## 4. nn.Module",
    "## 5. Classification",
    "## 6. Batches and epochs",
    "## 7. A worked example",
    "## 8. The Mac's GPU",
    "## 9. Your task",
]

TERMS = [
    "tensor",
    "shape",
    "dtype",
    "device",
    "matrix multiply",
    "broadcasting",
    "gradient",
    "autograd",
    "loss",
    "logits",
    "softmax",
    "cross-entropy",
    "argmax",
    "ReLU",
    "batch",
    "epoch",
    "learning rate",
    "optimizer",
    "nn.Module",
    "layer",
    "bias",
]


def _cells():
    return json.loads(STARTER_NB.read_text())["cells"]


def _src(cell):
    s = cell["source"]
    return "".join(s) if isinstance(s, list) else s


# --- the lesson ---


def test_starter_lesson_is_unrun():
    code = [c for c in _cells() if c["cell_type"] == "code"]
    assert code
    for c in code:
        assert c["execution_count"] is None and c["outputs"] == []


def test_lesson_sections_in_order():
    headings = [
        line
        for c in _cells()
        if c["cell_type"] == "markdown"
        for line in _src(c).splitlines()
        if line.startswith("## ")
    ]
    found = [next((h for h in headings if h.startswith(s)), None) for s in SECTIONS]
    assert None not in found, (
        f"missing: {[s for s, f in zip(SECTIONS, found, strict=True) if f is None]}"
    )
    assert [headings.index(f) for f in found] == sorted(headings.index(f) for f in found)


def test_every_code_cell_is_explained():
    cells = _cells()
    for i, c in enumerate(cells):
        if c["cell_type"] != "code":
            continue
        nxt = cells[i + 1] if i + 1 < len(cells) else None
        assert nxt and nxt["cell_type"] == "markdown", f"cell {i} has no explanation after it"
        assert _src(nxt).startswith("**What each line does**"), f"cell {i}: {_src(c)[:60]!r}"


def test_key_terms_are_defined():
    text = "\n".join(_src(c) for c in _cells() if c["cell_type"] == "markdown").lower()
    missing = [t for t in TERMS if f"**{t.lower()}**" not in text]
    assert not missing, f"not defined in bold: {missing}"


def test_lesson_runs_top_to_bottom(tmp_path):
    pytest.importorskip("torch")
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    work = tmp_path / "m01"
    shutil.copytree(M, work, ignore=shutil.ignore_patterns(".ipynb_checkpoints", "__pycache__"))
    nb = nbformat.read(work / "starter" / "lesson.ipynb", as_version=4)
    client = nbclient.NotebookClient(
        nb, timeout=180, kernel_name="python3", resources={"metadata": {"path": str(work)}}
    )
    client.execute()
    out = "".join(
        o.get("text", "") for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", [])
    )
    by_hand = re.search(r"cross-entropy by hand: (\S+)", out).group(1)
    pytorch = re.search(r"F.cross_entropy: +(\S+)", out).group(1)
    assert by_hand == pytorch, "print both losses the same way so they visibly match"


def test_lesson_matches_starter():
    changed = subprocess.run(
        ["git", "diff", "--quiet", "--", str(M / "lesson.ipynb")], cwd=REPO_ROOT
    ).returncode
    if changed == 1:  # 1 = differs; other codes mean no git checkout
        pytest.skip("you've run or edited the lesson")
    mine = [_src(c) for c in json.loads((M / "lesson.ipynb").read_text())["cells"]]
    assert mine == [_src(c) for c in _cells()]


# --- the exercises ---

EXERCISES = ["linear_forward", "mse_loss", "sgd_step", "MLP", "train_one_epoch", "accuracy"]


def _starter_defs():
    import ast

    source = (M / "starter" / "exercises.py").read_text()
    tree = ast.parse(source)
    defs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef | ast.ClassDef)}
    return source.splitlines(), defs


def _stub_bodies(name):
    """Source lines of each function that makes up exercise `name`."""
    import ast

    lines, defs = _starter_defs()
    node = defs[name]
    funcs = (
        [n for n in node.body if isinstance(n, ast.FunctionDef)]
        if isinstance(node, ast.ClassDef)
        else [node]
    )
    return [lines[f.lineno - 1 : f.end_lineno] for f in funcs]


@pytest.mark.parametrize("name", EXERCISES)
def test_each_exercise_has_step_comments(name):
    for body in _stub_bodies(name):
        assert any(re.match(r"^\s+# 1\. ", line) for line in body), f"{name}: no '# 1. ' step"
        assert any("raise NotImplementedError" in line for line in body)


@pytest.mark.parametrize("name", EXERCISES)
def test_each_exercise_says_what_it_is(name):
    import ast

    _, defs = _starter_defs()
    assert "In plain words:" in (ast.get_docstring(defs[name]) or "")


def test_step_comments_dont_count_as_written():
    from common.guide import _written

    _, defs = _starter_defs()
    assert not any(_written(defs, name) for name in EXERCISES)


@pytest.mark.parametrize("name", ["MLP", "train_one_epoch"])
def test_harder_exercises_point_to_the_lesson(name):
    import ast

    _, defs = _starter_defs()
    assert "section 7" in (ast.get_docstring(defs[name]) or "")


# --- the understanding checks ---


def test_question_ids_line_up():
    import tomllib

    starter = set(tomllib.loads((M / "starter" / "answers.toml").read_text()))
    solutions = set(tomllib.loads((M / "solutions" / "answers.toml").read_text()))
    hints = {
        str(e["id"])
        for e in tomllib.loads((M / "hints.toml").read_text())["exercise"]
        if str(e["id"]).startswith("q")
    }
    hashes = set(json.loads((M / "tests" / "answer_hashes.json").read_text()))
    expected = {f"q{i}" for i in range(1, 7)}
    assert starter == solutions == hints == hashes == expected


# --- the README ---

SYNTAX = [
    "`requires_grad=True`",
    "`.backward()`",
    "`.grad`",
    "`torch.no_grad()`",
    "`nn.Module`",
    "`nn.Linear`",
    "`optimizer.zero_grad()`",
    "`optimizer.step()`",
    "`F.cross_entropy`",
    "`.argmax(dim=1)`",
    "`.to(device)`",
    "`.item()`",
]


def _readme_section(title):
    text = (M / "README.md").read_text()
    assert f"## {title}" in text, f"README has no '## {title}' section"
    return text.split(f"## {title}", 1)[1].split("\n## ", 1)[0]


def test_readme_glossary_covers_terms():
    rows = [
        line.lower() for line in _readme_section("Glossary").splitlines() if line.startswith("|")
    ]
    terms = TERMS + ["MSE", "layer", "parameter", "nn.Module", "MPS"]
    missing = [t for t in terms if not any(r.startswith(f"| **{t.lower()}**") for r in rows)]
    assert not missing, f"glossary is missing: {missing}"


def test_readme_syntax_table():
    rows = [
        line for line in _readme_section("Syntax at a glance").splitlines() if line.startswith("|")
    ]
    missing = [s for s in SYNTAX if not any(r.startswith(f"| {s} ") for r in rows)]
    assert not missing, f"syntax table is missing: {missing}"


def test_m01_estimate():
    from common.course import load_modules

    m01 = next(m for m in load_modules(REPO_ROOT)[0] if m.id == "m01")
    assert m01.est_minutes == 240


def _markdown():
    return "\n".join(_src(c) for c in _cells() if c["cell_type"] == "markdown")


def test_accuracy_pieces_are_taught():
    """Exercise 6 needs ==, .float() and argmax(dim=1): the lesson must show all three."""
    code = "\n".join(_src(c) for c in _cells() if c["cell_type"] == "code")
    assert "argmax(dim=1)" in code
    assert ".float()" in code
    assert re.search(r"\w+ == \w+", code), "show comparing two tensors with =="
    rows = _readme_section("Syntax at a glance")
    assert "| `.float()` " in rows


@pytest.mark.parametrize("term", ["layer", "bias", "batch"])
def test_terms_are_not_used_before_they_are_explained(term):
    text = _markdown()
    first = re.search(rf"\b{term}", text, re.IGNORECASE).start()
    defined = text.lower().find(f"**{term}**")
    explained_here = text[first - 2 : first] == "**"
    points_ahead = "section" in text[first : first + 80]
    assert explained_here or points_ahead or first > defined, (
        f"'{term}' is used before it's explained: {text[first - 60 : first + 40]!r}"
    )


def test_gpu_maths_is_right():
    text = _markdown() + "\n".join(_src(c) for c in _cells())
    assert "137 billion multiply-adds" not in text


def test_hints_are_nudges_not_solutions():
    import tomllib

    hints = {
        str(e["id"]): e["hint"] for e in tomllib.loads((M / "hints.toml").read_text())["exercise"]
    }
    assert "p -= lr * p.grad" not in hints["3"]
    assert "torch.randperm(len(X), device=X.device)" not in hints["5"]
    assert "section 5" in hints["q5"]
