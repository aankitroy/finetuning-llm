"""M1 is written for complete beginners: these pin the structure that makes it so.

They read the *starter* copies, so a learner who has run the lesson or written
exercises doesn't break `make test`.
"""

import json
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
    last_code = max(i for i, c in enumerate(cells) if c["cell_type"] == "code")
    for i, c in enumerate(cells):
        if c["cell_type"] != "code" or i == last_code:
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


def test_lesson_matches_starter():
    changed = subprocess.run(
        ["git", "diff", "--quiet", "--", str(M / "lesson.ipynb")], cwd=REPO_ROOT
    ).returncode
    if changed:
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
    import re

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
