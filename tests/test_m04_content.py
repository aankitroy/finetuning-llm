"""M4 is written for beginners who have done M1 to M3: these pin the structure that makes it so.

They read the *starter* copies, so a learner who has run the lesson or written
exercises doesn't break `make test`.
"""

import ast
import hashlib
import json
import re
import shutil
import subprocess
import tomllib

import pytest

from tests.conftest import REPO_ROOT

M = REPO_ROOT / "modules" / "m04_training_loop"
STARTER_NB = M / "starter" / "lesson.ipynb"
SHAKESPEARE_SHA256 = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"

SECTIONS = [
    "## 0. What training really does",
    "## 1. Characters as tokens, and the train/validation split",
    "## 2. Batches of shifted windows",
    "## 3. The causal LM loss",
    "## 4. One training step",
    "## 5. Learning-rate schedules",
    "## 6. Gradient clipping",
    "## 7. Gradient accumulation",
    "## 8. Reading loss curves",
    "## 9. Your task",
]

TERMS = [
    "training",
    "step",
    "loss",
    "training set",
    "validation set",
    "context length",
    "batch",
    "logits",
    "cross-entropy",
    "causal LM loss",
    "optimizer",
    "AdamW",
    "learning rate",
    "learning-rate schedule",
    "warmup",
    "cosine decay",
    "gradient norm",
    "gradient clipping",
    "gradient accumulation",
    "micro-batch",
    "effective batch size",
    "dropout",
    "loss curve",
    "overfitting",
]

EXERCISES = [
    "get_batch",
    "causal_lm_loss",
    "lr_at",
    "clip_grad_norm",
    "train_step",
    "estimate_loss",
]


def _cells():
    return json.loads(STARTER_NB.read_text())["cells"]


def _src(cell):
    s = cell["source"]
    return "".join(s) if isinstance(s, list) else s


def _markdown():
    return "\n".join(_src(c) for c in _cells() if c["cell_type"] == "markdown")


def _code():
    return "\n".join(_src(c) for c in _cells() if c["cell_type"] == "code")


# --- the lesson ---


def test_starter_lesson_is_unrun():
    code = [c for c in _cells() if c["cell_type"] == "code"]
    assert code
    for c in code:
        assert c["execution_count"] is None and c["outputs"] == []


def test_lesson_sections_in_order():
    headings = [line for line in _markdown().splitlines() if line.startswith("## ")]
    found = [next((h for h in headings if h.startswith(s)), None) for s in SECTIONS]
    missing = [s for s, f in zip(SECTIONS, found, strict=True) if f is None]
    assert not missing, f"missing: {missing}"
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
    text = _markdown().lower()
    missing = [t for t in TERMS if f"**{t.lower()}**" not in text]
    assert not missing, f"not defined in bold: {missing}"


def test_lesson_runs_top_to_bottom(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("matplotlib")
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    work = tmp_path / "m04"
    shutil.copytree(M, work, ignore=shutil.ignore_patterns(".ipynb_checkpoints", "__pycache__"))
    nb = nbformat.read(work / "starter" / "lesson.ipynb", as_version=4)
    client = nbclient.NotebookClient(
        nb, timeout=300, kernel_name="python3", resources={"metadata": {"path": str(work)}}
    )
    client.execute()


def test_lesson_matches_starter():
    changed = subprocess.run(
        ["git", "diff", "--quiet", "--", str(M / "lesson.ipynb")], cwd=REPO_ROOT
    ).returncode
    if changed == 1:  # 1 = differs; other codes mean no git checkout
        pytest.skip("you've run or edited the lesson")
    mine = [_src(c) for c in json.loads((M / "lesson.ipynb").read_text())["cells"]]
    assert mine == [_src(c) for c in _cells()]


# --- the exercises ---


def _starter_defs():
    source = (M / "starter" / "exercises.py").read_text()
    tree = ast.parse(source)
    return source.splitlines(), {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}


@pytest.mark.parametrize("name", EXERCISES)
def test_each_exercise_has_steps_and_plain_words(name):
    lines, defs = _starter_defs()
    body = lines[defs[name].lineno - 1 : defs[name].end_lineno]
    assert any(re.match(r"^\s+# 1\. ", line) for line in body), f"{name}: no '# 1. ' step"
    doc = ast.get_docstring(defs[name]) or ""
    assert "In plain words:" in doc
    assert re.search(r"section \d", doc), f"{name} should point to a lesson section"


def test_step_comments_dont_count_as_written():
    from common.guide import _written

    _, defs = _starter_defs()
    assert not any(_written(defs, name) for name in EXERCISES)


def test_question_ids_line_up():
    starter = set(tomllib.loads((M / "starter" / "answers.toml").read_text()))
    solutions = set(tomllib.loads((M / "solutions" / "answers.toml").read_text()))
    hints = {
        str(e["id"])
        for e in tomllib.loads((M / "hints.toml").read_text())["exercise"]
        if str(e["id"]).startswith("q")
    }
    hashes = set(json.loads((M / "tests" / "answer_hashes.json").read_text()))
    assert starter == solutions == hints == hashes == {f"q{i}" for i in range(1, 7)}


# --- the README ---

SYNTAX = [
    "`torch.randint(high, (n,), generator=g)`",
    "`torch.Generator().manual_seed(0)`",
    "`torch.stack([...])`",
    "`.reshape(a, b)`",
    "`F.cross_entropy(logits, targets)`",
    "`math.log`, `math.cos`, `math.pi`",
    "`torch.optim.AdamW(model.parameters(), lr=...)`",
    "`optimizer.zero_grad()`",
    "`loss.backward()`",
    "`optimizer.step()`",
    "`p.grad`",
    "`.mul_(x)`",
    "`.item()`",
    "`model.eval()` / `model.train()`",
    "`torch.no_grad()`",
    "`plt.plot`, `plt.legend`",
]


def _readme_section(title):
    text = (M / "README.md").read_text()
    assert f"## {title}" in text, f"README has no '## {title}' section"
    return text.split(f"## {title}", 1)[1].split("\n## ", 1)[0]


def test_readme_glossary_covers_terms():
    rows = [
        line.lower() for line in _readme_section("Glossary").splitlines() if line.startswith("|")
    ]
    missing = [t for t in TERMS if not any(r.startswith(f"| **{t.lower()}**") for r in rows)]
    assert not missing, f"glossary is missing: {missing}"


def test_readme_syntax_table():
    rows = [
        line for line in _readme_section("Syntax at a glance").splitlines() if line.startswith("|")
    ]
    missing = [s for s in SYNTAX if not any(r.startswith(f"| {s}") for r in rows)]
    assert not missing, f"syntax table is missing: {missing}"


# --- specific to M4 ---


def test_lesson_shows_the_tools_the_exercises_need():
    code = _code()
    for tool in [
        "torch.randint(",
        "generator=",
        "torch.stack(",
        "batch, time, vocab = logits.shape",
        ".reshape(",
        "F.cross_entropy(",
        "math.cos(math.pi * progress)",
        "torch.sqrt(",
        "is not None",
        ".mul_(",
        "optimizer.zero_grad()",
        ".backward()",
        "optimizer.step()",
        "/ len(",
        ".item()",
        "model.eval()",
        "model.train()",
        "with torch.no_grad():",
    ]:
        assert tool in code, f"the lesson never shows {tool!r}"


def test_hints_are_nudges_not_solutions():
    hints = {
        str(e["id"]): e["hint"] for e in tomllib.loads((M / "hints.toml").read_text())["exercise"]
    }
    assert "math.cos" not in hints["3"]
    assert "/ len(batches)" not in hints["5"]
    assert "max_norm / total" not in hints["4"]


def test_right_answer_is_not_just_the_longest():
    starter = tomllib.loads((M / "starter" / "answers.toml").read_text())
    solutions = tomllib.loads((M / "solutions" / "answers.toml").read_text())
    longest_wins = 0
    for qid, sol in solutions.items():
        if sol["kind"] != "choice":
            continue
        options = dict(re.findall(r"^\s+([a-d])\) (.+)$", starter[qid]["question"], re.MULTILINE))
        right = len(options[sol["answer"]])
        if all(right > len(t) for k, t in options.items() if k != sol["answer"]):
            longest_wins += 1
    assert longest_wins <= 1, f"the longest option is the answer in {longest_wins} questions"


def test_training_text_is_the_real_tiny_shakespeare():
    data = (M / "shakespeare.txt").read_bytes()
    assert hashlib.sha256(data).hexdigest() == SHAKESPEARE_SHA256
    config = (REPO_ROOT / ".pre-commit-config.yaml").read_text()
    assert config.count("shakespeare") >= 2, "exclude it from end-of-file-fixer and whitespace"


def test_train_py_states_the_checkpoints_targets():
    """train.py prints the targets; they must be the ones the checkpoint checks."""
    train = (M / "train.py").read_text()
    test = (M / "tests" / "test_checkpoint.py").read_text()
    for name in ["TARGET_VAL_LOSS", "TARGET_REAL_WORDS"]:
        pattern = rf"^{name} = ([\d.]+)"
        assert re.search(pattern, train, re.M).group(1) == re.search(pattern, test, re.M).group(1)
