"""M2 is written for beginners who have done M1: these pin the structure that makes it so.

They read the *starter* copies, so a learner who has run the lesson or written
exercises doesn't break `make test`.
"""

import ast
import json
import re
import shutil
import subprocess
import tomllib

import pytest

from tests.conftest import REPO_ROOT

M = REPO_ROOT / "modules" / "m02_how_llms_work"
STARTER_NB = M / "starter" / "lesson.ipynb"
MODEL = "HuggingFaceTB/SmolLM2-135M"

SECTIONS = [
    "## 0. What a language model does",
    "## 1. Load a real model",
    "## 2. Inside the model",
    "## 3. Attention",
    "## 4. From logits to the next token",
    "## 5. A generation loop, by hand",
    "## 6. Temperature and top-p",
    "## 7. The KV cache",
    "## 8. Your task",
]

TERMS = [
    "language model",
    "token",
    "vocabulary",
    "tokenizer",
    "embedding",
    "transformer",
    "attention",
    "query",
    "key",
    "value",
    "causal mask",
    "logits",
    "greedy",
    "end-of-text token",
    "sampling",
    "temperature",
    "top-p",
    "KV cache",
]

EXERCISES = [
    "next_token_logits",
    "greedy",
    "softmax_with_temperature",
    "top_p_filter",
    "causal_attention",
    "generate",
]


def _cells():
    return json.loads(STARTER_NB.read_text())["cells"]


def _src(cell):
    s = cell["source"]
    return "".join(s) if isinstance(s, list) else s


def _markdown():
    return "\n".join(_src(c) for c in _cells() if c["cell_type"] == "markdown")


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


def _model_cached() -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    return isinstance(try_to_load_from_cache(MODEL, "config.json"), str)


def test_lesson_runs_top_to_bottom(tmp_path):
    pytest.importorskip("transformers")
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    if not _model_cached():
        pytest.skip(f"{MODEL} isn't downloaded on this machine")
    work = tmp_path / "m02"
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
    "`AutoTokenizer.from_pretrained`",
    "`AutoModelForCausalLM.from_pretrained`",
    "`tokenizer.encode`",
    "`tokenizer.decode`",
    "`model(input_ids).logits`",
    "`torch.topk`",
    "`torch.cat`",
    "`torch.sort`",
    "`torch.cumsum`",
    "`torch.multinomial`",
    "`torch.triu`",
    "`.masked_fill`",
    "`model.generate`",
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
