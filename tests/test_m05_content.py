"""M5 is written for beginners who have done M1 to M4: these pin the structure that makes it so.

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

M = REPO_ROOT / "modules" / "m05_full_finetune"
STARTER_NB = M / "starter" / "lesson.ipynb"
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

SECTIONS = [
    "## 0. What fine-tuning changes",
    "## 1. The original model tries the task",
    "## 2. Formatting the pairs, and padding a batch",
    "## 3. Loss masking: only the answer is graded",
    "## 4. Where the memory goes",
    "## 5. Fine-tuning for real",
    "## 6. The same run with `Trainer`",
    "## 7. Scoring the rewrites",
    "## 8. Forgetting",
    "## 9. Your task",
]

TERMS = [
    "fine-tuning",
    "pretrained weights",
    "fine-tuned weights",
    "full fine-tune",
    "loss masking",
    "optimizer state",
    "activations",
    "epoch",
    "`Trainer`",
    "`TrainingArguments`",
    "data collator",
    "chrF",
    "precision",
    "recall",
    "F-score",
    "buzzwords",
    "catastrophic forgetting",
]

EXERCISES = [
    "build_example",
    "collate",
    "masked_lm_loss",
    "optimizer_state_bytes",
    "chrf",
    "count_buzzwords",
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


def _model_cached() -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    return isinstance(try_to_load_from_cache(MODEL, "model.safetensors"), str)


def test_lesson_runs_top_to_bottom(tmp_path):
    pytest.importorskip("transformers")
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    if not _model_cached():
        pytest.skip(f"{MODEL} isn't downloaded on this machine")
    work = tmp_path / "m05"
    shutil.copytree(M, work, ignore=shutil.ignore_patterns(".ipynb_checkpoints", "__pycache__"))
    nb = nbformat.read(work / "starter" / "lesson.ipynb", as_version=4)
    client = nbclient.NotebookClient(
        nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(work)}}
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
    "`list_a + list_b`",
    "`[-100] * n`",
    "`items[:max_len]`",
    "`max(len(x) for x in rows)`",
    "`torch.tensor(rows, dtype=torch.long)`",
    "`logits[:, :-1, :]`, `labels[:, 1:]`",
    "`F.cross_entropy(..., ignore_index=-100)`",
    "`optimizer.state`",
    "`torch.is_tensor(x)`",
    "`.numel()`, `.element_size()`",
    "`TrainingArguments(...)`, `Trainer(...).train()`",
    '`.replace(" ", "")`',
    "`c.isalpha()`",
    "`words[i : i + n] == phrase`",
    "`continue`",
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


# --- specific to M5 ---


def test_lesson_shows_the_tools_the_exercises_need():
    code = _code()
    for tool in [
        "[-100] * len(",
        "[pad_id] * pad",
        "[1] * len(",
        "max(len(",
        "dtype=torch.long",
        "[:, :-1, :]",
        "[:, 1:]",
        "ignore_index=-100",
        "optimizer.state.values()",
        "torch.is_tensor(",
        ".numel() * value.element_size()",
        '.replace(" ", "")',
        "range(len(text) - n + 1)",
        "min(c, r.get(g, 0))",
        "continue",
        "c.isalpha()",
        ".split()",
        "w[i : i + n] == phrase",
        "TrainingArguments(",
        "data_collator=",
    ]:
        assert tool in code, f"the lesson never shows {tool!r}"


def test_hints_are_nudges_not_solutions():
    hints = {
        str(e["id"]): e["hint"] for e in tomllib.loads((M / "hints.toml").read_text())["exercise"]
    }
    assert "[:, :-1" not in hints["3"]
    assert "min(c" not in hints["5"]
    assert "isalpha" not in hints["6"]


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


def _rows(split):
    return [json.loads(line) for line in (M / "data" / f"{split}.jsonl").read_text().splitlines()]


def test_the_data_is_clean():
    """The splits are the right size, never overlap, and the plain side is buzzword-free."""
    import importlib.util

    pytest.importorskip("torch")  # the reference counter lives next to torch code
    spec = importlib.util.spec_from_file_location("sol", M / "solutions" / "exercises.py")
    sol = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sol)
    buzzwords = [w for w in (M / "data" / "buzzwords.txt").read_text().splitlines() if w.strip()]
    splits = {s: _rows(s) for s in ("train", "val", "test")}
    assert {s: len(r) for s, r in splits.items()} == {"train": 400, "val": 50, "test": 50}
    jargon = [{r["jargon"] for r in rows} for rows in splits.values()]
    assert sum(map(len, jargon)) == len(set().union(*jargon)) == 500, "a sentence is repeated"
    for rows in splits.values():
        for r in rows:
            assert set(r) == {"jargon", "plain"}
            assert sol.count_buzzwords(r["plain"], buzzwords) == 0, r["plain"]
    assert len(_rows("general")) == 10


def test_readme_states_the_checkpoints_targets():
    test = (M / "tests" / "test_checkpoint.py").read_text()
    readme = (M / "README.md").read_text()
    drop = re.search(r"^LOSS_DROP = ([\d.]+)", test, re.M).group(1)
    slack = re.search(r"^CHRF_SLACK = ([\d.]+)", test, re.M).group(1)
    assert f"at least **{drop}**" in readme
    assert f"within **{float(slack):g}** points" in readme
