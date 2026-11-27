"""M7 is written for beginners who have done M1 to M6: these pin the structure that makes it so.

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

M = REPO_ROOT / "modules" / "m07_lora_tools"
M5 = REPO_ROOT / "modules" / "m05_full_finetune"
STARTER_NB = M / "starter" / "lesson.ipynb"
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

SECTIONS = [
    "## 0. Why use a tool?",
    "## 1. Two tools",
    "## 2. The data format",
    "## 3. LoRA in mlx-lm",
    "## 4. LoRA in PEFT",
    "## 5. A short run with each",
    "## 6. Tokens per second",
    "## 7. Sequence length and memory",
    "## 8. Saving and loading an adapter",
    "## 9. Your task",
]

TERMS = [
    "mlx-lm",
    "PEFT",
    "adapter config",
    "keys",
    "chat record",
    "JSONL",
    "tokens per second",
    "padding",
    "sequence length",
    "bf16",
    "adapter file",
]

EXERCISES = [
    "to_chat_record",
    "mlx_lora_config",
    "peft_lora_config",
    "tokens_per_sec",
    "load_peft_adapter",
    "load_mlx_adapter",
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
    pytest.importorskip("mlx_lm")
    pytest.importorskip("peft")
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    if not _model_cached():
        pytest.skip(f"{MODEL} isn't downloaded on this machine")
    ignore = shutil.ignore_patterns(".ipynb_checkpoints", "__pycache__")
    work = tmp_path / "modules"
    shutil.copytree(M, work / M.name, ignore=ignore)
    shutil.copytree(M5 / "data", work / M5.name / "data")  # the lesson reads M5's data
    nb = nbformat.read(work / M.name / "starter" / "lesson.ipynb", as_version=4)
    client = nbclient.NotebookClient(
        nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(work / M.name)}}
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
    defs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef | ast.ClassDef)}
    return source.splitlines(), defs


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


def test_hints_name_every_exercise():
    names = {e.get("name") for e in tomllib.loads((M / "hints.toml").read_text())["exercise"]} - {
        None
    }
    assert names == set(EXERCISES)


# --- the README ---

SYNTAX = [
    '`{"role": "user", "content": text}`',
    "`json.dumps(record)`",
    "`alpha / r`",
    "`LoraConfig(r=..., lora_alpha=..., target_modules=[...])`",
    "`get_peft_model(model, config)`",
    "`model.print_trainable_parameters()`",
    "`linear_to_lora_layers(model, num_layers, lora)`",
    "`int(mask.sum())`",
    "`model.save_pretrained(folder)`",
    "`PeftModel.from_pretrained(base, folder)`",
    "`load(name, adapter_path=folder)`",
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


# --- specific to M7 ---


def test_lesson_shows_the_tools_the_exercises_need():
    code = _code()
    for tool in [
        '"messages"',
        "json.dumps(record)",
        '"scale": alpha / r',
        '"self_attn.q_proj"',
        "linear_to_lora_layers(",
        "LoraConfig(",
        "lora_alpha=16",
        "get_peft_model(",
        'int(b["attention_mask"].sum())',
        "save_pretrained(",
        "PeftModel.from_pretrained(",
        "adapter_path=",
    ]:
        assert tool in code, f"the lesson never shows {tool!r}"


def test_hints_are_nudges_not_solutions():
    hints = {
        str(e["id"]): e["hint"] for e in tomllib.loads((M / "hints.toml").read_text())["exercise"]
    }
    assert "alpha / r" not in hints["2"]
    assert "lora_alpha" not in hints["3"]
    assert "from_pretrained" not in hints["5"]
    assert "adapter_path" not in hints["6"]


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


def test_readme_states_the_checkpoints_targets():
    test = (M / "tests" / "test_checkpoint.py").read_text()
    readme = (M / "README.md").read_text()
    lesson = _markdown()

    def const(name):
        return float(re.search(rf"^{name} = ([\d.]+)", test, re.M).group(1))

    assert f"at most **{const('LOSS_BAR'):g}**" in readme
    assert f"at least **{const('STYLE_GAIN'):g}** chrF" in readme
    assert f"at least **{100 * const('RELOAD_SHARE'):g}%**" in readme
    assert f"under **{const('MAX_PEAK_GB'):g} GB**" in readme
    assert f"{const('LOSS_BAR'):g} or lower" in lesson
    assert f"at least {const('STYLE_GAIN'):g} chrF" in lesson
    assert f"at least {100 * const('RELOAD_SHARE'):g}%" in lesson
    assert f"under {const('MAX_PEAK_GB'):g} GB" in lesson


def test_train_py_and_the_lesson_agree_on_the_adapter():
    train = (M / "train.py").read_text()
    assert 'TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]' in train
    assert '"--r", type=int, default=8' in train
    assert '"--alpha", type=float, default=16.0' in train
    assert "r, alpha = 8, 16" in _code()
