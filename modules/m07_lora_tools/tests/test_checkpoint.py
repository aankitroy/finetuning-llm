"""M7 checkpoint: the same LoRA, in mlx-lm and in PEFT, trained, saved and loaded back.

Run `uv run python modules/m07_lora_tools/train.py` first, then `make check M=07`.
"""

import hashlib
import importlib.util
import json
import os
import sys

import pytest
import torch
from conftest import MODULE_DIR

from common.checkpoint import check_answer
from common.progress import report_metric

sys.path.insert(0, str(MODULE_DIR))
from tiny_models import make_tiny_hf_model, save_tiny_mlx_model  # noqa: E402


def _mlx_works() -> bool:
    """mlx-lm is installed and MLX can use a GPU (Apple-silicon Macs only)."""
    if sys.platform != "darwin" or importlib.util.find_spec("mlx_lm") is None:
        return False
    try:
        import mlx.core as mx

        return bool(mx.metal.is_available())
    except Exception:
        return False


HAS_MLX = _mlx_works()
needs_mlx = pytest.mark.skipif(not HAS_MLX, reason="mlx-lm runs only on Apple-silicon Macs")
ATTN = ["q_proj", "k_proj", "v_proj", "o_proj"]
SYSTEM = "Rewrite the user's text in plain English."


# --- exercise 1 ---


def test_to_chat_record(ex):
    pair = {"jargon": "We will leverage synergies.", "plain": "We'll work together."}
    rec = ex.to_chat_record(pair, SYSTEM)
    assert isinstance(rec, dict) and list(rec) == ["messages"], (
        'Return a dict with one key, "messages": {"messages": [...]}'
    )
    msgs = rec["messages"]
    assert [m.get("role") for m in msgs] == ["system", "user", "assistant"], (
        "Three messages, in order: system, user, assistant"
    )
    assert [m["content"] for m in msgs] == [SYSTEM, pair["jargon"], pair["plain"]], (
        "The system prompt, then the jargon (the user's text), then the plain rewrite (the "
        "answer the model should learn)"
    )


# --- exercise 2 ---


def test_mlx_lora_config(ex):
    cfg = ex.mlx_lora_config(8, 16, ATTN, 28)
    assert isinstance(cfg, dict) and set(cfg) == {"num_layers", "lora_parameters"}, (
        'Return {"num_layers": ..., "lora_parameters": {...}}'
    )
    assert cfg["num_layers"] == 28, "num_layers is passed in: keep it as it is"
    lora = cfg["lora_parameters"]
    assert set(lora) == {"rank", "scale", "dropout", "keys"}, (
        "lora_parameters holds four things: rank, scale, dropout and keys"
    )
    assert lora["rank"] == 8
    assert lora["scale"] != 16, (
        "mlx-lm wants the scale, not alpha. In M6 the update was multiplied by alpha / r"
    )
    assert lora["scale"] == 2.0, f"scale = alpha / r = 16 / 8 = 2, not {lora['scale']}"
    assert lora["dropout"] == 0.0
    assert lora["keys"] == [f"self_attn.{t}" for t in ATTN], (
        'mlx-lm names layers from inside a block: "q_proj" is "self_attn.q_proj"'
    )
    other = ex.mlx_lora_config(4, 16, ["q_proj"], 2)["lora_parameters"]
    assert other["scale"] == 4.0 and other["keys"] == ["self_attn.q_proj"], (
        "Use the r, alpha and targets you're given: rank 4, alpha 16 is a scale of 4"
    )


# --- exercise 3 ---


def test_peft_lora_config(ex):
    from peft import LoraConfig

    cfg = ex.peft_lora_config(8, 16, ATTN)
    assert isinstance(cfg, LoraConfig), "Return a peft LoraConfig"
    assert cfg.r == 8
    assert cfg.lora_alpha != 2.0, (
        "PEFT wants alpha itself (lora_alpha=16) and works out the scale on its own"
    )
    assert cfg.lora_alpha == 16, f"lora_alpha should be alpha (16), not {cfg.lora_alpha}"
    assert set(cfg.target_modules) == set(ATTN), (
        f"target_modules should be the plain names {ATTN}, not {cfg.target_modules}"
    )
    assert cfg.lora_dropout == 0.0
    assert str(cfg.task_type).endswith("CAUSAL_LM"), 'task_type="CAUSAL_LM"'


# --- exercise 4 ---


def test_tokens_per_sec(ex):
    masks = [torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]), torch.tensor([[1, 1, 1, 1, 1]])]
    got = ex.tokens_per_sec(masks, 2.0)
    assert got != 6.5, (
        "Count only the 1s in each mask: the 0s are padding, not tokens the model learned from"
    )
    assert got == pytest.approx(5.0), f"(5 + 5) real tokens in 2 seconds is 5.0, not {got}"


# --- exercises 5 and 6 ---


def _logits_hf(model, ids):
    model.eval()
    with torch.no_grad():
        return model(input_ids=ids).logits


def test_load_peft_adapter(ex, tmp_path):
    from peft import LoraConfig, get_peft_model

    ids = torch.tensor([[2, 5, 9, 14, 3]])
    base_logits = _logits_hf(make_tiny_hf_model(), ids)
    cfg = LoraConfig(r=4, lora_alpha=8, target_modules=ATTN, init_lora_weights=False)
    trained = get_peft_model(make_tiny_hf_model(), cfg)  # B starts random, so it changes things
    want = _logits_hf(trained, ids)
    trained.save_pretrained(tmp_path)
    loaded = ex.load_peft_adapter(make_tiny_hf_model(), str(tmp_path))
    assert loaded is not None, "Return the model PeftModel.from_pretrained gives you"
    got = _logits_hf(loaded, ids)
    assert not torch.allclose(got, base_logits), (
        "That's the model without its adapter. Load the adapter onto it: "
        "PeftModel.from_pretrained(base_model, adapter_dir)"
    )
    assert torch.allclose(got, want, atol=1e-5), "The loaded model should match the saved one"


@needs_mlx
def test_load_mlx_adapter(ex, tmp_path):
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm import load
    from mlx_lm.tuner.utils import linear_to_lora_layers

    folder = save_tiny_mlx_model(tmp_path / "tiny")
    ids = mx.array([[2, 5, 9, 14, 3]])
    model, _ = load(str(folder))
    base_logits = model(ids)
    model.freeze()
    lora = {"rank": 4, "scale": 2.0, "dropout": 0.0, "keys": [f"self_attn.{t}" for t in ATTN]}
    linear_to_lora_layers(model, 2, lora)
    mx.random.seed(1)
    model.load_weights(  # B starts at zero: give it something to say
        [
            (n, mx.random.normal(p.shape) * 0.5)
            for n, p in tree_flatten(model.trainable_parameters())
            if n.endswith("lora_b")
        ],
        strict=False,
    )
    want = model(ids)
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    mx.save_safetensors(
        str(adapter / "adapters.safetensors"), dict(tree_flatten(model.trainable_parameters()))
    )
    (adapter / "adapter_config.json").write_text(
        json.dumps({"fine_tune_type": "lora", "num_layers": 2, "lora_parameters": lora})
    )
    out = ex.load_mlx_adapter(str(folder), str(adapter))
    assert isinstance(out, tuple) and len(out) == 2, (
        "Return what mlx_lm.load returns: (model, tokenizer)"
    )
    got = out[0](ids)
    assert not mx.allclose(got, base_logits).item(), (
        "That's the model without its adapter. Pass adapter_path=adapter_dir to load"
    )
    assert mx.allclose(got, want, atol=1e-4).item(), "The loaded model should match the saved one"


# --- the runs ---

LOSS_BAR = 0.80  # the held-out loss after training must be at most this (1.16 untrained)
STYLE_GAIN = 3.0  # rewrites must score at least this much more chrF than the untuned model's
RELOAD_SHARE = 0.95  # the reloaded adapter must give the same reply for this share of prompts
MAX_PEAK_GB = 18.0  # each tool's run must fit in this much memory


def _run(outputs, tool):
    f = outputs / tool / "run.json"
    if tool == "mlx" and not HAS_MLX:
        pytest.skip("mlx-lm runs only on Apple-silicon Macs")
    assert f.exists(), (
        f"No {tool} run found. Run: uv run python modules/m07_lora_tools/train.py --tool {tool}"
    )
    return json.loads(f.read_text())


@pytest.mark.parametrize("tool", ["mlx", "peft"])
def test_run_trains_and_reloads(outputs, tool):
    s = _run(outputs, tool)
    if tool == ("mlx" if HAS_MLX else "peft"):
        for key in ("final_loss", "peak_mem_gb", "tokens_per_sec", "trainable_pct"):
            report_metric(key, s[key])
    need = round(RELOAD_SHARE * s["reload_total"])
    assert s["reload_matches"] >= need, (
        f"The adapter loaded back from disk gave the same reply for only {s['reload_matches']} "
        f"of {s['reload_total']} prompts. Loading must put back exactly what was trained: check "
        f"your load_{tool}_adapter (exercise {5 if tool == 'peft' else 6})."
    )
    if s["tiny"]:
        assert s["heldout_after"] < s["heldout_before"], "The tiny run should still learn"
        return
    assert s["heldout_after"] <= LOSS_BAR, (
        f"{tool}'s held-out loss after training is {s['heldout_after']:.3f}; it needs to be at "
        f"most {LOSS_BAR:g} (it was {s['heldout_before']:.3f} before). Try the defaults: rank 8, "
        "alpha 16, learning rate 2e-4, 2 epochs."
    )
    gain = s["tuned"]["chrf"] - s["base"]["chrf"]
    assert gain >= STYLE_GAIN, (
        f"{tool}'s rewrites scored chrF {s['tuned']['chrf']:.1f} against the reference rewrites, "
        f"vs {s['base']['chrf']:.1f} before training: they should be at least {STYLE_GAIN:g} "
        "points closer to the house style."
    )
    assert s["peak_mem_gb"] <= MAX_PEAK_GB, (
        f"{tool}'s run peaked at {s['peak_mem_gb']:.1f} GB; it should fit in {MAX_PEAK_GB:g} GB. "
        "Close other apps and the lesson notebook, and keep --batch 8 and --max-len 160."
    )


@pytest.mark.parametrize("tool", ["mlx", "peft"])
def test_run_was_full_size_with_your_code(outputs, tool):
    if os.environ.get("FTCOURSE_SOLUTIONS"):
        pytest.skip("reference-solution run (used by the project's own tests)")
    s = _run(outputs, tool)
    assert not s["tiny"], "That was a --tiny run. Run train.py without --tiny."
    assert not s.get("used_solutions"), (
        "That run used the reference solutions. Re-run train.py with your own code."
    )
    now = hashlib.sha256((MODULE_DIR / "exercises.py").read_bytes()).hexdigest()
    assert s.get("exercises_sha256") == now, (
        "exercises.py has changed since your last run. Run train.py again, so the run uses the "
        "code you're being checked on."
    )


ANSWER_IDS = sorted(json.loads((MODULE_DIR / "tests" / "answer_hashes.json").read_text()))


@pytest.mark.parametrize("qid", ANSWER_IDS)
def test_understanding(qid):
    ok, message = check_answer(MODULE_DIR, "m07", qid)
    assert ok, message
