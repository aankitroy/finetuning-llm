"""M2 checkpoint: your generation loop matches model.generate() token for token.

Run `uv run python modules/m02_how_llms_work/generate.py` first, then `make check M=02`.
"""

import json
import os

import pytest
import torch
from conftest import MODULE_DIR
from torch.nn import functional as F

from common.checkpoint import check_answer
from common.progress import report_metric

PROMPT = torch.tensor([[1, 5, 9, 22, 7]])


def test_next_token_logits(ex, tiny):
    logits = ex.next_token_logits(tiny, PROMPT)
    assert logits.shape == (128,), (
        "Return the logits for the last position only: one number per vocabulary token, "
        f"shape (128,) here. Got {tuple(logits.shape)}."
    )
    with torch.no_grad():
        expected = tiny(PROMPT).logits[0, -1]
    assert torch.allclose(logits, expected), "Take model(input_ids).logits, then [0, -1]"


def test_greedy(ex):
    choice = ex.greedy(torch.tensor([0.1, 2.0, -1.0]))
    assert isinstance(choice, int), "Return a plain Python int (use int(...) or .item())"
    assert choice == 1, "Greedy picks the position of the biggest logit"


def test_softmax_with_temperature(ex):
    logits = torch.tensor([2.0, 1.0, 0.1])
    p1 = ex.softmax_with_temperature(logits, 1.0)
    assert torch.isclose(p1.sum(), torch.tensor(1.0)), "Probabilities must add up to 1"
    assert torch.allclose(p1, torch.softmax(logits, dim=-1)), "At temperature 1, it's plain softmax"
    sharp = ex.softmax_with_temperature(logits, 0.5).max()
    flat = ex.softmax_with_temperature(logits, 2.0).max()
    assert sharp > p1.max() > flat, (
        "Divide the logits by the temperature before softmax: low temperature should make the top "
        "token more likely, high temperature less"
    )


def test_top_p_filter(ex):
    probs = torch.tensor([0.5, 0.3, 0.15, 0.05])
    kept = ex.top_p_filter(probs, 0.75)
    assert torch.allclose(kept, torch.tensor([0.625, 0.375, 0.0, 0.0])), (
        "Keep the most likely tokens until their total reaches p (include the one that crosses "
        "it), zero the rest, then divide by the new total so it sums to 1. "
        f"Got {kept.tolist()}"
    )
    shuffled = torch.tensor([0.05, 0.3, 0.5, 0.15])
    assert torch.allclose(ex.top_p_filter(shuffled, 0.75), torch.tensor([0.0, 0.375, 0.625, 0.0])), (
        "Sort by probability first: the input isn't always in order"
    )
    only_top = ex.top_p_filter(probs, 0.1)
    assert torch.allclose(only_top, torch.tensor([1.0, 0.0, 0.0, 0.0])), (
        "Always keep at least the most likely token"
    )


def test_causal_attention(ex):
    g = torch.Generator().manual_seed(0)
    q, k, v = (torch.randn(6, 8, generator=g) for _ in range(3))
    out = ex.causal_attention(q, k, v)
    assert out.shape == (6, 8), "One output row per token, same width as v"
    assert torch.allclose(out[0], v[0], atol=1e-6), (
        "The first token can only look at itself, so its output must equal v[0]. "
        "Did you apply the causal mask?"
    )
    expected = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert torch.allclose(out, expected, atol=1e-5), (
        "Scores = q @ k.T divided by the square root of the width; mask the future with -inf; "
        "softmax each row; multiply by v"
    )


def _library_greedy(model, n, eos=None):
    with torch.no_grad():
        out = model.generate(
            PROMPT, max_new_tokens=n, do_sample=False, pad_token_id=0, eos_token_id=eos
        )
    return out[0, PROMPT.shape[1] :].tolist()


def test_generate_greedy_matches_library(ex, tiny):
    mine = ex.generate(tiny, PROMPT, max_new_tokens=20)
    assert isinstance(mine, list), "Return a list of the new token ids (plain ints)"
    assert mine == _library_greedy(tiny, 20), (
        "In greedy mode (temperature 0) your tokens must equal model.generate()'s. "
        "Each step: logits for the last position, pick the biggest, append it, repeat."
    )


def test_generate_stops_at_eos(ex, tiny):
    third = _library_greedy(tiny, 3)[2]
    mine = ex.generate(tiny, PROMPT, max_new_tokens=20, eos_token_id=third)
    assert mine == _library_greedy(tiny, 20, eos=third), (
        "Stop as soon as you produce eos_token_id, and include that token in the result"
    )


def test_generate_sampling_respects_top_p(ex, tiny):
    g = torch.Generator().manual_seed(1)
    mine = ex.generate(tiny, PROMPT, max_new_tokens=10, temperature=0.8, top_p=0.5, generator=g)
    assert len(mine) == 10
    ids = PROMPT
    for token in mine:
        with torch.no_grad():
            logits = tiny(ids).logits[0, -1]
        allowed = ex.top_p_filter(torch.softmax(logits / 0.8, dim=-1), 0.5) > 0
        assert allowed[token], "With top_p, every sampled token must come from the top-p set"
        ids = torch.cat([ids, torch.tensor([[token]])], dim=1)
    again = ex.generate(
        tiny, PROMPT, max_new_tokens=10, temperature=0.8, top_p=0.5,
        generator=torch.Generator().manual_seed(1),
    )
    assert again == mine, (
        "Pass the generator to torch.multinomial so the same seed gives the same text"
    )


# --- your run of generate.py ---


def _summary(outputs):
    f = outputs / "run_summary.json"
    assert f.exists(), "No run found. Run: uv run python modules/m02_how_llms_work/generate.py"
    return json.loads(f.read_text())


def test_run_matched_the_library(outputs):
    s = _summary(outputs)
    report_metric("tokens_per_sec", s["your_tokens_per_sec"])
    assert s["matched"], (
        "Your loop and model.generate() produced different tokens in your last run. "
        f"Yours: {s['your_text']!r}  Library: {s['library_text']!r}"
    )


def test_run_used_the_real_model(outputs):
    if os.environ.get("FTCOURSE_SOLUTIONS"):
        pytest.skip("reference-solution run (used by the project's own tests)")
    s = _summary(outputs)
    assert s["model"] == "HuggingFaceTB/SmolLM2-135M", (
        "That run used the tiny test model. Run generate.py without --tiny to use the real one."
    )


def test_run_with_your_code(outputs):
    if os.environ.get("FTCOURSE_SOLUTIONS"):
        pytest.skip("reference-solution run (used by the project's own tests)")
    assert not _summary(outputs).get("used_solutions"), (
        "That run used the reference solutions. Re-run generate.py with your own code."
    )


ANSWER_IDS = sorted(json.loads((MODULE_DIR / "tests" / "answer_hashes.json").read_text()))


@pytest.mark.parametrize("qid", ANSWER_IDS)
def test_understanding(qid):
    ok, message = check_answer(MODULE_DIR, "m02", qid)
    assert ok, message
