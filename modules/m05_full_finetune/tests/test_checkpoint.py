"""M5 checkpoint: your fine-tune makes a real model write plain English.

Run `uv run python modules/m05_full_finetune/train.py --loop` and `... --trainer` first,
then `make check M=05`.
"""

import json
import math
import os
import sys

import pytest
import torch
import torch.nn as nn
from conftest import MODULE_DIR

from common.checkpoint import check_answer
from common.progress import report_metric

sys.path.insert(0, str(MODULE_DIR))
from tiny_model import make_tiny_model  # noqa: E402

LOSS_DROP = 0.4  # how far the held-out loss on the reference rewrites must fall
CHRF_SLACK = 3.0  # the tuned model's chrF may be at most this far below the base model's


def test_build_example(ex):
    ids, labels = ex.build_example([5, 6, 7], [8, 9])
    assert ids == [5, 6, 7, 8, 9], "input_ids is the prompt followed by the answer"
    assert labels == [-100, -100, -100, 8, 9], (
        "labels copies the answer and puts -100 on every prompt token, so the loss ignores "
        f"them. Got {labels}"
    )
    assert isinstance(ids, list) and isinstance(labels, list), "Return two plain lists"
    ids, labels = ex.build_example([1, 2, 3], [4, 5, 6], max_len=4)
    assert ids == [1, 2, 3, 4] and labels == [-100, -100, -100, 4], (
        "With max_len, keep only the first max_len tokens of both lists"
    )
    ids, labels = ex.build_example([1, 2], [3], max_len=10)
    assert ids == [1, 2, 3] and labels == [-100, -100, 3], "Shorter than max_len: keep it all"


def test_collate(ex):
    examples = [([1, 2, 3], [-100, 2, 3]), ([4, 5], [-100, 5])]
    batch = ex.collate(examples, pad_id=0)
    assert set(batch) >= {"input_ids", "attention_mask", "labels"}, (
        "Return a dict with input_ids, attention_mask and labels"
    )
    assert batch["input_ids"].tolist() == [[1, 2, 3], [4, 5, 0]], (
        "Pad every row on the right with pad_id, up to the longest row"
    )
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]], (
        "attention_mask is 1 for real tokens and 0 for padding (M3 section 5)"
    )
    assert batch["labels"].tolist() == [[-100, 2, 3], [-100, 5, -100]], (
        "Pad labels with -100, not pad_id: padding must not count in the loss"
    )
    for k in ("input_ids", "attention_mask", "labels"):
        assert batch[k].dtype == torch.long, f"{k} should be a long tensor"


def test_masked_lm_loss(ex):
    model = make_tiny_model()
    ids = torch.randint(3, 128, (2, 9), generator=torch.Generator().manual_seed(0))
    labels = ids.clone()
    labels[:, :4] = -100
    labels[1, 7:] = -100
    logits = model(input_ids=ids).logits
    got = ex.masked_lm_loss(logits, labels)
    want = model(input_ids=ids, labels=labels).loss
    assert got.dim() == 0, "Return one number (a 0-d tensor)"
    assert torch.allclose(got, want, atol=1e-5), (
        "Shift by one: the logits at position t guess the label at t + 1, so drop the last "
        "logit and the first label. Then cross-entropy with ignore_index=-100. "
        f"Expected {want.item():.4f} (what the model itself computes), got {got.item():.4f}"
    )
    flat = torch.zeros(1, 3, 10)
    one = ex.masked_lm_loss(flat, torch.tensor([[-100, 4, -100]]))
    assert math.isclose(one.item(), math.log(10), rel_tol=1e-5), (
        "Only the positions that aren't -100 count: here just one, with an even guess = ln 10"
    )


def test_optimizer_state_bytes(ex):
    w = nn.Parameter(torch.zeros(10, 4))
    b = nn.Parameter(torch.zeros(4))
    opt = torch.optim.AdamW([w, b], lr=0.1)
    assert ex.optimizer_state_bytes(opt) == 0, "Before the first step, AdamW has stored nothing"
    (w.sum() + b.sum()).backward()
    opt.step()
    got = ex.optimizer_state_bytes(opt)
    # exp_avg and exp_avg_sq for 44 float32 numbers, plus a 4-byte float32 step count each
    want = 2 * 44 * 4 + 2 * 4
    assert got == want, (
        "Add up numel() * element_size() for every tensor in optimizer.state.values(). AdamW "
        f"keeps two numbers per weight plus a step count. Expected {want} bytes, got {got}"
    )
    assert isinstance(got, int), "Return a whole number of bytes"


def test_chrf(ex):
    assert math.isclose(ex.chrf("the cat sat", "the cat sat"), 100.0), "Identical text scores 100"
    assert ex.chrf("", "something") == 0.0, "An empty rewrite scores 0"
    assert ex.chrf("xyz", "abc") == 0.0, "No characters in common scores 0"
    got = ex.chrf("ab", "abc", max_n=2)
    assert math.isclose(got, 100 * 35 / 55, rel_tol=1e-6), (
        "For 'ab' vs 'abc' with max_n=2: 1-grams P=1, R=2/3; 2-grams P=1, R=1/2. Average P=1 "
        f"and R=7/12, then F = (1 + 4) * P * R / (4 * P + R) = 63.64. Got {got}"
    )
    assert math.isclose(ex.chrf("a b", "ab", max_n=2), 100.0), (
        "Remove the spaces first: chrF compares characters, not words"
    )
    assert math.isclose(ex.chrf("aaa", "a", max_n=1), 100 * 5 / 3 / (4 / 3 + 1), rel_tol=1e-6), (
        "Clip the matches: 'a' can only match as often as it appears in the reference (P = 1/3)"
    )
    near = ex.chrf("We will talk again later.", "We'll talk again later.")
    far = ex.chrf("Leverage synergies going forward.", "We'll talk again later.")
    assert near > 70 > 20 > far, "A close rewrite should score far higher than an unrelated one"


def test_count_buzzwords(ex):
    words = ["leverage", "circle back", "best-in-class"]
    assert ex.count_buzzwords("Let's circle back and LEVERAGE this.", words) == 2, (
        "Lowercase the text, and match multi-word phrases too"
    )
    assert ex.count_buzzwords("A best in class, best-in-class team.", words) == 2, (
        "Turn every non-letter (hyphens, commas) into a space, in the text and the buzzwords"
    )
    assert ex.count_buzzwords("Leveraged circles backed.", words) == 0, (
        "Match whole words only: 'leveraged' is not 'leverage'"
    )
    assert ex.count_buzzwords("leverage leverage leverage", words) == 3, "Count every time"
    assert ex.count_buzzwords("", words) == 0


# --- your runs of train.py ---


def _summary(outputs, mode):
    f = outputs / f"run_{mode}.json"
    assert f.exists(), (
        f"No {mode} run found. Run: uv run python modules/m05_full_finetune/train.py --{mode}"
    )
    return json.loads(f.read_text())


@pytest.mark.parametrize("mode", ["loop", "trainer"])
def test_run_beat_the_base_model(outputs, mode):
    s = _summary(outputs, mode)
    if mode == "loop":
        report_metric("final_loss", s["val_loss"])
        report_metric("peak_mem_gb", s["peak_mem_gb"])
        report_metric("tokens_per_sec", s["tokens_per_sec"])
        report_metric("trainable_pct", s["trainable_pct"])
    if s["tiny"]:
        assert s["val_loss"] < s["initial_val_loss"], "The tiny run should still learn"
        return
    base, tuned = s["base"], s["tuned"]
    assert s["val_loss"] <= s["initial_val_loss"] - LOSS_DROP, (
        f"Your {mode} run's loss on the held-out rewrites went from {s['initial_val_loss']:.3f} "
        f"to {s['val_loss']:.3f}. It needs to fall by at least {LOSS_DROP}: that's the sign the "
        "model has learned the plain style."
    )
    assert tuned["buzzwords"] <= base["buzzwords"], (
        f"Your {mode} run left {tuned['buzzwords']:.2f} buzzwords per rewrite; the original "
        f"model left {base['buzzwords']:.2f}. Fine-tuning should remove them, not add them."
    )
    assert tuned["chrf"] >= base["chrf"] - CHRF_SLACK, (
        f"Your {mode} run's chrF was {tuned['chrf']:.1f}, the original model's {base['chrf']:.1f}. "
        "The rewrites have drifted away from the meaning: look at the examples train.py printed."
    )


@pytest.mark.parametrize("mode", ["loop", "trainer"])
def test_run_was_full_size_with_your_code(outputs, mode):
    if os.environ.get("FTCOURSE_SOLUTIONS"):
        pytest.skip("reference-solution run (used by the project's own tests)")
    s = _summary(outputs, mode)
    assert not s["tiny"], f"That was a --tiny run. Run train.py --{mode} without --tiny."
    assert not s.get("used_solutions"), (
        f"That run used the reference solutions. Re-run train.py --{mode} with your own code."
    )


ANSWER_IDS = sorted(json.loads((MODULE_DIR / "tests" / "answer_hashes.json").read_text()))


@pytest.mark.parametrize("qid", ANSWER_IDS)
def test_understanding(qid):
    ok, message = check_answer(MODULE_DIR, "m05", qid)
    assert ok, message
