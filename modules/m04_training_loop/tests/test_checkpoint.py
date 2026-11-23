"""M4 checkpoint: your training loop trains a tiny GPT that writes Shakespeare-like text.

Run `uv run python modules/m04_training_loop/train.py` first, then `make check M=04`.
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
from model import TinyGPT  # noqa: E402

TARGET_VAL_LOSS = 1.70
TARGET_REAL_WORDS = 0.80


def _tiny_model(seed=0, dropout=0.0):
    torch.manual_seed(seed)
    return TinyGPT(vocab_size=11, block_size=8, n_embd=16, n_head=2, n_layer=1, dropout=dropout)


def test_get_batch(ex):
    data = torch.arange(100, 200)
    g = torch.Generator().manual_seed(0)
    x, y = ex.get_batch(data, block_size=8, batch_size=5, generator=g)
    assert x.shape == (5, 8) and y.shape == (5, 8), (
        f"x and y should both be (batch_size, block_size) = (5, 8). Got {tuple(x.shape)} and "
        f"{tuple(y.shape)}"
    )
    assert x.dtype == data.dtype and y.dtype == data.dtype, "Keep the ids as they are (long ints)"
    for row_x, row_y in zip(x, y, strict=True):
        start = int(row_x[0]) - 100
        assert torch.equal(row_x, data[start : start + 8]), (
            "Each row of x is block_size ids in a row, cut from data: data[i : i + block_size]"
        )
        assert torch.equal(row_y, data[start + 1 : start + 9]), (
            "y is x moved along by one: data[i + 1 : i + block_size + 1]. Each y is the next "
            "character after the x in the same spot."
        )
    starts = {int(r[0]) for r in ex.get_batch(data, 8, 64, torch.Generator().manual_seed(1))[0]}
    assert len(starts) > 10, "Pick a random start for each row with torch.randint"
    assert max(starts) - 100 <= len(data) - 8 - 1, (
        "The last start must leave room for y too: torch.randint(len(data) - block_size, ...)"
    )
    a = ex.get_batch(data, 8, 4, torch.Generator().manual_seed(3))[0]
    b = ex.get_batch(data, 8, 4, torch.Generator().manual_seed(3))[0]
    assert torch.equal(a, b), "Pass generator=generator to torch.randint, so runs repeat exactly"


def test_causal_lm_loss(ex):
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 5)
    targets = torch.tensor([[1, 0, 4], [2, 2, 3]])
    got = ex.causal_lm_loss(logits, targets)
    want = nn.functional.cross_entropy(logits.reshape(6, 5), targets.reshape(6))
    assert got.dim() == 0, "Return one number (a 0-d tensor): the average over every position"
    assert torch.allclose(got, want), (
        "Flatten to (batch * time, vocab) and (batch * time,), then use F.cross_entropy. "
        f"Expected {want.item():.4f}, got {got.item():.4f}"
    )
    flat = ex.causal_lm_loss(torch.zeros(1, 4, 65), torch.zeros(1, 4, dtype=torch.long))
    assert math.isclose(flat.item(), math.log(65), rel_tol=1e-5), (
        "With all scores equal, every one of 65 characters is equally likely: loss = ln(65)"
    )
    assert (
        got.requires_grad is False
        and ex.causal_lm_loss(logits.requires_grad_(), targets).requires_grad
    ), "Keep it a tensor (no .item()), so you can call .backward() on it"


def test_lr_at(ex):
    kw = dict(max_lr=1e-3, min_lr=1e-4, warmup_steps=10, total_steps=110)
    assert math.isclose(ex.lr_at(0, **kw), 1e-4), "Warmup starts at max_lr * 1 / warmup_steps"
    assert math.isclose(ex.lr_at(4, **kw), 5e-4), "Halfway through warmup: max_lr * 5 / 10"
    assert math.isclose(ex.lr_at(9, **kw), 1e-3), "The last warmup step reaches max_lr"
    assert math.isclose(ex.lr_at(10, **kw), 1e-3), "Cosine starts from max_lr right after warmup"
    assert math.isclose(ex.lr_at(60, **kw), 5.5e-4), (
        "Halfway through the cosine part, the rate is halfway between max_lr and min_lr"
    )
    assert math.isclose(ex.lr_at(110, **kw), 1e-4) and math.isclose(ex.lr_at(500, **kw), 1e-4), (
        "From total_steps on, stay at min_lr"
    )
    rates = [ex.lr_at(s, **kw) for s in range(10, 110)]
    falling = all(a >= b for a, b in zip(rates[:-1], rates[1:], strict=True))
    assert falling, "After warmup the rate only goes down"


def _params_with_grads(*grads):
    params = []
    for g in grads:
        p = nn.Parameter(torch.zeros_like(g))
        p.grad = g.clone()
        params.append(p)
    return params


def test_clip_grad_norm(ex):
    params = _params_with_grads(torch.tensor([3.0, 0.0]), torch.tensor([[0.0, 4.0]]))
    norm = ex.clip_grad_norm(params, max_norm=1.0)
    assert math.isclose(float(norm), 5.0, rel_tol=1e-5), (
        "The total norm is the square root of the sum of every gradient value squared, across "
        f"all parameters: sqrt(3² + 4²) = 5. Got {norm}"
    )
    assert torch.allclose(params[0].grad, torch.tensor([0.6, 0.0])), (
        "Scale every gradient by max_norm / total_norm (here 1 / 5), in place"
    )
    assert torch.allclose(params[1].grad, torch.tensor([[0.0, 0.8]]))
    small = _params_with_grads(torch.tensor([0.3, 0.4]))
    assert math.isclose(float(ex.clip_grad_norm(small, max_norm=1.0)), 0.5, rel_tol=1e-5)
    assert torch.allclose(small[0].grad, torch.tensor([0.3, 0.4])), (
        "A norm already below max_norm is left alone: clipping only ever shrinks gradients"
    )
    no_grad = nn.Parameter(torch.ones(3))
    mixed = [*_params_with_grads(torch.tensor([6.0, 8.0])), no_grad]
    assert math.isclose(float(ex.clip_grad_norm(mixed, max_norm=5.0)), 10.0, rel_tol=1e-5), (
        "Skip parameters whose .grad is None"
    )
    model = _tiny_model()
    model(torch.randint(0, 11, (2, 8))).sum().backward()
    theirs = [p.grad.clone() for p in model.parameters()]
    ref = _tiny_model()
    ref(torch.randint(0, 11, (2, 8))).sum().backward()
    for p, g in zip(ref.parameters(), theirs, strict=True):
        p.grad = g.clone()
    want = nn.utils.clip_grad_norm_(ref.parameters(), 0.5)
    got = ex.clip_grad_norm(list(model.parameters()), 0.5)
    assert math.isclose(float(got), float(want), rel_tol=1e-4), "Should match PyTorch's version"
    for p, q in zip(model.parameters(), ref.parameters(), strict=True):
        assert torch.allclose(p.grad, q.grad, atol=1e-6)


def _batches(seed, n, rows):
    g = torch.Generator().manual_seed(seed)
    return [
        (torch.randint(0, 11, (rows, 8), generator=g), torch.randint(0, 11, (rows, 8), generator=g))
        for _ in range(n)
    ]


def test_train_step(ex):
    ((x, y),) = _batches(0, 1, 8)
    big = _tiny_model()
    nn.functional.cross_entropy(big(x).reshape(-1, 11), y.reshape(-1)).backward()
    want = [p.grad.clone() for p in big.parameters()]

    model = _tiny_model()
    for p in model.parameters():  # stale gradients from "an earlier step"
        p.grad = torch.full_like(p, 7.0)
    opt = torch.optim.SGD(model.parameters(), lr=0.0)
    halves = [(x[:4], y[:4]), (x[4:], y[4:])]
    loss = ex.train_step(model, halves, opt, max_norm=1e9)
    for p, g in zip(model.parameters(), want, strict=True):
        assert torch.allclose(p.grad, g, atol=1e-5), (
            "Two half-batches should give the same gradients as one full batch. Zero the old "
            "gradients first, and divide each micro-batch's loss by len(batches) before "
            ".backward()."
        )
    full = nn.functional.cross_entropy(big(x).reshape(-1, 11), y.reshape(-1)).item()
    assert isinstance(loss, float), "Return a plain float (use .item())"
    assert math.isclose(loss, full, rel_tol=1e-4), (
        f"Return the average loss over the micro-batches ({full:.4f}), got {loss:.4f}"
    )

    model = _tiny_model()
    opt = torch.optim.SGD(model.parameters(), lr=0.0)
    ex.train_step(model, _batches(1, 2, 4), opt, max_norm=0.01)
    total = torch.sqrt(sum((p.grad**2).sum() for p in model.parameters())).item()
    assert total <= 0.01 + 1e-6, "Clip the gradients (exercise 4) before optimizer.step()"

    model = _tiny_model()
    before = [p.detach().clone() for p in model.parameters()]
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    ex.train_step(model, _batches(2, 1, 4), opt, max_norm=1.0)
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters(), strict=True)), (
        "Call optimizer.step() so the weights actually change"
    )


def test_estimate_loss(ex):
    model = _tiny_model(dropout=0.5)
    data = torch.randint(0, 11, (200,), generator=torch.Generator().manual_seed(0))
    got = ex.estimate_loss(model, data, 8, 4, iters=3, generator=torch.Generator().manual_seed(5))
    assert isinstance(got, float), "Return a plain float"
    assert model.training, "Put the model back in training mode (model.train()) at the end"
    assert all(p.grad is None for p in model.parameters()), (
        "Measuring the loss must not compute gradients: wrap it in torch.no_grad()"
    )
    g = torch.Generator().manual_seed(5)
    model.eval()
    want = []
    with torch.no_grad():
        for _ in range(3):
            ix = torch.randint(len(data) - 8, (4,), generator=g)
            x = torch.stack([data[i : i + 8] for i in ix])
            y = torch.stack([data[i + 1 : i + 9] for i in ix])
            want.append(nn.functional.cross_entropy(model(x).reshape(-1, 11), y.reshape(-1)).item())
    model.train()
    assert math.isclose(got, sum(want) / 3, rel_tol=1e-4), (
        "Average the loss over `iters` batches from get_batch, in eval mode (dropout off). "
        f"Expected {sum(want) / 3:.4f}, got {got:.4f}"
    )


# --- your run of train.py ---


def _summary(outputs):
    f = outputs / "run_summary.json"
    assert f.exists(), "No run found. Run: uv run python modules/m04_training_loop/train.py"
    return json.loads(f.read_text())


def test_run_reached_the_target_loss(outputs):
    s = _summary(outputs)
    report_metric("final_loss", s["val_loss"])
    report_metric("peak_mem_gb", s["peak_mem_gb"])
    if s["tiny"]:
        assert s["val_loss"] < s["initial_val_loss"] - 0.5, "The tiny run should still learn"
        return
    assert s["val_loss"] < TARGET_VAL_LOSS, (
        f"Validation loss in your last run was {s['val_loss']:.3f}; it needs to be below "
        f"{TARGET_VAL_LOSS}. train.py prints the loss as it goes: look at where it stopped falling."
    )


def test_samples_look_like_text(outputs):
    s = _summary(outputs)
    if s["tiny"]:
        pytest.skip("the tiny run is too short to write words")
    assert s["real_word_rate"] >= TARGET_REAL_WORDS, (
        f"Only {s['real_word_rate']:.0%} of the words in your sample appear in the training text "
        f"(need {TARGET_REAL_WORDS:.0%}). Read outputs/m04/samples.txt."
    )


def test_run_was_full_size(outputs):
    if os.environ.get("FTCOURSE_SOLUTIONS"):
        pytest.skip("reference-solution run (used by the project's own tests)")
    assert not _summary(outputs)["tiny"], (
        "That was a --tiny run. Run train.py without --tiny for the real model."
    )


def test_run_with_your_code(outputs):
    if os.environ.get("FTCOURSE_SOLUTIONS"):
        pytest.skip("reference-solution run (used by the project's own tests)")
    assert not _summary(outputs).get("used_solutions"), (
        "That run used the reference solutions. Re-run train.py with your own code."
    )


ANSWER_IDS = sorted(json.loads((MODULE_DIR / "tests" / "answer_hashes.json").read_text()))


@pytest.mark.parametrize("qid", ANSWER_IDS)
def test_understanding(qid):
    ok, message = check_answer(MODULE_DIR, "m04", qid)
    assert ok, message
