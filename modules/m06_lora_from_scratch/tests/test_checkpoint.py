"""M6 checkpoint: your LoRA layers tune a real model with under 1% of its weights.

Run `uv run python modules/m06_lora_from_scratch/train.py` first, then `make check M=06`.
"""

import hashlib
import json
import os
import sys

import pytest
import torch
import torch.nn as nn
from conftest import MODULE_DIR, ROOT

from common.checkpoint import check_answer
from common.progress import report_metric

sys.path.insert(0, str(ROOT / "modules" / "m05_full_finetune"))
from tiny_model import make_tiny_model  # noqa: E402

MAX_TRAINABLE_PCT = 1.0  # LoRA must train less than this share of the weights
MARGIN = 0.05  # the held-out loss may be at most 5% above your M5 full fine-tune's
MERGE_TOLERANCE = 0.01  # merging must not change the held-out loss by more than this
ATTN = ["q_proj", "k_proj", "v_proj", "o_proj"]


def _layer(ex, r=4, alpha=8.0):
    torch.manual_seed(0)
    return ex.LoRALinear(nn.Linear(16, 8), r=r, alpha=alpha)


def test_lora_linear_starts_equal_to_the_base(ex):
    layer = _layer(ex)
    assert tuple(layer.A.shape) == (4, 16), (
        f"A should have shape (r, in_features) = (4, 16), not {tuple(layer.A.shape)}"
    )
    assert tuple(layer.B.shape) == (8, 4), (
        f"B should have shape (out_features, r) = (8, 4), not {tuple(layer.B.shape)}"
    )
    assert isinstance(layer.A, nn.Parameter) and isinstance(layer.B, nn.Parameter), (
        "Wrap A and B in nn.Parameter, so the optimizer can find and train them"
    )
    assert torch.count_nonzero(layer.B) == 0, "B starts as all zeros (torch.zeros)"
    assert torch.count_nonzero(layer.A) > 0, "A starts random (torch.randn), not zeros"
    x = torch.randn(3, 5, 16)
    assert torch.allclose(layer(x), layer.base(x)), (
        "At the start the layer must give exactly the base layer's output: B is zero, so "
        "B @ A adds nothing"
    )
    assert not layer.base.weight.requires_grad and not layer.base.bias.requires_grad, (
        "Freeze the base layer: requires_grad_(False) on each of its parameters"
    )
    assert layer.A.requires_grad and layer.B.requires_grad, "A and B are the trainable part"
    assert layer.scale == 2.0, f"scale is alpha / r = 8 / 4 = 2, not {layer.scale}"


def test_lora_linear_adds_the_scaled_update(ex):
    layer = _layer(ex)
    with torch.no_grad():
        layer.B.copy_(torch.randn(8, 4))
    x = torch.randn(3, 16)
    want = layer.base(x) + 2.0 * x @ (layer.B @ layer.A).T
    assert torch.allclose(layer(x), want, atol=1e-5), (
        "The output is base(x) + scale * (x @ A.T @ B.T): the base layer plus the small update "
        "B @ A, times alpha / r"
    )
    layer(x).sum().backward()
    assert layer.B.grad is not None and layer.base.weight.grad is None, (
        "Training should reach B and A only, never the frozen base weight"
    )


def test_lora_linear_follows_the_base_dtype(ex):
    base = nn.Linear(16, 8).to(torch.bfloat16)
    layer = ex.LoRALinear(base, r=2, alpha=4.0)
    assert layer.A.dtype == torch.bfloat16 and layer.B.dtype == torch.bfloat16, (
        "Make A and B with the base weight's dtype (dtype=base.weight.dtype), and its device too"
    )


def test_lora_linear_is_made_where_the_base_lives(ex):
    # "meta" tensors have a shape but no numbers: a stand-in for the Mac's GPU in tests
    layer = ex.LoRALinear(nn.Linear(16, 8, device="meta"), r=2, alpha=4.0)
    assert isinstance(layer.A, nn.Parameter) and isinstance(layer.B, nn.Parameter), (
        "A and B must still be nn.Parameters. Put device= inside torch.randn/torch.zeros: "
        "calling .to(...) on an nn.Parameter returns a plain tensor, which never trains"
    )
    assert layer.A.device.type == "meta" and layer.B.device.type == "meta", (
        "Make A and B on the base weight's device (device=base.weight.device), or they'll be "
        "on the CPU while the model is on the GPU"
    )


def test_lora_params(ex):
    assert ex.lora_params(896, 896, 8) == 14336, "r * (in + out) = 8 * (896 + 896)"
    assert ex.lora_params(896, 128, 8) == 8192, "k_proj and v_proj are 896 -> 128 in Qwen2.5-0.5B"
    assert ex.lora_params(10, 4, 1) == 14
    layer = _layer(ex)
    count = sum(p.numel() for p in [layer.A, layer.B])
    assert ex.lora_params(16, 8, 4) == count, "It should match the size of A plus the size of B"


def test_inject_lora(ex):
    model = make_tiny_model()
    ids = torch.randint(3, 128, (2, 7), generator=torch.Generator().manual_seed(0))
    before = model(input_ids=ids).logits
    replaced = ex.inject_lora(model, ATTN, r=2, alpha=4.0)
    assert isinstance(replaced, list), "Return the list of the names you replaced"
    assert len(replaced) == 8, (
        f"The tiny model has 2 layers x 4 attention projections = 8 to replace, not {len(replaced)}"
    )
    assert all(name.split(".")[-1] in ATTN for name in replaced), (
        f"Only replace layers whose own name is a target. Got {replaced[:3]}"
    )
    assert isinstance(model.get_submodule("model.layers.0.self_attn.q_proj"), ex.LoRALinear), (
        "Put the LoRALinear where the old layer was: setattr(parent, child_name, new_layer)"
    )
    assert isinstance(model.get_submodule("model.layers.0.mlp.up_proj"), nn.Linear), (
        "Leave layers that aren't targets alone"
    )
    after = model(input_ids=ids).logits
    assert torch.allclose(before, after, atol=1e-6), (
        "Injecting LoRA must not change the model's output before training"
    )
    wrong = [
        n for n, p in model.named_parameters() if p.requires_grad and not n.endswith((".A", ".B"))
    ]
    assert not wrong, f"Freeze every original weight first (step 1). Still trainable: {wrong[:3]}"
    adapters = [p for n, p in model.named_parameters() if n.endswith((".A", ".B"))]
    assert adapters and all(p.requires_grad for p in adapters), (
        "The new A and B matrices must be trainable: freeze the model first (step 1), then swap "
        "in the LoRA layers (step 7). Freezing afterwards freezes the adapters too."
    )
    try:
        again = ex.inject_lora(model, ATTN, r=2, alpha=4.0)
    except AttributeError:
        again = None
    assert again == [], (
        "Calling inject_lora twice should replace nothing the second time: a LoRALinear isn't "
        "an nn.Linear (check isinstance(module, nn.Linear))"
    )


def test_count_trainable(ex):
    model = make_tiny_model()
    total = sum(p.numel() for p in model.parameters())
    assert tuple(ex.count_trainable(model)) == (total, total), (
        "Before LoRA, every parameter is trainable: return (trainable, total)"
    )
    ex.inject_lora(model, ATTN, r=2, alpha=4.0)
    trainable, new_total = ex.count_trainable(model)
    # 2 layers, rank 2: q and o are 32 -> 32, k and v are 32 -> 16
    want = 2 * 2 * (2 * (32 + 32) + 2 * (32 + 16))
    assert trainable == want, (
        f"Count only the parameters with requires_grad: expected {want}, got {trainable}"
    )
    assert new_total == total + want, "total counts every parameter, frozen or not"


def test_merge_lora(ex):
    layer = _layer(ex)
    with torch.no_grad():
        layer.B.copy_(torch.randn(8, 4))
    before = layer.base.weight.clone()
    merged = ex.merge_lora(layer)
    assert torch.equal(layer.base.weight, before), (
        "Don't change layer.base: make a fresh nn.Linear and put the merged weight in that. "
        "Changing the base in place makes the LoRA layer add the update twice."
    )
    assert type(merged) is nn.Linear, f"Return a plain nn.Linear, not {type(merged).__name__}"
    x = torch.randn(5, 16)
    assert torch.allclose(merged(x), layer(x), atol=1e-5), (
        "The merged weight is W + scale * (B @ A), and the bias is copied: it must give the same "
        "output as the LoRA layer"
    )
    with torch.no_grad():
        layer.B.zero_()
    assert not torch.allclose(merged(x), layer(x)), (
        "The merged layer should hold its own copy of the weight, not keep reading B and A"
    )
    no_bias = ex.LoRALinear(nn.Linear(16, 8, bias=False), r=2, alpha=2.0)
    try:
        merged_no_bias = ex.merge_lora(no_bias)
    except AttributeError:
        merged_no_bias = None
    assert merged_no_bias is not None and merged_no_bias.bias is None, (
        "A base layer without a bias (like Qwen's o_proj) merges without one: only copy the bias "
        "if base.bias is not None (step 5)"
    )
    on_meta = ex.LoRALinear(nn.Linear(16, 8, device="meta"), r=2, alpha=4.0)
    try:
        device = ex.merge_lora(on_meta).weight.device.type
    except (NotImplementedError, RuntimeError):  # copying between devices fails
        device = None
    assert device == "meta", (
        "Make the merged layer on the base weight's device (device=base.weight.device, step 2)"
    )


# --- your run of train.py ---


def _summary(outputs):
    f = outputs / "run.json"
    assert f.exists(), "No run found. Run: uv run python modules/m06_lora_from_scratch/train.py"
    return json.loads(f.read_text())


def test_run_is_small_and_good(outputs):
    s = _summary(outputs)
    report_metric("final_loss", s["heldout_after"])
    report_metric("trainable_pct", s["trainable_pct"])
    assert abs(s["heldout_merged"] - s["heldout_after"]) < MERGE_TOLERANCE, (
        f"Merging changed the held-out loss from {s['heldout_after']:.3f} to "
        f"{s['heldout_merged']:.3f}. A merged layer must compute exactly what the LoRA layer did."
    )
    if s["tiny"]:  # a tiny model is mostly attention, so it can't get under 1%
        assert s["heldout_after"] < s["heldout_before"], "The tiny run should still learn"
        return
    assert s["trainable_pct"] < MAX_TRAINABLE_PCT, (
        f"Your run trained {s['trainable_pct']:.2f}% of the weights; LoRA should train under "
        f"{MAX_TRAINABLE_PCT:g}%. Use a smaller --r, or fewer --targets."
    )
    limit = s["m05_heldout"] * (1 + MARGIN)
    assert s["heldout_after"] <= limit, (
        f"Your LoRA run's held-out loss is {s['heldout_after']:.3f}; {s['m05_source']} reached "
        f"{s['m05_heldout']:.3f}, so it needs to be at most {limit:.3f} (within "
        f"{100 * MARGIN:g}%). Try the defaults: rank 8, learning rate 2e-4, 2 epochs."
    )


def test_run_was_full_size_with_your_code(outputs):
    if os.environ.get("FTCOURSE_SOLUTIONS"):
        pytest.skip("reference-solution run (used by the project's own tests)")
    s = _summary(outputs)
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
    ok, message = check_answer(MODULE_DIR, "m06", qid)
    assert ok, message
