"""M1 checkpoint: your PyTorch building blocks work, and your MLP reaches 95% accuracy.

Run `uv run python modules/m01_pytorch_essentials/train.py` first, then `make check M=01`.
"""

import json

import pytest
import torch

from common.progress import report_metric


def test_linear_forward(ex):
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    w = torch.tensor([[0.5], [-1.0]])
    b = torch.tensor([0.25])
    out = ex.linear_forward(x, w, b)
    assert out.shape == (2, 1), "Output should have one row per input and one column per output"
    assert torch.allclose(out, torch.tensor([[-1.25], [-2.25]])), "Compute x @ w + b"


def test_mse_loss(ex):
    pred = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([1.0, 0.0, 6.0])
    loss = ex.mse_loss(pred, target)
    assert loss.dim() == 0, "The loss should be a single number (a 0-d tensor)"
    assert torch.isclose(loss, torch.tensor(13 / 3)), "Mean of the squared differences"


def test_sgd_step_updates_and_zeroes_grads(ex):
    w = torch.tensor([1.0, -2.0], requires_grad=True)
    (w**2).sum().backward()  # grad = 2w = [2, -4]
    ex.sgd_step([w], lr=0.1)
    assert torch.allclose(w.detach(), torch.tensor([0.8, -1.6])), "w should become w - lr * grad"
    assert w.grad is None or torch.all(w.grad == 0), "Reset the gradient after the step"
    assert w.requires_grad, "Update the tensor in place, under torch.no_grad()"


def test_mlp_shapes_and_layers(ex):
    model = ex.MLP(in_features=2, hidden=16, out_features=2)
    out = model(torch.randn(5, 2))
    assert out.shape == (5, 2), "The MLP should return one logit per class for each input"
    linears = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]
    assert len(linears) == 2, "Use exactly two nn.Linear layers"
    assert out.requires_grad, "Logits must stay connected to the graph (no .detach())"


def test_train_one_epoch_reduces_loss(ex):
    from data import make_moons

    torch.manual_seed(0)
    X, y = make_moons(n=512, noise=0.1, seed=0)
    model = ex.MLP(in_features=2, hidden=32, out_features=2)
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    first = ex.train_one_epoch(model, X, y, opt, batch_size=64)
    for _ in range(10):
        last = ex.train_one_epoch(model, X, y, opt, batch_size=64)
    assert isinstance(first, float), "Return the mean loss as a Python float (use .item())"
    assert last < first * 0.7, f"Loss should fall when training: {first:.3f} -> {last:.3f}"


def test_accuracy(ex):
    class Fixed(torch.nn.Module):
        def forward(self, x):
            return x  # logits are the inputs themselves

    X = torch.tensor([[2.0, 1.0], [0.0, 3.0], [5.0, 4.0], [1.0, 0.0]])
    y = torch.tensor([0, 1, 1, 0])
    assert ex.accuracy(Fixed(), X, y) == pytest.approx(0.75)


def test_trained_model_reaches_95_percent(outputs):
    summary = outputs / "run_summary.json"
    assert summary.exists(), (
        "No training run found. Run: uv run python modules/m01_pytorch_essentials/train.py"
    )
    s = json.loads(summary.read_text())
    report_metric("accuracy", s["val_accuracy"])
    assert s["val_accuracy"] >= 0.95, (
        f"Validation accuracy is {s['val_accuracy']:.1%}; it needs to reach 95%. "
        "Try more epochs, a bigger hidden layer, or check your train_one_epoch."
    )


@pytest.mark.optional_checkpoint
def test_trained_on_the_mac_gpu(outputs):
    if not torch.backends.mps.is_available():
        pytest.skip("MPS isn't available on this machine")
    s = json.loads((outputs / "run_summary.json").read_text())
    assert s["device"] == "mps", "train.py should pick the 'mps' device when it's available"
