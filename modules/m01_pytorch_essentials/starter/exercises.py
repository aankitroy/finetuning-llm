"""M1 exercises. Replace each `raise NotImplementedError` with your code.

Check your work any time with:  make check M=01
"""

import torch
from torch import nn


def linear_forward(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Exercise 1. A linear layer by hand: return x @ w + b.

    x: (batch, in), w: (in, out), b: (out,)  ->  (batch, out)
    """
    raise NotImplementedError("Exercise 1: linear_forward")


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Exercise 2. Mean squared error, as a single-number tensor. Don't use nn.MSELoss."""
    raise NotImplementedError("Exercise 2: mse_loss")


def sgd_step(params: list[torch.Tensor], lr: float) -> None:
    """Exercise 3. One step of gradient descent, by hand.

    For each parameter: p = p - lr * p.grad, done in place inside `torch.no_grad()`,
    then reset its gradient (p.grad = None) so the next backward() starts fresh.
    """
    raise NotImplementedError("Exercise 3: sgd_step")


class MLP(nn.Module):
    """Exercise 4. A 2-layer network: Linear -> ReLU -> Linear.

    forward() returns raw logits (no softmax) of shape (batch, out_features).
    """

    def __init__(self, in_features: int, hidden: int, out_features: int):
        super().__init__()
        raise NotImplementedError("Exercise 4: build the layers in MLP.__init__")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Exercise 4: MLP.forward")


def train_one_epoch(model, X, y, optimizer, batch_size: int) -> float:
    """Exercise 5. One pass over the data in shuffled mini-batches.

    For each batch: forward -> cross-entropy loss -> zero_grad -> backward -> step.
    Return the mean loss over the batches as a Python float.
    """
    raise NotImplementedError("Exercise 5: train_one_epoch")


@torch.no_grad()
def accuracy(model, X, y) -> float:
    """Exercise 6. Fraction of examples where argmax(logits) equals the label."""
    raise NotImplementedError("Exercise 6: accuracy")
