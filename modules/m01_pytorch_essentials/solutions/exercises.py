"""M1 reference solutions. Try the exercises yourself first!"""

import torch
from torch import nn
from torch.nn import functional as F


def linear_forward(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return x @ w + b


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return ((pred - target) ** 2).mean()


def sgd_step(params: list[torch.Tensor], lr: float) -> None:
    with torch.no_grad():
        for p in params:
            p -= lr * p.grad
            p.grad = None


class MLP(nn.Module):
    def __init__(self, in_features: int, hidden: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.fc2 = nn.Linear(hidden, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


def train_one_epoch(model, X, y, optimizer, batch_size: int) -> float:
    model.train()
    perm = torch.randperm(len(X), device=X.device)
    losses = []
    for i in range(0, len(X), batch_size):
        idx = perm[i : i + batch_size]
        loss = F.cross_entropy(model(X[idx]), y[idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return sum(losses) / len(losses)


@torch.no_grad()
def accuracy(model, X, y) -> float:
    model.eval()
    return (model(X).argmax(dim=1) == y).float().mean().item()
