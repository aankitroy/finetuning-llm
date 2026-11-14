"""A toy two-class dataset: two interleaving half-moons (no scikit-learn needed)."""

import math

import torch


def make_moons(n: int = 1000, noise: float = 0.2, seed: int = 0):
    """Return ``X`` of shape (n, 2) and labels ``y`` of shape (n,) with values 0 or 1."""
    g = torch.Generator().manual_seed(seed)
    n0 = n // 2
    n1 = n - n0
    t0 = torch.rand(n0, generator=g) * math.pi
    t1 = torch.rand(n1, generator=g) * math.pi
    upper = torch.stack([torch.cos(t0), torch.sin(t0)], dim=1)
    lower = torch.stack([1 - torch.cos(t1), 0.5 - torch.sin(t1)], dim=1)
    X = torch.cat([upper, lower]) + noise * torch.randn(n, 2, generator=g)
    y = torch.cat([torch.zeros(n0), torch.ones(n1)]).long()
    perm = torch.randperm(n, generator=g)
    return X[perm], y[perm]
