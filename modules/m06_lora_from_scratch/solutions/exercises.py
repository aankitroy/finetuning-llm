"""M6 reference solutions. Use `make solution M=06 EX=3` instead of reading this file."""

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base, r, alpha):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        w = base.weight
        self.A = nn.Parameter(
            torch.randn(r, base.in_features, device=w.device, dtype=w.dtype) / base.in_features**0.5
        )
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=w.device, dtype=w.dtype))
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + self.scale * (x @ self.A.T @ self.B.T)


def lora_params(in_features, out_features, r):
    return r * (in_features + out_features)


def inject_lora(model, targets, r, alpha):
    for p in model.parameters():
        p.requires_grad_(False)
    replaced = []
    for name, module in list(model.named_modules()):
        parent_name, _, child_name = name.rpartition(".")
        if child_name in targets and isinstance(module, nn.Linear):
            parent = model.get_submodule(parent_name)
            setattr(parent, child_name, LoRALinear(module, r, alpha))
            replaced.append(name)
    return replaced


def count_trainable(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def merge_lora(layer):
    base = layer.base
    merged = nn.Linear(
        base.in_features,
        base.out_features,
        bias=base.bias is not None,
        device=base.weight.device,
        dtype=base.weight.dtype,
    )
    with torch.no_grad():
        merged.weight.copy_(base.weight + layer.scale * layer.B @ layer.A)
        if base.bias is not None:
            merged.bias.copy_(base.bias)
    return merged
