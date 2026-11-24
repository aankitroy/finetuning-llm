"""M4 reference solutions. Use `make solution M=04 EX=3` instead of reading this file."""

import math

import torch
import torch.nn.functional as F


def get_batch(data, block_size, batch_size, generator=None):
    starts = torch.randint(len(data) - block_size, (batch_size,), generator=generator)
    x = torch.stack([data[i : i + block_size] for i in starts])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in starts])
    return x, y


def causal_lm_loss(logits, targets):
    batch, time, vocab = logits.shape
    return F.cross_entropy(logits.reshape(batch * time, vocab), targets.reshape(batch * time))


def lr_at(step, max_lr, min_lr, warmup_steps, total_steps):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= total_steps:
        return min_lr
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def clip_grad_norm(params, max_norm):
    grads = [p.grad for p in params if p.grad is not None]
    total = torch.sqrt(sum((g**2).sum() for g in grads)).item()
    if total > max_norm:
        scale = max_norm / total
        for g in grads:
            g.mul_(scale)
    return total


def train_step(model, batches, optimizer, max_norm):
    optimizer.zero_grad()
    total = 0.0
    for x, y in batches:
        loss = causal_lm_loss(model(x), y) / len(batches)
        loss.backward()
        total += loss.item()
    clip_grad_norm(list(model.parameters()), max_norm)
    optimizer.step()
    return total


def estimate_loss(model, data, block_size, batch_size, iters, generator=None):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for _ in range(iters):
            x, y = get_batch(data, block_size, batch_size, generator)
            total += causal_lm_loss(model(x), y).item()
    model.train()
    return total / iters
