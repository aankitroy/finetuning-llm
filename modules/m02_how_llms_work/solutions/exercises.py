"""M2 reference solutions. Try the exercises yourself first!"""

import torch
from torch.nn import functional as F


@torch.no_grad()
def next_token_logits(model, input_ids: torch.Tensor) -> torch.Tensor:
    return model(input_ids).logits[0, -1]


def greedy(logits: torch.Tensor) -> int:
    return int(logits.argmax())


def softmax_with_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    return torch.softmax(logits / temperature, dim=-1)


def top_p_filter(probs: torch.Tensor, p: float) -> torch.Tensor:
    sorted_probs, order = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    keep_sorted = (cumulative - sorted_probs) < p
    keep = torch.zeros_like(probs, dtype=torch.bool)
    keep[order[keep_sorted]] = True
    kept = torch.where(keep, probs, torch.zeros_like(probs))
    return kept / kept.sum()


def causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    scores = q @ k.T / q.shape[-1] ** 0.5
    n = q.shape[0]
    mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(mask, float("-inf"))
    return F.softmax(scores, dim=-1) @ v


def generate(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 0.0,
    top_p: float = 1.0,
    eos_token_id: int | None = None,
    generator: torch.Generator | None = None,
) -> list[int]:
    new_tokens = []
    for _ in range(max_new_tokens):
        logits = next_token_logits(model, input_ids)
        if temperature == 0:
            token = greedy(logits)
        else:
            probs = top_p_filter(softmax_with_temperature(logits, temperature), top_p)
            token = int(torch.multinomial(probs, 1, generator=generator))
        new_tokens.append(token)
        next_id = torch.tensor([[token]], device=input_ids.device)
        input_ids = torch.cat([input_ids, next_id], dim=1)
        if token == eos_token_id:
            break
    return new_tokens
