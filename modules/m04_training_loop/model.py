"""A tiny GPT that reads and writes one character at a time. Given to you: M4 is about training it.

Every part is one you met in M2: token and position embeddings, causal self-attention,
a small MLP, layer norm, and a final layer that scores every possible next token.
With the default sizes it has about 4.8 million parameters (SmolLM2-135M has 135 million).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    """One transformer block: attention, then an MLP, each with layer norm and a shortcut."""

    def __init__(self, n_embd: int, n_head: int, dropout: float):
        super().__init__()
        self.n_head = n_head
        self.ln1 = nn.LayerNorm(n_embd)
        self.qkv = nn.Linear(n_embd, 3 * n_embd)  # queries, keys and values in one layer
        self.proj = nn.Linear(n_embd, n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.GELU(), nn.Linear(4 * n_embd, n_embd)
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(c, dim=2)
        # Split each into n_head smaller heads: (b, t, c) -> (b, n_head, t, c // n_head).
        q, k, v = (z.view(b, t, self.n_head, c // self.n_head).transpose(1, 2) for z in (q, k, v))
        # The same causal attention you wrote in M2, done by PyTorch's faster built-in.
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).reshape(b, t, c)
        x = x + self.drop(self.proj(y))
        return x + self.drop(self.mlp(self.ln2(x)))


class TinyGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        block_size: int = 256,
        n_embd: int = 256,
        n_head: int = 4,
        n_layer: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, dropout) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """Token ids (batch, time) -> logits (batch, time, vocab): scores for each next token."""
        t = idx.shape[1]
        positions = torch.arange(t, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(positions)
        return self.head(self.ln(self.blocks(x)))

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample max_new_tokens more ids after idx, one at a time (the M2 loop)."""
        was_training = self.training
        self.eval()
        for _ in range(max_new_tokens):
            logits = self(idx[:, -self.block_size :])[:, -1, :]
            probs = F.softmax(logits / temperature, dim=-1)
            next_id = torch.multinomial(probs.cpu(), 1, generator=generator).to(idx.device)
            idx = torch.cat([idx, next_id], dim=1)
        self.train(was_training)
        return idx


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
