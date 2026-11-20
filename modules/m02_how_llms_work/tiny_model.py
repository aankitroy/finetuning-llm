"""A tiny Llama-style model with random weights, built in code (no download).

It has the same parts as SmolLM2, just far smaller, so the checkpoint and
`generate.py --tiny` run in a second. Its "text" is meaningless: the weights
were never trained.
"""

import torch
from transformers import LlamaConfig, LlamaForCausalLM


def make_tiny_model(seed: int = 0) -> LlamaForCausalLM:
    torch.manual_seed(seed)
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )
    return LlamaForCausalLM(config).eval()
