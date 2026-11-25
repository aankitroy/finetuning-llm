"""A tiny Qwen2-style model with random weights, built in code (no download).

It has the same parts as Qwen2.5-0.5B, just far smaller, so the checkpoint tests and
`train.py --tiny` run in seconds on the CPU. Its "text" is meaningless.
"""

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM


def make_tiny_model(seed: int = 0, vocab_size: int = 128) -> Qwen2ForCausalLM:
    torch.manual_seed(seed)
    config = Qwen2Config(
        vocab_size=vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        tie_word_embeddings=True,
    )
    return Qwen2ForCausalLM(config)
