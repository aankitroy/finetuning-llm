"""Tiny stand-ins for Qwen2.5, built in code (no download), for the tests and `train.py --tiny`.

`make_tiny_hf_model` is M5's tiny Qwen2 for PyTorch and PEFT. `save_tiny_mlx_model` writes a
tiny Qwen2 for mlx-lm to a folder, with a small tokenizer, so `mlx_lm.load` can read it like
a real model. Their "text" is meaningless.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "modules" / "m05_full_finetune"))

from tiny_model import make_tiny_model as make_tiny_hf_model  # noqa: E402, F401

TINY = {
    "model_type": "qwen2",
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "rms_norm_eps": 1e-6,
    "vocab_size": 128,
    "rope_theta": 10000.0,
    "max_position_embeddings": 256,
    "tie_word_embeddings": True,
}


def save_tiny_mlx_model(folder: Path, seed: int = 0) -> Path:
    """Write a tiny random Qwen2 (weights, config, tokenizer) that mlx_lm.load can read."""
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm.models import qwen2
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    mx.random.seed(seed)
    model = qwen2.Model(qwen2.ModelArgs.from_dict(TINY))
    mx.save_safetensors(str(folder / "model.safetensors"), dict(tree_flatten(model.parameters())))
    (folder / "config.json").write_text(json.dumps(TINY))
    vocab = {"<pad>": 0, "<end>": 1, **{f"w{i}": i for i in range(2, TINY["vocab_size"])}}
    tok = Tokenizer(WordLevel(vocab, unk_token="<pad>"))
    tok.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>", eos_token="<end>")
    fast.save_pretrained(folder)
    return folder
