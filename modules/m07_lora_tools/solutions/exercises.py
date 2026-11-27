"""M7 reference solutions. Use `make solution M=07 EX=3` instead of reading this file."""


def to_chat_record(pair, system):
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": pair["jargon"]},
            {"role": "assistant", "content": pair["plain"]},
        ]
    }


def mlx_lora_config(r, alpha, targets, num_layers):
    return {
        "num_layers": num_layers,
        "lora_parameters": {
            "rank": r,
            "scale": alpha / r,
            "dropout": 0.0,
            "keys": ["self_attn." + t for t in targets],
        },
    }


def peft_lora_config(r, alpha, targets):
    from peft import LoraConfig

    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=list(targets),
        lora_dropout=0.0,
        task_type="CAUSAL_LM",
    )


def tokens_per_sec(attention_masks, seconds):
    tokens = sum(int(mask.sum()) for mask in attention_masks)
    return tokens / seconds


def load_peft_adapter(base_model, adapter_dir):
    from peft import PeftModel

    return PeftModel.from_pretrained(base_model, adapter_dir)


def load_mlx_adapter(model_path, adapter_dir):
    from mlx_lm import load

    return load(model_path, adapter_path=adapter_dir)
