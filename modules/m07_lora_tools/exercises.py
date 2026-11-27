"""M7 exercises. Replace each `raise NotImplementedError` with your code.

Each exercise says in plain words what it is, then lists numbered steps as comments.
Write one or two lines of code under each step, then delete the `raise` line.
The lesson section named in each exercise shows every piece you need.

Exercise 1 turns an M5 example into the format the tools read. Exercises 2 and 3 set up LoRA
in mlx-lm and in PEFT: the same rank 8 adapter you built in M6, described two ways.
Exercise 4 measures speed. Exercise 5 loads a saved adapter back, in each tool.
train.py also uses your M5 exercises for the batches.

Check your work any time with:  make check M=07
Stuck?  make hint M=07 EX=1  (a nudge, logged on your dashboard)
"""


def to_chat_record(pair, system):
    """Exercise 1. One M5 example as a chat: a system message, the user's text, the answer.

    In plain words: the tools read training data as a list of messages, each a dict with a
    "role" and its "content". Make three: the system prompt, the user's jargon
    (pair["jargon"]), and the assistant's plain rewrite (pair["plain"]). Return them inside a
    dict under the key "messages". See lesson section 2.
    """
    # 1. system_msg = {"role": "system", "content": system}
    # 2. user_msg = {"role": "user", "content": pair["jargon"]}
    # 3. answer_msg = {"role": "assistant", "content": pair["plain"]}
    # 4. Return {"messages": [system_msg, user_msg, answer_msg]}  (in that order)
    raise NotImplementedError("Exercise 1: to_chat_record")


def mlx_lora_config(r, alpha, targets, num_layers):
    """Exercise 2. The LoRA settings mlx-lm reads: how many layers, and the adapter on each.

    In plain words: mlx-lm wants two things. "num_layers": how many of the model's blocks get
    adapters (counted from the top). "lora_parameters": a dict with the rank, the scale, the
    dropout, and the "keys", the layers to wrap. mlx-lm takes the scale itself, not alpha:
    work it out the way you did in M6. Its keys are named from inside a block, so "q_proj"
    becomes "self_attn.q_proj". See lesson section 3.
    """
    # 1. keys = ["self_attn." + t for t in targets]
    # 2. lora = {"rank": r, "scale": ..., "dropout": 0.0, "keys": keys}
    #    (the scale is the number your M6 LoRALinear multiplied B @ A by)
    # 3. Return {"num_layers": num_layers, "lora_parameters": lora}
    raise NotImplementedError("Exercise 2: mlx_lora_config")


def peft_lora_config(r, alpha, targets):
    """Exercise 3. The same adapter, described for Hugging Face's PEFT.

    In plain words: PEFT keeps its settings in a LoraConfig. Unlike mlx-lm, it takes alpha
    itself (lora_alpha) and works out the scale on its own. Its target_modules are the plain
    layer names, like "q_proj". Set the dropout to 0, as in M6, and tell it the task is a
    causal language model, so it knows how to wrap one. See lesson section 4.
    """
    # 1. from peft import LoraConfig
    # 2. Return LoraConfig(r=r, lora_alpha=..., target_modules=list(targets),
    #    lora_dropout=0.0, task_type="CAUSAL_LM")
    raise NotImplementedError("Exercise 3: peft_lora_config")


def tokens_per_sec(attention_masks, seconds):
    """Exercise 4. How many real tokens went through training each second.

    In plain words: each batch has an attention mask: 1 for a real token, 0 for padding.
    Count the 1s in every mask (padding isn't work the model learns from), add them up, and
    divide by how many seconds training took. See lesson section 6.
    """
    # 1. tokens = 0
    # 2. For each mask in attention_masks: tokens += int(mask.sum())
    # 3. Return tokens / seconds
    raise NotImplementedError("Exercise 4: tokens_per_sec")


def load_peft_adapter(base_model, adapter_dir):
    """Exercise 5a. Put a saved PEFT adapter back on a fresh copy of the model.

    In plain words: train.py saved the adapter's A and B (and its LoraConfig) in adapter_dir.
    PEFT has a class, PeftModel, that wraps a base model and loads an adapter from a folder in
    one call. Return the wrapped model. See lesson section 8.
    """
    # 1. from peft import PeftModel
    # 2. Return PeftModel.from_pretrained(base_model, adapter_dir)
    raise NotImplementedError("Exercise 5: load_peft_adapter")


def load_mlx_adapter(model_path, adapter_dir):
    """Exercise 5b. Load the model in mlx-lm, with a saved adapter on it.

    In plain words: mlx-lm's load function reads a model (and its tokenizer) and, if you tell
    it where an adapter folder is, wraps the right layers and loads A and B into them. Return
    what it returns: the model and the tokenizer. See lesson section 8.
    """
    # 1. from mlx_lm import load
    # 2. Return load(model_path, adapter_path=adapter_dir)
    raise NotImplementedError("Exercise 5: load_mlx_adapter")
