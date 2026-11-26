"""M6 exercises. Replace each `raise NotImplementedError` with your code.

Each exercise says in plain words what it is, then lists numbered steps as comments.
Write one or two lines of code under each step, then delete the `raise` line.
The lesson section named in each exercise shows every piece you need.

Exercise 1 is the LoRA layer itself. Exercise 2 counts what it adds. Exercises 3 and 4 put
LoRA inside a real model and count what trains. Exercise 5 folds the adapter back into the
weights when you're done. train.py also uses your M5 exercises for the batches and the loss.

Check your work any time with:  make check M=06
Stuck?  make hint M=06 EX=1  (a nudge, logged on your dashboard)
"""

import torch  # noqa: F401  (you'll want torch.randn, torch.zeros and torch.no_grad)
import torch.nn as nn


class LoRALinear(nn.Module):
    """Exercise 1. A frozen nn.Linear plus a small trainable update, B @ A.

    In plain words: keep the original layer (base) exactly as it is and freeze it. Next to it,
    make two small matrices: A, which squeezes the input down to r numbers, and B, which
    spreads them back out to the output size. B starts at zero, so at first the layer gives
    exactly what base gives. The output is base(x) plus (alpha / r) times x pushed through A
    and then B. See lesson section 2.
    """

    def __init__(self, base, r, alpha):
        super().__init__()
        # 1. self.base = base
        # 2. Freeze it: for each p in self.base.parameters(): p.requires_grad_(False)
        # 3. w = base.weight, so you can make A and B on its device and with its dtype.
        #    self.A = nn.Parameter(torch.randn(r, base.in_features, device=w.device,
        #    dtype=w.dtype) / base.in_features ** 0.5)
        # 4. self.B = nn.Parameter(torch.zeros(base.out_features, r, device=..., dtype=...))
        # 5. self.scale = alpha / r
        raise NotImplementedError("Exercise 1: LoRALinear.__init__")

    def forward(self, x):
        # 1. Return self.base(x) + self.scale * (x @ self.A.T @ self.B.T)
        raise NotImplementedError("Exercise 1: LoRALinear.forward")


def lora_params(in_features, out_features, r):
    """Exercise 2. How many trainable numbers a rank-r adapter adds to one layer.

    In plain words: A is r by in_features and B is out_features by r. Add up the numbers in
    both. lora_params(10, 4, 1) is 14. See lesson section 1.
    """
    # 1. Return r * in_features + out_features * r  (or, shorter, r * (in + out)).
    raise NotImplementedError("Exercise 2: lora_params")


def inject_lora(model, targets, r, alpha):
    """Exercise 3. Put a LoRALinear in place of every target layer, and freeze the rest.

    In plain words: first freeze every weight in the model. Then walk through all its layers;
    each has a dotted name like "model.layers.0.self_attn.q_proj". If the last part of the
    name is in targets (like "q_proj") and the layer is an nn.Linear, swap it for a
    LoRALinear wrapped around it. Return the list of the names you swapped. See lesson
    section 4.
    """
    # 1. For each p in model.parameters(): p.requires_grad_(False)
    # 2. replaced = []
    # 3. For each name, module in list(model.named_modules()):   (list() first: you're about
    #    to change the model while walking through it)
    # 4.   parent_name, _, child_name = name.rpartition(".")
    #      ("model.layers.0.self_attn.q_proj" -> "model.layers.0.self_attn", ".", "q_proj")
    # 5.   If child_name in targets and isinstance(module, nn.Linear):
    # 6.     parent = model.get_submodule(parent_name)
    # 7.     setattr(parent, child_name, LoRALinear(module, r, alpha)), then
    #        replaced.append(name)
    # 8. Return replaced.
    raise NotImplementedError("Exercise 3: inject_lora")


def count_trainable(model):
    """Exercise 4. How many of the model's numbers will train, and how many there are in all.

    In plain words: add up p.numel() for the parameters whose requires_grad is True, and for
    all parameters. Return both, as (trainable, total). See lesson section 4.
    """
    # 1. trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # 2. total = the same sum without the if.
    # 3. Return trainable, total.
    raise NotImplementedError("Exercise 4: count_trainable")


def merge_lora(layer):
    """Exercise 5. Fold a trained adapter back into one plain nn.Linear.

    In plain words: the LoRA layer computes base(x) + scale * (x @ A.T @ B.T), which is the same
    as one layer whose weight is W + scale * (B @ A). Make a fresh nn.Linear of the same size,
    copy that weight into it, and copy the bias if there is one. See lesson section 6.
    """
    # 1. base = layer.base
    # 2. merged = nn.Linear(base.in_features, base.out_features, bias=base.bias is not None,
    #    device=base.weight.device, dtype=base.weight.dtype)
    # 3. Inside `with torch.no_grad():` (no training happens here):
    # 4.   merged.weight.copy_(base.weight + layer.scale * layer.B @ layer.A)
    # 5.   If base.bias is not None: merged.bias.copy_(base.bias)
    # 6. Return merged.
    raise NotImplementedError("Exercise 5: merge_lora")
