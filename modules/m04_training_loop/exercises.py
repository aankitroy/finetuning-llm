"""M4 exercises. Replace each `raise NotImplementedError` with your code.

Each exercise says in plain words what it is, then lists numbered steps as comments.
Write one or two lines of code under each step, then delete the `raise` line.
The lesson section named in each exercise shows every piece you need.

Together these six are a complete training loop. train.py plugs them into the tiny GPT
in model.py and trains it on Shakespeare. In M5 you'll reuse them on a real model.

Check your work any time with:  make check M=04
Stuck?  make hint M=04 EX=1  (a nudge, logged on your dashboard)
"""

import math  # noqa: F401  (you'll want math.cos and math.pi)

import torch  # noqa: F401  (you'll want torch.randint and torch.stack)
import torch.nn.functional as F  # noqa: F401  (you'll want F.cross_entropy)


def get_batch(data, block_size, batch_size, generator=None):
    """Exercise 1. Cut a batch of random training examples out of the text.

    In plain words: pick batch_size random places in data. At each place, x is the next
    block_size ids, and y is the same window moved along by one, so y[t] is the character
    that comes after x[t]. Return (x, y), each shaped (batch_size, block_size).
    See lesson section 2.
    """
    # 1. Pick batch_size random starts: torch.randint(len(data) - block_size, (batch_size,),
    #    generator=generator). The "- block_size" leaves room for y's extra character.
    # 2. x: stack the windows data[i : i + block_size] for each start i (torch.stack).
    # 3. y: stack data[i + 1 : i + block_size + 1] for each start i.
    # 4. Return x, y.
    raise NotImplementedError("Exercise 1: get_batch")


def causal_lm_loss(logits, targets):
    """Exercise 2. How wrong the model's next-character guesses are, on average.

    In plain words: logits has a score for every character at every position,
    shaped (batch, time, vocab). targets has the right next character at every position,
    shaped (batch, time). Cross-entropy wants one row per guess, so flatten both first.
    See lesson section 3.
    """
    # 1. Read the sizes: batch, time, vocab = logits.shape
    # 2. Flatten logits to (batch * time, vocab) and targets to (batch * time,) with .reshape.
    # 3. Return F.cross_entropy(flat_logits, flat_targets). Keep it a tensor: no .item().
    raise NotImplementedError("Exercise 2: causal_lm_loss")


def lr_at(step, max_lr, min_lr, warmup_steps, total_steps):
    """Exercise 3. The learning rate to use at a given step: warmup, then cosine.

    In plain words: for the first warmup_steps steps, climb in a straight line up to
    max_lr. Then glide down to min_lr along half a cosine wave, arriving at total_steps.
    After that, stay at min_lr. See lesson section 5.
    """
    # 1. If step < warmup_steps: return max_lr * (step + 1) / warmup_steps.
    # 2. If step >= total_steps: return min_lr.
    # 3. progress = (step - warmup_steps) / (total_steps - warmup_steps), a number from 0 to 1.
    # 4. Return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress)).
    raise NotImplementedError("Exercise 3: lr_at")


def clip_grad_norm(params, max_norm):
    """Exercise 4. Shrink the gradients if, all together, they're too big.

    In plain words: treat every gradient value of every parameter as one long list and
    measure its length (the norm). If it's longer than max_norm, scale every gradient
    down by the same amount so the length becomes max_norm. Return the length from
    before clipping. See lesson section 6.
    """
    # 1. grads = the .grad of each parameter in params, skipping any that are None.
    # 2. total = the square root of the sum of every gradient value squared:
    #    torch.sqrt(sum((g**2).sum() for g in grads)).item()
    # 3. If total > max_norm: multiply each g in place by max_norm / total (g.mul_(...)).
    # 4. Return total.
    raise NotImplementedError("Exercise 4: clip_grad_norm")


def train_step(model, batches, optimizer, max_norm):
    """Exercise 5. One training step, accumulating gradients over a few small batches.

    In plain words: clear the old gradients. For each (x, y) in batches, work out the loss,
    divide it by how many batches there are, and call .backward(), which adds to the
    gradients. Then clip the gradients and take one optimizer step. Return the average loss.
    See lesson section 4 (the step) and section 7 (accumulation).
    """
    # 1. optimizer.zero_grad() to clear the gradients from the last step.
    # 2. Start a running total at 0.0.
    # 3. For each x, y in batches:
    # 4.   loss = causal_lm_loss(model(x), y) / len(batches)   (exercise 2)
    # 5.   loss.backward(), then add loss.item() to the total.
    # 6. clip_grad_norm(list(model.parameters()), max_norm)   (exercise 4)
    # 7. optimizer.step(), then return the total.
    raise NotImplementedError("Exercise 5: train_step")


def estimate_loss(model, data, block_size, batch_size, iters, generator=None):
    """Exercise 6. Measure the loss honestly, without training.

    In plain words: switch the model to evaluation mode (dropout off), and without
    computing any gradients, average the loss over `iters` random batches from data.
    Then switch back to training mode. See lesson section 8.
    """
    # 1. model.eval()
    # 2. Start a running total at 0.0.
    # 3. Inside `with torch.no_grad():`, repeat iters times:
    # 4.   x, y = get_batch(data, block_size, batch_size, generator)   (exercise 1)
    # 5.   Add causal_lm_loss(model(x), y).item() to the total.
    # 6. model.train(), then return total / iters.
    raise NotImplementedError("Exercise 6: estimate_loss")
