"""M2 exercises. Replace each `raise NotImplementedError` with your code.

Each exercise says in plain words what it is, then lists numbered steps as comments.
Write one or two lines of code under each step, then delete the `raise` line.
The lesson section named in each exercise shows every piece you need.

Check your work any time with:  make check M=02
Stuck?  make hint M=02 EX=1  (a nudge, logged on your dashboard)
"""

import torch
from torch.nn import functional as F  # noqa: F401  (you'll want F.softmax)


@torch.no_grad()
def next_token_logits(model, input_ids: torch.Tensor) -> torch.Tensor:
    """Exercise 1. The model's scores for the next token.

    In plain words: run the model on the tokens so far and keep only the scores for
    what comes *after the last token*. See lesson section 4.

    input_ids has shape (1, seq): one prompt, seq tokens. Return shape (vocab,).

    The `@torch.no_grad()` line above the function switches off gradient recording for the
    whole function, like a `with torch.no_grad():` block around its body (M1 section 2).
    """
    # 1. Run the model on input_ids and take the .logits from what it returns.
    #    Their shape is (1, seq, vocab): a score for every vocabulary token at every position.
    # 2. Keep the first (only) prompt and its last position, and return that.
    raise NotImplementedError("Exercise 1: next_token_logits")


def greedy(logits: torch.Tensor) -> int:
    """Exercise 2. Greedy decoding: always take the single most likely token.

    In plain words: the next token is whichever has the biggest score. See lesson
    section 4. Return it as a plain Python int.
    """
    # 1. Find the position of the biggest logit (the same tool you used in M1).
    # 2. Turn it into a plain Python int and return it.
    raise NotImplementedError("Exercise 2: greedy")


def softmax_with_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Exercise 3. Turn scores into probabilities, sharper or flatter.

    In plain words: dividing the logits by a temperature below 1 makes the top token
    even more likely; above 1 spreads the chances out. See lesson section 6.
    """
    # 1. Divide the logits by the temperature.
    # 2. Apply softmax so they become probabilities that add up to 1, and return them.
    raise NotImplementedError("Exercise 3: softmax_with_temperature")


def top_p_filter(probs: torch.Tensor, p: float) -> torch.Tensor:
    """Exercise 4. Top-p (nucleus) filtering.

    In plain words: keep only the most likely tokens, just enough of them that their
    probabilities add up to at least p, and throw the long tail of unlikely tokens
    away. See lesson section 6.

    Return a tensor the same shape as probs: kept tokens renormalised to sum to 1,
    every other token 0. Always keep at least the most likely token.
    """
    # 1. Sort the probabilities from biggest to smallest, remembering where each came from
    #    (torch.sort gives you both the sorted values and their original positions).
    # 2. Make a running total of the sorted probabilities.
    # 3. Decide which sorted tokens to keep: a token is kept if the total of the tokens
    #    *before* it is still less than p. (So the token that crosses p is kept too.)
    # 4. Map that keep/drop decision back to the original positions.
    # 5. Set the dropped tokens to 0.
    # 6. Divide by the new total so the kept probabilities add up to 1, and return.
    raise NotImplementedError("Exercise 4: top_p_filter")


def causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Exercise 5. One attention head, by hand, with a causal mask.

    In plain words: each token asks a question (its query), compares it with every
    token's label (its key), and takes a weighted mix of their information (values),
    looking only at itself and earlier tokens. See lesson section 3.

    q, k, v each have shape (seq, d). Return shape (seq, d).
    """
    # 1. Scores: every query against every key (a matrix multiply with k turned on its side),
    #    divided by the square root of d. Shape (seq, seq).
    # 2. Build the causal mask: True for every position above the diagonal (the future).
    # 3. Put -inf in the scores wherever the mask is True.
    # 4. Softmax each row, so each token's attention weights add up to 1.
    # 5. Multiply the weights by v and return the result.
    raise NotImplementedError("Exercise 5: causal_attention")


def generate(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 0.0,
    top_p: float = 1.0,
    eos_token_id: int | None = None,
    generator: torch.Generator | None = None,
) -> list[int]:
    """Exercise 6. The generation loop: how a chatbot writes, one token at a time.

    In plain words: predict the next token, stick it on the end, and repeat. With
    temperature 0, always take the most likely token (greedy); otherwise sample from
    the top-p probabilities. Stop early at the end-of-text token. Lesson section 5 shows
    the greedy loop; section 6 shows sampling.

    Return only the new tokens, as a list of plain ints.
    """
    # 1. Make an empty list for the new tokens.
    # 2. Repeat max_new_tokens times:
    # 3.   Get the next-token logits (exercise 1).
    # 4.   If temperature is 0: pick greedily (exercise 2).
    # 5.   Otherwise: temperature softmax (exercise 3), then top-p (exercise 4), then draw
    #      one token with torch.multinomial, passing generator=generator.
    # 6.   Add the token to your list.
    # 7.   Append it to input_ids as a (1, 1) tensor on the same device (torch.cat, dim=1).
    # 8.   If the token is eos_token_id, stop.
    # 9. Return the list.
    raise NotImplementedError("Exercise 6: generate")
