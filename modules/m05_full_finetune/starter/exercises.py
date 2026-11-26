"""M5 exercises. Replace each `raise NotImplementedError` with your code.

Each exercise says in plain words what it is, then lists numbered steps as comments.
Write one or two lines of code under each step, then delete the `raise` line.
The lesson section named in each exercise shows every piece you need.

Exercises 1 to 3 turn example pairs into training batches and grade only the answer.
Exercise 4 measures what full fine-tuning costs in memory. Exercises 5 and 6 score the
rewrites, so you can tell whether fine-tuning worked.

Check your work any time with:  make check M=05
Stuck?  make hint M=05 EX=1  (a nudge, logged on your dashboard)
"""

import torch  # noqa: F401  (you'll want torch.tensor and torch.is_tensor)
import torch.nn.functional as F  # noqa: F401  (you'll want F.cross_entropy)


def build_example(prompt_ids, answer_ids, max_len=None):
    """Exercise 1. One training example, with the prompt masked out of the loss.

    In plain words: the model reads the prompt and the answer as one sequence. But we only
    want to grade it on the answer, so the labels say -100 ("ignore this") for every prompt
    token and copy the answer tokens. build_example([5, 6, 7], [8, 9]) gives
    ([5, 6, 7, 8, 9], [-100, -100, -100, 8, 9]). See lesson section 3.
    """
    # 1. input_ids = prompt_ids + answer_ids   (+ joins two lists)
    # 2. labels = [-100] * len(prompt_ids) + answer_ids
    # 3. If max_len is not None: keep only the first max_len items of both ([:max_len]).
    # 4. Return input_ids, labels.
    raise NotImplementedError("Exercise 1: build_example")


def collate(examples, pad_id):
    """Exercise 2. Turn a list of examples of different lengths into one batch.

    In plain words: examples is a list of (input_ids, labels) pairs from exercise 1. Pad
    every row on the right up to the longest one: input_ids with pad_id, labels with -100
    (so padding is never graded). The attention mask is 1 for real tokens and 0 for padding.
    Return a dict of three long tensors. See lesson section 2.
    """
    # 1. longest = the length of the longest input_ids in examples (max(...)).
    # 2. Make three empty lists: input_ids, attention_mask, labels.
    # 3. For each ids, labs in examples:
    # 4.   pad = longest - len(ids)
    # 5.   Append ids + [pad_id] * pad, then [1] * len(ids) + [0] * pad,
    #      then labs + [-100] * pad, to the three lists.
    # 6. Return {"input_ids": ..., "attention_mask": ..., "labels": ...}, each made with
    #    torch.tensor(the_list, dtype=torch.long).
    raise NotImplementedError("Exercise 2: collate")


def masked_lm_loss(logits, labels):
    """Exercise 3. The loss on the answer tokens only, each guessed from the one before.

    In plain words: the logits at position t are the model's guess for the token at t + 1.
    So drop the last position of the logits and the first position of the labels to line
    them up, then use cross-entropy that skips every label of -100. It should give the same
    number as model(input_ids, labels=labels).loss. See lesson section 3.
    """
    # 1. shifted_logits = logits[:, :-1, :]   (every position but the last)
    # 2. shifted_labels = labels[:, 1:]       (every position but the first)
    # 3. vocab = shifted_logits.shape[-1]
    # 4. Return F.cross_entropy(shifted_logits.reshape(-1, vocab), shifted_labels.reshape(-1),
    #    ignore_index=-100).
    raise NotImplementedError("Exercise 3: masked_lm_loss")


def optimizer_state_bytes(optimizer):
    """Exercise 4. How much memory the optimizer is using for its own notes.

    In plain words: AdamW remembers two running averages for every weight. They live in
    optimizer.state, a dict with one entry per weight; each entry is a dict of tensors (and
    sometimes plain numbers). Add up the bytes of every tensor. See lesson section 4.
    """
    # 1. Start a total at 0.
    # 2. For each state in optimizer.state.values():
    # 3.   For each value in state.values():
    # 4.     If torch.is_tensor(value): add value.numel() * value.element_size() to the total.
    # 5. Return the total.
    raise NotImplementedError("Exercise 4: optimizer_state_bytes")


def chrf(hypothesis, reference, max_n=6, beta=2.0):
    """Exercise 5. How close a rewrite is to the reference, from 0 to 100.

    In plain words: remove the spaces, then compare the two texts' pieces of 1 character,
    2 characters, ... up to max_n characters. Precision: what share of the rewrite's pieces
    also appear in the reference. Recall: what share of the reference's pieces the rewrite
    has. Average each over the piece sizes, then combine them into one F-score that cares
    about recall more (beta = 2). See lesson section 7.
    """
    # 1. hyp = hypothesis with spaces removed (.replace(" ", "")), ref = the same for reference.
    # 2. Make two empty lists: precisions and recalls.
    # 3. For n from 1 to max_n (range(1, max_n + 1)):
    # 4.   Count every piece of n characters in hyp into a dict hyp_counts (hyp[i : i + n] for
    #      i in range(len(hyp) - n + 1)), and the same for ref into ref_counts.
    # 5.   If len(hyp_counts) == 0 or len(ref_counts) == 0: skip this n (continue).
    # 6.   matches = the sum, over each piece g and count c in hyp_counts, of
    #      min(c, ref_counts.get(g, 0)).
    # 7.   Append matches / (total count in hyp_counts) to precisions, and
    #      matches / (total count in ref_counts) to recalls.
    # 8. If precisions is empty, return 0.0.
    # 9. p and r = the averages of precisions and recalls. If p + r == 0, return 0.0.
    # 10. Return 100 * (1 + beta**2) * p * r / (beta**2 * p + r).
    raise NotImplementedError("Exercise 5: chrf")


def count_buzzwords(text, buzzwords):
    """Exercise 6. How many buzzwords are left in a rewrite.

    In plain words: lowercase the text and turn everything that isn't a letter into a space,
    then split it into words. Do the same to each buzzword (some are several words, like
    "circle back"). Count every place where the buzzword's words appear in a row.
    See lesson section 7.
    """
    # 1. Write a small helper inside this function (or above it) that turns text into a list
    #    of lowercase words: "".join(c if c.isalpha() else " " for c in text.lower()).split()
    # 2. words = the helper applied to text. Start a total at 0.
    # 3. For each buzzword in buzzwords:
    # 4.   phrase = the helper applied to buzzword, and n = len(phrase).
    # 5.   For each i in range(len(words) - n + 1): if words[i : i + n] == phrase, add 1.
    # 6. Return the total.
    raise NotImplementedError("Exercise 6: count_buzzwords")
