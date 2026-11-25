"""M5 reference solutions. Use `make solution M=05 EX=3` instead of reading this file."""

import torch
import torch.nn.functional as F


def build_example(prompt_ids, answer_ids, max_len=None):
    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids
    if max_len is not None:
        input_ids = input_ids[:max_len]
        labels = labels[:max_len]
    return input_ids, labels


def collate(examples, pad_id):
    longest = max(len(ids) for ids, _ in examples)
    input_ids, attention_mask, labels = [], [], []
    for ids, labs in examples:
        pad = longest - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        attention_mask.append([1] * len(ids) + [0] * pad)
        labels.append(labs + [-100] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def masked_lm_loss(logits, labels):
    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    vocab = shifted_logits.shape[-1]
    return F.cross_entropy(
        shifted_logits.reshape(-1, vocab), shifted_labels.reshape(-1), ignore_index=-100
    )


def optimizer_state_bytes(optimizer):
    total = 0
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value):
                total += value.numel() * value.element_size()
    return total


def chrf(hypothesis, reference, max_n=6, beta=2.0):
    hyp = hypothesis.replace(" ", "")
    ref = reference.replace(" ", "")
    precisions, recalls = [], []
    for n in range(1, max_n + 1):
        hyp_counts, ref_counts = {}, {}
        for i in range(len(hyp) - n + 1):
            gram = hyp[i : i + n]
            hyp_counts[gram] = hyp_counts.get(gram, 0) + 1
        for i in range(len(ref) - n + 1):
            gram = ref[i : i + n]
            ref_counts[gram] = ref_counts.get(gram, 0) + 1
        if len(hyp_counts) == 0 or len(ref_counts) == 0:
            continue
        matches = sum(min(c, ref_counts.get(g, 0)) for g, c in hyp_counts.items())
        precisions.append(matches / sum(hyp_counts.values()))
        recalls.append(matches / sum(ref_counts.values()))
    if len(precisions) == 0:
        return 0.0
    p = sum(precisions) / len(precisions)
    r = sum(recalls) / len(recalls)
    if p + r == 0:
        return 0.0
    return 100 * (1 + beta**2) * p * r / (beta**2 * p + r)


def _words(text):
    return "".join(c if c.isalpha() else " " for c in text.lower()).split()


def count_buzzwords(text, buzzwords):
    words = _words(text)
    total = 0
    for buzzword in buzzwords:
        phrase = _words(buzzword)
        n = len(phrase)
        for i in range(len(words) - n + 1):
            if words[i : i + n] == phrase:
                total += 1
    return total
