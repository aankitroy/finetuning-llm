"""M3 reference solutions. Use `make solution M=03 EX=n` to see one at a time."""


def pair_counts(ids: list[int]) -> dict[tuple[int, int], int]:
    counts: dict[tuple[int, int], int] = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    out, i = [], 0
    while i < len(ids):
        if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def train_bpe(text: str, num_merges: int) -> dict[tuple[int, int], int]:
    ids = list(text.encode("utf-8"))
    merges: dict[tuple[int, int], int] = {}
    for new_id in range(256, 256 + num_merges):
        counts = pair_counts(ids)
        if not counts:
            break
        pair = max(counts, key=counts.get)
        ids = merge(ids, pair, new_id)
        merges[pair] = new_id
    return merges


def encode(text: str, merges: dict[tuple[int, int], int]) -> list[int]:
    ids = list(text.encode("utf-8"))
    while True:
        counts = pair_counts(ids)
        if not counts:
            break
        pair = min(counts, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break
        ids = merge(ids, pair, merges[pair])
    return ids


def decode(ids: list[int], merges: dict[tuple[int, int], int]) -> str:
    vocab = {i: bytes([i]) for i in range(256)}
    for (a, b), new_id in merges.items():
        vocab[new_id] = vocab[a] + vocab[b]
    return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")


def render_chatml(
    messages: list[dict[str, str]],
    add_generation_prompt: bool = False,
    default_system: str | None = None,
) -> str:
    pieces = []
    if default_system is not None and (not messages or messages[0]["role"] != "system"):
        pieces.append(f"<|im_start|>system\n{default_system}<|im_end|>\n")
    for m in messages:
        pieces.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
    if add_generation_prompt:
        pieces.append("<|im_start|>assistant\n")
    return "".join(pieces)
