"""M3 exercises. Replace each `raise NotImplementedError` with your code.

Each exercise says in plain words what it is, then lists numbered steps as comments.
Write one or two lines of code under each step, then delete the `raise` line.
The lesson section named in each exercise shows every piece you need.

Exercises 1 to 5 build a byte-level BPE tokenizer, the same kind real models use.
Exercise 6 formats a chat the way SmolLM2-Instruct expects.

Check your work any time with:  make check M=03
Stuck?  make hint M=03 EX=1  (a nudge, logged on your dashboard)
"""


def pair_counts(ids: list[int]) -> dict[tuple[int, int], int]:
    """Exercise 1. Count how often each pair of neighbouring ids appears.

    In plain words: look at every two ids that sit next to each other and count each
    pair. For [1, 2, 3, 1, 2] that's (1, 2) twice, (2, 3) once and (3, 1) once.
    See lesson section 2.

    Keep the pairs in the order you first meet them (a plain dict does that for you).
    """
    # 1. Make an empty dict for the counts.
    # 2. Walk over each neighbouring pair: zip(ids, ids[1:]) gives them to you.
    # 3.   Add 1 to that pair's count (counts.get(pair, 0) + 1).
    # 4. Return the dict.
    raise NotImplementedError("Exercise 1: pair_counts")


def merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Exercise 2. Replace every occurrence of one pair with a single new id.

    In plain words: walk along the ids; wherever `pair` appears, write `new_id`
    instead of the two ids and jump past both. merge([1, 2, 3, 1, 2], (1, 2), 99)
    gives [99, 3, 99]. See lesson section 2.
    """
    # 1. Make an empty list for the result, and start a position counter i at 0.
    # 2. While i is still inside ids:
    # 3.   If there's a next id and (ids[i], ids[i + 1]) equals pair: add new_id, move i on by 2.
    # 4.   Otherwise: add ids[i] unchanged, move i on by 1.
    # 5. Return the result.
    raise NotImplementedError("Exercise 2: merge")


def train_bpe(text: str, num_merges: int) -> dict[tuple[int, int], int]:
    """Exercise 3. Learn BPE merges from some text.

    In plain words: start from the text's UTF-8 bytes (ids 0 to 255). Find the most
    common pair, give it a new id (256, then 257, ...), merge it everywhere, and repeat
    num_merges times. The list of merges you learned *is* the tokenizer. See lesson
    section 2.

    Return {pair: new_id}, in the order you learned them.
    """
    # 1. Turn the text into a list of byte values: list(text.encode("utf-8")).
    # 2. Make an empty dict for the merges.
    # 3. Repeat num_merges times, with the new id going 256, 257, ...:
    # 4.   Count the pairs (exercise 1). If there are none, stop.
    # 5.   Pick the most frequent pair: max(counts, key=counts.get). On a tie, max keeps the
    #      one it met first, which is the rule this course uses.
    # 6.   Merge that pair everywhere (exercise 2) and remember merges[pair] = new_id.
    # 7. Return the merges.
    raise NotImplementedError("Exercise 3: train_bpe")


def encode(text: str, merges: dict[tuple[int, int], int]) -> list[int]:
    """Exercise 4. Turn text into token ids using the merges you learned.

    In plain words: start from the UTF-8 bytes and keep applying merges until none
    apply. Always apply the merge that was learned *first* (lowest new id) among the
    pairs present, because that's the order training built them in. See lesson section 3.
    """
    # 1. Turn the text into a list of byte values.
    # 2. Loop:
    # 3.   Count the pairs (exercise 1). If there are none, stop.
    # 4.   Of those pairs, pick the one with the lowest merge id:
    #      min(counts, key=lambda p: merges.get(p, float("inf"))).
    # 5.   If that pair isn't in merges at all, nothing more can merge: stop.
    # 6.   Merge it (exercise 2), using merges[pair] as the new id.
    # 7. Return the ids.
    raise NotImplementedError("Exercise 4: encode")


def decode(ids: list[int], merges: dict[tuple[int, int], int]) -> str:
    """Exercise 5. Turn token ids back into text.

    In plain words: every id stands for some bytes. Ids 0 to 255 are one byte each;
    each new id is the bytes of its two parts joined together. Join all the bytes and
    read them as UTF-8. See lesson section 3.

    Some id sequences cut a character in half (an emoji is 4 bytes). Don't crash on those:
    show the replacement character � instead.
    """
    # 1. Build the vocabulary: a dict from each id 0 to 255 to bytes([id]).
    # 2. Go through merges in order: vocab[new_id] = vocab[a] + vocab[b] for each pair (a, b).
    # 3. Join the bytes of every id in ids: b"".join(...).
    # 4. Decode them with .decode("utf-8", errors="replace") and return the text.
    raise NotImplementedError("Exercise 5: decode")


def render_chatml(
    messages: list[dict[str, str]],
    add_generation_prompt: bool = False,
    default_system: str | None = None,
) -> str:
    """Exercise 6. Format a conversation the way a ChatML model expects.

    In plain words: a chat model doesn't see a list of messages, it sees one long string.
    Each message becomes

        <|im_start|>ROLE
        CONTENT<|im_end|>

    followed by a newline. If default_system is given and the conversation doesn't
    start with a system message, a system message with that text goes first. With
    add_generation_prompt, the string ends with the start of an assistant turn, so the
    model knows it's its turn to speak. See lesson section 6.

    messages look like [{"role": "user", "content": "Hi"}].
    """
    # 1. Make a list for the pieces of text.
    # 2. If default_system is given and messages is empty or messages[0]["role"] isn't
    #    "system": add the system message piece first.
    # 3. For each message, add f"<|im_start|>{role}\n{content}<|im_end|>\n".
    # 4. If add_generation_prompt: add "<|im_start|>assistant\n".
    # 5. Join the pieces into one string ("".join) and return it.
    raise NotImplementedError("Exercise 6: render_chatml")
