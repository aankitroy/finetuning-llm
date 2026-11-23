"""M3 checkpoint: your tokenizer round-trips text and your chat formatting matches the real one.

Run `uv run python modules/m03_tokenizers/tokenize_check.py` first, then `make check M=03`.
"""

import json
import os

import pytest
from conftest import MODULE_DIR

from common.checkpoint import check_answer
from common.progress import report_metric

# "aaabdaaabac" is the classic BPE example: a=97, b=98, c=99, d=100 as bytes.
TEXT = "aaabdaaabac"
MERGES = {(97, 97): 256, (256, 97): 257, (257, 98): 258}


def test_pair_counts(ex):
    counts = ex.pair_counts([1, 2, 3, 1, 2])
    assert counts == {(1, 2): 2, (2, 3): 1, (3, 1): 1}, (
        f"Count every neighbouring pair: (ids[0], ids[1]), (ids[1], ids[2]), ... Got {counts}"
    )
    assert list(counts) == [(1, 2), (2, 3), (3, 1)], (
        "Keep pairs in the order you first meet them (a plain dict does this)"
    )
    assert ex.pair_counts([7]) == {} and ex.pair_counts([]) == {}, (
        "Fewer than two ids means no pairs: return an empty dict"
    )


def test_merge(ex):
    assert ex.merge([1, 2, 3, 1, 2], (1, 2), 99) == [99, 3, 99], (
        "Replace every (1, 2) with 99 and keep everything else"
    )
    assert ex.merge([1, 1, 1], (1, 1), 256) == [256, 1], (
        "Pairs don't overlap: after merging a pair, jump past both ids (skip 2, not 1)"
    )
    assert ex.merge([5, 6], (6, 5), 300) == [5, 6], "A pair that isn't there changes nothing"


def test_train_bpe(ex):
    merges = ex.train_bpe(TEXT, 3)
    assert merges == MERGES, (
        "Start from the UTF-8 bytes; each round, merge the most frequent pair (on a tie, the one "
        f"you met first) into the next new id, starting at 256. Expected {MERGES}, got {merges}"
    )
    assert list(merges.values()) == [256, 257, 258], "New ids go up by one each round"
    assert ex.train_bpe("a", 5) == {}, "Stop when there are no pairs left to merge"


def test_encode(ex):
    assert ex.encode(TEXT, MERGES) == [258, 100, 258, 97, 99], (
        "Start from the bytes and keep applying merges until none apply"
    )
    by_rank = {(98, 99): 256, (97, 98): 257}
    got = ex.encode("abc", by_rank)
    assert got == [97, 256], (
        "Apply the merge that was learned first (lowest new id), not simply the leftmost pair. "
        f"Expected [97, 256], got {got}"
    )
    assert ex.encode("", MERGES) == [], "Empty text has no tokens"
    assert ex.encode("é", {}) == [195, 169], "With no merges, encoding is just the UTF-8 bytes"


def test_decode(ex):
    assert ex.decode([258, 100, 258, 97, 99], MERGES) == TEXT, (
        "Each new id stands for the bytes of its two parts joined; join all the bytes, then "
        "decode them as UTF-8"
    )
    for text in ["héllo 🙂", "नमस्ते", "  two  spaces\n", ""]:
        merges = ex.train_bpe("héllo 🙂 नमस्ते " * 3, 20)
        assert ex.decode(ex.encode(text, merges), merges) == text, (
            f"decode(encode(text)) must give back the original text. Failed on {text!r}"
        )
    assert ex.decode([240], {}) == "�", (
        "Half an emoji isn't valid UTF-8: use .decode('utf-8', errors='replace') so it "
        "becomes � instead of crashing"
    )


def test_render_chatml(ex):
    hi = [{"role": "user", "content": "Hi"}]
    assert ex.render_chatml(hi) == "<|im_start|>user\nHi<|im_end|>\n", (
        "Each message is <|im_start|>ROLE, a newline, the content, <|im_end|>, a newline"
    )
    assert ex.render_chatml(hi, add_generation_prompt=True) == (
        "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"
    ), (
        "add_generation_prompt adds the start of an assistant turn at the end: "
        "<|im_start|>assistant\\n"
    )
    assert ex.render_chatml(hi, default_system="Be brief.") == (
        "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n"
    ), "When the first message isn't a system message, put the default system message first"
    own = [{"role": "system", "content": "Talk like a pirate."}, *hi]
    assert ex.render_chatml(own, default_system="Be brief.") == (
        "<|im_start|>system\nTalk like a pirate.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n"
    ), "If the conversation already starts with a system message, don't add the default one"
    turns = [
        {"role": "user", "content": "2+2?"},
        {"role": "assistant", "content": "4"},
        {"role": "user", "content": "And 3+3?"},
    ]
    assert ex.render_chatml(turns) == (
        "<|im_start|>user\n2+2?<|im_end|>\n<|im_start|>assistant\n4<|im_end|>\n"
        "<|im_start|>user\nAnd 3+3?<|im_end|>\n"
    ), "Render every message, in order"


# --- your run of tokenize_check.py ---


def _summary(outputs):
    f = outputs / "run_summary.json"
    assert f.exists(), "No run found. Run: uv run python modules/m03_tokenizers/tokenize_check.py"
    return json.loads(f.read_text())


def test_run_matched_the_real_template(outputs):
    s = _summary(outputs)
    report_metric("bytes_per_token", s["your_bytes_per_token"])
    assert s["templates_compared"] >= 5, "Run tokenize_check.py to compare all 5 conversations"
    assert s["templates_matched"] == s["templates_compared"], (
        f"Only {s['templates_matched']} of {s['templates_compared']} conversations matched the "
        "real chat template in your last run. tokenize_check.py prints the first difference."
    )
    assert s["roundtrip_ok"], (
        "Some text didn't come back unchanged after encode then decode in your last run"
    )


def test_run_used_the_real_tokenizer(outputs):
    if os.environ.get("FTCOURSE_SOLUTIONS"):
        pytest.skip("reference-solution run (used by the project's own tests)")
    s = _summary(outputs)
    assert s["source"] == "HuggingFaceTB/SmolLM2-135M-Instruct", (
        "That run used the stored copy of the template. Run tokenize_check.py without --offline."
    )


def test_run_with_your_code(outputs):
    if os.environ.get("FTCOURSE_SOLUTIONS"):
        pytest.skip("reference-solution run (used by the project's own tests)")
    assert not _summary(outputs).get("used_solutions"), (
        "That run used the reference solutions. Re-run tokenize_check.py with your own code."
    )


ANSWER_IDS = sorted(json.loads((MODULE_DIR / "tests" / "answer_hashes.json").read_text()))


@pytest.mark.parametrize("qid", ANSWER_IDS)
def test_understanding(qid):
    ok, message = check_answer(MODULE_DIR, "m03", qid)
    assert ok, message
