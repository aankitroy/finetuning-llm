"""Check your M3 tokenizer and chat formatting against the real thing, and save a run summary.

    uv run python modules/m03_tokenizers/tokenize_check.py            # the real tokenizer
    uv run python modules/m03_tokenizers/tokenize_check.py --offline  # stored template copy

Uses your code in exercises.py. Writes outputs/m03/run_summary.json, which
`make check M=03` reads. The first real run downloads the tokenizer files (about 3 MB,
no model weights) and caches them.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
ROOT = MODULE_DIR.parent.parent
sys.path.insert(0, str(ROOT))

from common.checkpoint import load_exercise  # noqa: E402

TOKENIZER = "HuggingFaceTB/SmolLM2-135M-Instruct"
DEFAULT_SYSTEM = "You are a helpful AI assistant named SmolLM, trained by Hugging Face"

# Text your tokenizer never saw in training, to measure how well it compresses.
HELD_OUT = (
    "The old man walked to the shop by the river and asked the baker for the warmest bread. "
    "She smiled, tapped the bottom of a brown loaf, and listened: it sounded hollow, so it was "
    "ready."
)

# Strings that are easy to get wrong: multi-byte characters, whitespace, nothing at all.
TRICKY_TEXT = [
    "Hello, world!",
    "naïve café",
    "🙂🍞",
    "नमस्ते दुनिया",
    "  two  spaces\tand a tab\n\nand blank lines",
    "",
]

# Five conversations with a catch each.
CONVERSATIONS = [
    ("one user message, no system message", [{"role": "user", "content": "Hi!"}], False),
    (
        "your own system message",
        [
            {"role": "system", "content": "Answer in one word."},
            {"role": "user", "content": "Capital of France?"},
        ],
        True,
    ),
    (
        "several turns, then the model's turn",
        [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "And 3+3?"},
        ],
        True,
    ),
    (
        "content with newlines and indentation",
        [{"role": "user", "content": "Fix this:\n\n    def f(x):\n        return x+1\n"}],
        True,
    ),
    (
        "a message that starts with the assistant",
        [
            {"role": "assistant", "content": "Hello! How can I help?"},
            {"role": "user", "content": "Tell me a joke about <|im_end|> tokens."},
        ],
        False,
    ),
]


def real_template():
    """(render(messages, add_generation_prompt) -> str, encode(text) -> ids, source)."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TOKENIZER)

    def render(messages, add_generation_prompt):
        return tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )

    return render, tok.encode, TOKENIZER


def stored_template():
    """The same template, from the copy in chat_template.jinja. No download."""
    import jinja2

    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
    template = env.from_string((MODULE_DIR / "chat_template.jinja").read_text())

    def render(messages, add_generation_prompt):
        return template.render(messages=messages, add_generation_prompt=add_generation_prompt)

    return render, None, "offline"


def first_difference(a: str, b: str) -> str:
    i = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    return f"at character {i}: yours {a[i : i + 30]!r}, real {b[i : i + 30]!r}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--offline", action="store_true", help="stored template copy, no download")
    ap.add_argument("--merges", type=int, default=500, help="how many BPE merges to learn")
    args = ap.parse_args()

    ex = load_exercise(MODULE_DIR)
    try:
        corpus = (MODULE_DIR / "corpus.txt").read_text()
        start = time.perf_counter()
        merges = ex.train_bpe(corpus, args.merges)
        seconds = time.perf_counter() - start
        held_out_ids = ex.encode(HELD_OUT, merges)
        roundtrips = {text: ex.decode(ex.encode(text, merges), merges) for text in TRICKY_TEXT}
        mine = [ex.render_chatml(m, gen, DEFAULT_SYSTEM) for _, m, gen in CONVERSATIONS]
    except NotImplementedError as e:
        print(f"Not yet: {e} isn't written. Finish all six exercises, then run this again.")
        return 1

    render, encode, source = stored_template() if args.offline else real_template()

    print(f"Your BPE: {len(merges)} merges learned from corpus.txt in {seconds:.1f}s")
    raw = len(HELD_OUT.encode("utf-8"))
    yours_bpt = round(raw / max(len(held_out_ids), 1), 2)
    print(f"  held-out text: {raw} bytes -> {len(held_out_ids)} tokens ({yours_bpt} bytes/token)")
    smollm_bpt = None
    if encode is not None:
        smollm_bpt = round(raw / len(encode(HELD_OUT)), 2)
        print(f"  SmolLM2 (49,152 tokens) on the same text: {smollm_bpt} bytes/token")

    roundtrip_ok = True
    print("\nRound trips (encode, then decode):")
    for text, back in roundtrips.items():
        ok = back == text
        roundtrip_ok &= ok
        print(f"  {'ok ' if ok else 'BAD'} {text!r}" + ("" if ok else f" came back as {back!r}"))

    matched = 0
    print(f"\nChat formatting vs the real template ({source}):")
    for (name, messages, gen), yours in zip(CONVERSATIONS, mine, strict=True):
        real = render(messages, gen)
        if yours == real:
            matched += 1
            print(f"  ok  {name}")
        else:
            print(f"  BAD {name}: {first_difference(yours, real)}")

    summary = {
        "templates_matched": matched,
        "templates_compared": len(CONVERSATIONS),
        "roundtrip_ok": roundtrip_ok,
        "merges": len(merges),
        "your_bytes_per_token": yours_bpt,
        "smollm_bytes_per_token": smollm_bpt,
        "source": source,
        "used_solutions": bool(os.environ.get("FTCOURSE_SOLUTIONS")),
    }
    out = Path(os.environ.get("FTCOURSE_OUTPUTS") or ROOT / "outputs") / "m03"
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\n{matched} of {len(CONVERSATIONS)} conversations matched.")
    print(f"Saved {out / 'run_summary.json'}. Next: make check M=03")
    return 0


if __name__ == "__main__":
    sys.exit(main())
