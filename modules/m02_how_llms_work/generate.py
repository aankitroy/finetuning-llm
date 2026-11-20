"""Run your M2 generation loop against model.generate() and save a run summary.

    uv run python modules/m02_how_llms_work/generate.py           # real model (SmolLM2-135M)
    uv run python modules/m02_how_llms_work/generate.py --tiny    # tiny random model, no download

Uses your code in exercises.py. Writes outputs/m02/run_summary.json, which
`make check M=02` reads. The first real run downloads the model (about 270 MB) and
caches it for next time.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

MODULE_DIR = Path(__file__).resolve().parent
ROOT = MODULE_DIR.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(MODULE_DIR))

from common.checkpoint import load_exercise  # noqa: E402

MODEL = "HuggingFaceTB/SmolLM2-135M"


def load(tiny: bool):
    """(model, encode, decode, eos_token_id, name). CPU and float32, so the comparison is exact."""
    if tiny:
        from tiny_model import make_tiny_model

        model = make_tiny_model()
        return (
            model,
            lambda text: [1, 5, 9, 22, 7],
            lambda ids: " ".join(map(str, ids)),
            None,
            "tiny",
        )
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    return model, tokenizer.encode, tokenizer.decode, tokenizer.eos_token_id, MODEL


def rate(tokens: int, seconds: float) -> float:
    return round(tokens / seconds, 1) if seconds > 0 else 0.0


def timed(fn):
    start = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - start


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tiny", action="store_true", help="tiny random model, no download")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--tokens", type=int, default=20, help="how many new tokens to compare")
    args = ap.parse_args()

    ex = load_exercise(MODULE_DIR)
    model, encode, decode, eos, name = load(args.tiny)
    input_ids = torch.tensor([encode(args.prompt)])
    print(f"Model: {name}\nPrompt: {args.prompt!r}\n")

    yours, your_s = timed(lambda: ex.generate(model, input_ids, args.tokens, eos_token_id=eos))
    with torch.no_grad():
        library, lib_s = timed(
            lambda: model.generate(
                input_ids,
                max_new_tokens=args.tokens,
                do_sample=False,
                eos_token_id=eos,
                pad_token_id=eos if eos is not None else 0,
            )[0, input_ids.shape[1] :].tolist()
        )
    matched = yours == library
    print(f"Your loop:        {decode(yours)!r}")
    print(f"model.generate(): {decode(library)!r}")
    print("Match: yes, token for token." if matched else "Match: NO. See the two outputs above.")

    print(
        f"\nSpeed: your loop {rate(len(yours), your_s)} tokens/s, "
        f"generate() {rate(len(library), lib_s)} tokens/s (it uses a KV cache, section 7)"
    )

    summary = {
        "matched": matched,
        "model": name,
        "prompt": args.prompt,
        "your_text": decode(yours),
        "library_text": decode(library),
        "tokens_compared": len(library),
        "your_tokens_per_sec": rate(len(yours), your_s),
        "library_tokens_per_sec": rate(len(library), lib_s),
        "used_solutions": bool(os.environ.get("FTCOURSE_SOLUTIONS")),
    }
    out = Path(os.environ.get("FTCOURSE_OUTPUTS") or ROOT / "outputs") / "m02"
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nSaved {out / 'run_summary.json'}. Next: make check M=02")
    try:
        sampled = ex.generate(
            model,
            input_ids,
            args.tokens,
            temperature=0.8,
            top_p=0.9,
            eos_token_id=eos,
            generator=torch.Generator().manual_seed(0),
        )
        print(f"\nFor fun, sampled at temperature 0.8, top_p 0.9: {decode(sampled)!r}")
    except NotImplementedError as e:
        print(f"\n(Sampling demo skipped: {e}. Write exercises 3 and 4 to see it.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
