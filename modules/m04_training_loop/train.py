"""Train the tiny GPT on Shakespeare with your M4 training loop, and save a run summary.

    uv run python modules/m04_training_loop/train.py                    # the real run, ~2 minutes
    uv run python modules/m04_training_loop/train.py --tiny             # a small model, seconds
    uv run python modules/m04_training_loop/train.py --precision bf16   # the stretch

Uses your code in exercises.py. Writes outputs/m04/run_summary.json (which
`make check M=04` reads), loss_curve.png and samples.txt.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import resource
import sys
import time
from pathlib import Path

import torch

MODULE_DIR = Path(__file__).resolve().parent
ROOT = MODULE_DIR.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(MODULE_DIR))

from model import TinyGPT, count_parameters  # noqa: E402

from common.checkpoint import load_exercise  # noqa: E402

FULL = dict(block_size=256, n_embd=256, n_head=4, n_layer=6, dropout=0.1)
TINY = dict(block_size=32, n_embd=64, n_head=2, n_layer=2, dropout=0.0)
TARGET_VAL_LOSS = 1.70  # the checkpoint's targets (tests/test_checkpoint.py)
TARGET_REAL_WORDS = 0.80


def pick_device(tiny: bool) -> str:
    return "mps" if torch.backends.mps.is_available() and not tiny else "cpu"


def memory_gb(device: str) -> float:
    """Memory in use right now: the GPU's on a Mac, the process's peak otherwise."""
    if device == "mps":
        return torch.mps.driver_allocated_memory() / 1e9
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1e9 if sys.platform == "darwin" else peak / 1e6  # bytes on macOS, KB on Linux


def real_word_rate(sample: str, known: set[str]) -> float:
    """The share of words in the sample that also appear somewhere in the training text."""
    words = re.findall(r"[a-z']+", sample.lower())
    return round(sum(w in known for w in words) / len(words), 3) if words else 0.0


def save_curve(history: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [h["step"] for h in history]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, [h["train_loss"] for h in history], label="train")
    ax.plot(steps, [h["val_loss"] for h in history], label="validation")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title("Your tiny GPT learning Shakespeare")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tiny", action="store_true", help="small model on the CPU, seconds")
    ap.add_argument("--steps", type=int, default=None, help="training steps (default 2000)")
    ap.add_argument("--batch", type=int, default=16, help="rows in each micro-batch")
    ap.add_argument("--accum", type=int, default=2, help="micro-batches per step")
    ap.add_argument("--lr", type=float, default=1e-3, help="the peak learning rate")
    ap.add_argument("--min-lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=100, help="warmup steps")
    ap.add_argument("--clip", type=float, default=1.0, help="max gradient norm")
    ap.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    steps = args.steps or (150 if args.tiny else 2000)
    warmup = min(args.warmup, max(1, steps // 10))
    eval_every, eval_iters = (50, 5) if args.tiny else (200, 20)

    ex = load_exercise(MODULE_DIR)
    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    device = pick_device(args.tiny)

    text = (MODULE_DIR / "shakespeare.txt").read_text()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long, device=device)
    split = int(0.9 * len(data))
    train_data, val_data = data[:split], data[split:]
    known_words = set(re.findall(r"[a-z']+", text[:split].lower()))

    config = TINY if args.tiny else FULL
    model = TinyGPT(vocab_size=len(chars), **config).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    block = config["block_size"]
    autocast = (
        torch.autocast(device_type=device, dtype=torch.bfloat16)
        if args.precision == "bf16"
        else contextlib.nullcontext()
    )

    print(f"{count_parameters(model) / 1e6:.2f}M parameters, {len(chars)} characters, on {device}")
    print(f"{steps} steps of {args.accum} x {args.batch} windows of {block} characters\n")
    history = []
    peak = memory_gb(device)
    start = time.perf_counter()
    try:
        initial = ex.estimate_loss(model, val_data, block, args.batch, eval_iters, gen)
        print(f"  step     0  val loss {initial:.3f}  (before any training)")
        for step in range(steps):
            lr = ex.lr_at(step, args.lr, args.min_lr, warmup, steps)
            for group in opt.param_groups:
                group["lr"] = lr
            batches = [ex.get_batch(train_data, block, args.batch, gen) for _ in range(args.accum)]
            with autocast:
                ex.train_step(model, batches, opt, args.clip)
            if (step + 1) % eval_every == 0 or step + 1 == steps:
                with autocast:
                    tr = ex.estimate_loss(model, train_data, block, args.batch, eval_iters, gen)
                    va = ex.estimate_loss(model, val_data, block, args.batch, eval_iters, gen)
                peak = max(peak, memory_gb(device))
                history.append({"step": step + 1, "lr": lr, "train_loss": tr, "val_loss": va})
                took = time.perf_counter() - start
                print(
                    f"  step {step + 1:5d}  train loss {tr:.3f}  val loss {va:.3f}  "
                    f"lr {lr:.1e}  {took:5.0f}s"
                )
        final_val = ex.estimate_loss(model, val_data, block, args.batch, eval_iters * 5, gen)
    except NotImplementedError as e:
        print(f"Not yet: {e} isn't written. Finish all six exercises, then run this again.")
        return 1
    except Exception as e:  # a bug in an exercise: say so plainly, not with a traceback
        print(f"One of your exercises raised {type(e).__name__}: {e}")
        print("Run make check M=04 to see which one, and why.")
        return 1
    seconds = time.perf_counter() - start

    prompt = torch.tensor([[stoi["\n"]]], device=device)
    ids = model.generate(prompt, 600 if not args.tiny else 100, generator=gen)[0].tolist()
    sample = "".join(chars[i] for i in ids)
    words = real_word_rate(sample, known_words)

    out = Path(os.environ.get("FTCOURSE_OUTPUTS") or ROOT / "outputs") / "m04"
    out.mkdir(parents=True, exist_ok=True)
    (out / "samples.txt").write_text(sample + "\n")
    save_curve(history, out / "loss_curve.png")
    summary = {
        "val_loss": round(final_val, 4),
        "train_loss": round(history[-1]["train_loss"], 4),
        "initial_val_loss": round(initial, 4),
        "real_word_rate": words,
        "peak_mem_gb": round(peak, 2),
        "steps": steps,
        "params": count_parameters(model),
        "precision": args.precision,
        "device": device,
        "seconds": round(seconds, 1),
        "tiny": args.tiny,
        "used_solutions": bool(os.environ.get("FTCOURSE_SOLUTIONS")),
        "history": history,
    }
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(f"\nA sample from your model:\n{'-' * 40}\n{sample.strip()}\n{'-' * 40}")
    if args.tiny:
        print(
            f"Validation loss {final_val:.3f}, real words {words:.0%} (a --tiny run has no target)"
        )
    else:
        print(f"Validation loss {final_val:.3f} (need below {TARGET_VAL_LOSS})")
        print(f"Real words in the sample: {words:.0%} (need {TARGET_REAL_WORDS:.0%})")
    print(f"Peak memory {peak:.2f} GB, {seconds:.0f}s")
    print(f"Saved {out}/run_summary.json, loss_curve.png and samples.txt. Next: make check M=04")
    return 0


if __name__ == "__main__":
    sys.exit(main())
