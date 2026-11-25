"""Fine-tune every weight of Qwen2.5-0.5B-Instruct to rewrite jargon in plain English.

    uv run python modules/m05_full_finetune/train.py --loop      # your own training loop
    uv run python modules/m05_full_finetune/train.py --trainer   # the same, with HF Trainer
    uv run python modules/m05_full_finetune/train.py --loop --tiny   # a tiny model, seconds

Uses your code in exercises.py. Each run scores the original model and the fine-tuned one on
50 held-out rewrites, checks that ordinary questions still get answered, and writes
outputs/m05/run_loop.json or run_trainer.json, which `make check M=05` reads.
The first run downloads the model (about 1 GB) and caches it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import sys
import tempfile
import time
import traceback
from pathlib import Path

import torch

MODULE_DIR = Path(__file__).resolve().parent
ROOT = MODULE_DIR.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(MODULE_DIR))

from common.checkpoint import load_exercise  # noqa: E402

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
SYSTEM = "Rewrite the user's text in plain English. Reply with the rewrite only."
DATA = MODULE_DIR / "data"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def memory_gb(device: str) -> float:
    if device == "mps":
        return torch.mps.driver_allocated_memory() / 1e9
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1e9 if sys.platform == "darwin" else peak / 1e6


class RealText:
    """The real model's tokenizer and chat template."""

    def __init__(self):
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(MODEL)
        self.tok.padding_side = "left"  # for batched generation (M3 section 5)
        self.pad_id = self.tok.pad_token_id
        self.end_id = self.tok.convert_tokens_to_ids("<|im_end|>")

    def prompt_ids(self, text: str, system: str | None = SYSTEM) -> list[int]:
        messages = [{"role": "user", "content": text}]
        if system is not None:
            messages.insert(0, {"role": "system", "content": system})
        return self.tok.apply_chat_template(messages, add_generation_prompt=True)["input_ids"]

    def answer_ids(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False) + [self.end_id]

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=True).strip()


class TinyText:
    """A stand-in for --tiny: one id per character, so no download is needed."""

    pad_id, end_id = 0, 1

    def prompt_ids(self, text: str, system: str | None = SYSTEM) -> list[int]:
        return [2] + [3 + ord(c) % 125 for c in text[:40]] + [2]

    def answer_ids(self, text: str) -> list[int]:
        return [3 + ord(c) % 125 for c in text[:40]] + [self.end_id]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(97 + i % 26) for i in ids if i > 2)


def load_model(tiny: bool, device: str):
    if tiny:
        from tiny_model import make_tiny_model

        return make_tiny_model().to(device)
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(device)


@torch.no_grad()
def generate(model, text_tools, prompts: list[list[int]], max_new: int, device: str) -> list[str]:
    """Greedy replies for many prompts, 10 at a time, padded on the left."""
    model.eval()
    outs = []
    for i in range(0, len(prompts), 10):
        chunk = prompts[i : i + 10]
        longest = max(len(p) for p in chunk)
        ids = torch.tensor([[text_tools.pad_id] * (longest - len(p)) + p for p in chunk])
        mask = torch.tensor([[0] * (longest - len(p)) + [1] * len(p) for p in chunk])
        new = model.generate(
            input_ids=ids.to(device),
            attention_mask=mask.to(device),
            max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=text_tools.pad_id,
            eos_token_id=text_tools.end_id,
        )[:, longest:]
        for row in new.tolist():
            if text_tools.end_id in row:
                row = row[: row.index(text_tools.end_id)]
            outs.append(text_tools.decode(row))
    model.train()
    return outs


def score(ex, outputs: list[str], tests: list[dict], answers: list[str], general, buzzwords):
    chrf = sum(ex.chrf(o, t["plain"]) for o, t in zip(outputs, tests, strict=True)) / len(tests)
    buzz = sum(ex.count_buzzwords(o, buzzwords) for o in outputs) / len(outputs)
    right = sum(g["expect"] in a.lower() for g, a in zip(general, answers, strict=True))
    return {"chrf": round(chrf, 2), "buzzwords": round(buzz, 3), "general_right": right}


def base_outputs(args, device, text_tools, test_prompts, general_prompts, max_new):
    """The original model's replies. Cached, since they don't depend on your code."""
    out_dir = Path(os.environ.get("FTCOURSE_OUTPUTS") or ROOT / "outputs") / "m05"
    key = hashlib.sha256(
        json.dumps([args.tiny, device, MODEL, max_new, test_prompts, general_prompts]).encode()
    ).hexdigest()[:16]
    cache = out_dir / "base_outputs.json"
    if cache.exists():
        saved = json.loads(cache.read_text())
        if saved.get("key") == key:
            return saved["rewrites"], saved["general"]
    print("Scoring the original model first (once; cached for next time)...")
    model = load_model(args.tiny, device)
    rewrites = generate(model, text_tools, test_prompts, max_new, device)
    general = generate(model, text_tools, general_prompts, max_new, device)
    del model
    out_dir.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"key": key, "rewrites": rewrites, "general": general}) + "\n")
    return rewrites, general


class ExerciseError(Exception):
    """An exercise returned the wrong kind of thing."""


def checked_batch(ex, rows, pad_id):
    batch = ex.collate(rows, pad_id)
    if not isinstance(batch, dict) or not all(
        torch.is_tensor(batch.get(k)) for k in ("input_ids", "attention_mask", "labels")
    ):
        raise ExerciseError(
            "collate should return a dict with input_ids, attention_mask and labels, each a "
            "tensor. Run make check M=05: test_collate shows what's wrong."
        )
    return batch


def checked_loss(ex, logits, labels):
    loss = ex.masked_lm_loss(logits, labels)
    if not torch.is_tensor(loss) or loss.dim() != 0:
        raise ExerciseError(
            "masked_lm_loss should return a one-number tensor (no .item()). Run make check M=05: "
            "test_masked_lm_loss shows what's wrong."
        )
    return loss


@torch.no_grad()
def heldout_loss(model, text_tools, rows, device) -> float:
    """The loss on the held-out reference rewrites, measured with the course's own masking.

    This is the number the checkpoint grades, so a bug in your exercises can't change it.
    """
    model.eval()
    total, count = 0.0, 0
    for i in range(0, len(rows), 8):
        seqs = []
        for r in rows[i : i + 8]:
            p, a = text_tools.prompt_ids(r["jargon"]), text_tools.answer_ids(r["plain"])
            seqs.append((p + a, [-100] * len(p) + a))
        longest = max(len(s) for s, _ in seqs)
        ids = [s + [text_tools.pad_id] * (longest - len(s)) for s, _ in seqs]
        mask = [[1] * len(s) + [0] * (longest - len(s)) for s, _ in seqs]
        labels = [lab + [-100] * (longest - len(lab)) for _, lab in seqs]
        out = model(
            input_ids=torch.tensor(ids, device=device),
            attention_mask=torch.tensor(mask, device=device),
            labels=torch.tensor(labels, device=device),
        )
        total += out.loss.item() * len(seqs)
        count += len(seqs)
    model.train()
    return total / count


def lr_at(step, max_lr, warmup, total):
    if step < warmup:
        return max_lr * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return 0.1 * max_lr + 0.45 * max_lr * (1 + math.cos(math.pi * progress))


def val_loss_loop(ex, model, val_batches, device):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for batch in val_batches:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            total += checked_loss(ex, logits.logits, batch["labels"]).item()
    model.train()
    return total / len(val_batches)


def train_loop(ex, args, model, train, val, pad_id, device):
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    g = torch.Generator().manual_seed(args.seed)
    steps_per_epoch = math.ceil(len(train) / args.batch)
    total = steps_per_epoch * args.epochs
    val_batches = [
        checked_batch(ex, val[i : i + args.batch], pad_id) for i in range(0, len(val), args.batch)
    ]
    initial = val_loss_loop(ex, model, val_batches, device)
    print(f"  before training  val loss {initial:.3f}")
    peak, tokens, step, history = memory_gb(device), 0, 0, []
    start = time.perf_counter()
    for epoch in range(args.epochs):
        order = torch.randperm(len(train), generator=g).tolist()
        for i in range(0, len(train), args.batch):
            batch = checked_batch(ex, [train[j] for j in order[i : i + args.batch]], pad_id)
            batch = {k: v.to(device) for k, v in batch.items()}
            for group in opt.param_groups:
                group["lr"] = lr_at(step, args.lr, max(1, total // 10), total)
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            loss = checked_loss(ex, logits.logits, batch["labels"])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tokens += int(batch["attention_mask"].sum())
            step += 1
            if step % 5 == 0 or step == 1:
                peak = max(peak, memory_gb(device))
        val = val_loss_loop(ex, model, val_batches, device)
        history.append({"epoch": epoch + 1, "train_loss": loss.item(), "val_loss": val})
        took = time.perf_counter() - start
        print(f"  epoch {epoch + 1}  train loss {loss.item():.3f}  val loss {val:.3f}  {took:.0f}s")
    seconds = time.perf_counter() - start
    return {
        "initial_val_loss": initial,
        "val_loss": history[-1]["val_loss"],
        "history": history,
        "peak": peak,
        "tokens_per_sec": tokens / seconds,
        "seconds": seconds,
        "optimizer_state_bytes": ex.optimizer_state_bytes(opt),
    }


def train_trainer(ex, args, model, train, val, pad_id, device):
    from transformers import Trainer, TrainerCallback, TrainingArguments

    peak = [memory_gb(device)]

    class Memory(TrainerCallback):
        def on_step_end(self, *_, **__):
            peak[0] = max(peak[0], memory_gb(device))

    def as_dicts(rows):
        return [{"input_ids": ids, "labels": labels} for ids, labels in rows]

    checked_batch(ex, train[:2], pad_id)  # fail early, with a clear message

    def collator(features):
        return ex.collate([(f["input_ids"], f["labels"]) for f in features], pad_id)

    with tempfile.TemporaryDirectory() as tmp:
        targs = TrainingArguments(
            output_dir=tmp,
            per_device_train_batch_size=args.batch,
            per_device_eval_batch_size=args.batch,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_steps=max(1, math.ceil(len(train) / args.batch) * args.epochs // 10),
            weight_decay=0.0,
            max_grad_norm=1.0,
            eval_strategy="epoch",
            logging_strategy="epoch",
            save_strategy="no",
            report_to=[],
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            use_cpu=device == "cpu",
            seed=args.seed,
            disable_tqdm=True,
        )
        trainer = Trainer(
            model=model,
            args=targs,
            train_dataset=as_dicts(train),
            eval_dataset=as_dicts(val),
            data_collator=collator,
            callbacks=[Memory()],
        )
        initial = trainer.evaluate()["eval_loss"]
        print(f"  before training  val loss {initial:.3f}")
        start = time.perf_counter()
        trainer.train()
        seconds = time.perf_counter() - start
        val_loss = trainer.evaluate()["eval_loss"]
        history = [
            {"epoch": round(h["epoch"]), "val_loss": h["eval_loss"]}
            for h in trainer.state.log_history
            if "eval_loss" in h
        ]
        tokens = sum(len(ids) for ids, _ in train) * args.epochs
        state = ex.optimizer_state_bytes(trainer.optimizer)
    print(f"  after training   val loss {val_loss:.3f}  {seconds:.0f}s")
    return {
        "initial_val_loss": initial,
        "val_loss": val_loss,
        "history": history,
        "peak": peak[0],
        "tokens_per_sec": tokens / seconds,
        "seconds": seconds,
        "optimizer_state_bytes": state,
    }


def where(e: Exception) -> str:
    frames = [
        f for f in traceback.extract_tb(e.__traceback__) if f.filename.endswith("exercises.py")
    ]
    return f" It happened in {frames[-1].name}, line {frames[-1].lineno}." if frames else ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--loop", action="store_true", help="train with your own loop")
    mode.add_argument("--trainer", action="store_true", help="train with Hugging Face's Trainer")
    ap.add_argument("--tiny", action="store_true", help="a tiny random model on the CPU, seconds")
    ap.add_argument("--epochs", type=int, default=2, choices=range(1, 11), metavar="1-10")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max-len", type=int, default=160)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    name = "loop" if args.loop else "trainer"
    if args.tiny and args.lr == ap.get_default("lr"):
        args.lr = 3e-3  # a tiny random model needs a much bigger step

    ex = load_exercise(MODULE_DIR)
    exercises_file = MODULE_DIR / ("solutions" if os.environ.get("FTCOURSE_SOLUTIONS") else "")
    exercises_sha256 = hashlib.sha256((exercises_file / "exercises.py").read_bytes()).hexdigest()
    torch.manual_seed(args.seed)
    device = "mps" if torch.backends.mps.is_available() and not args.tiny else "cpu"
    buzzwords = [w for w in (DATA / "buzzwords.txt").read_text().splitlines() if w.strip()]
    train_rows, val_rows, test_rows = (
        read_jsonl(DATA / f"{s}.jsonl") for s in ("train", "val", "test")
    )
    general = read_jsonl(DATA / "general.jsonl")
    if args.tiny:
        train_rows, val_rows, test_rows, general = (
            train_rows[:32],
            val_rows[:8],
            test_rows[:4],
            general[:2],
        )
    max_new = 16 if args.tiny else 96

    try:
        text_tools = TinyText() if args.tiny else RealText()
        test_prompts = [text_tools.prompt_ids(t["jargon"]) for t in test_rows]
        general_prompts = [text_tools.prompt_ids(g["question"], system=None) for g in general]
        base_rewrites, base_general = base_outputs(
            args, device, text_tools, test_prompts, general_prompts, max_new
        )

        def examples(rows):
            return [
                ex.build_example(
                    text_tools.prompt_ids(r["jargon"]),
                    text_tools.answer_ids(r["plain"]),
                    args.max_len,
                )
                for r in rows
            ]

        train, val = examples(train_rows), examples(val_rows)
        model = load_model(args.tiny, device)
        params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nFine-tuning all {params:,} parameters with your {name}, on {device}")
        print(f"{len(train)} examples, {args.epochs} epochs, batches of {args.batch}\n")
        heldout_before = heldout_loss(model, text_tools, val_rows, device)
        run = (train_loop if args.loop else train_trainer)(
            ex, args, model, train, val, text_tools.pad_id, device
        )
        heldout_after = heldout_loss(model, text_tools, val_rows, device)
        tuned_rewrites = generate(model, text_tools, test_prompts, max_new, device)
        tuned_general = generate(model, text_tools, general_prompts, max_new, device)
        base = score(ex, base_rewrites, test_rows, base_general, general, buzzwords)
        tuned = score(ex, tuned_rewrites, test_rows, tuned_general, general, buzzwords)
    except NotImplementedError as e:
        print(f"Not yet: {e} isn't written. Finish all six exercises, then run this again.")
        return 1
    except ExerciseError as e:
        print(f"Your exercises returned something unexpected: {e}")
        return 1
    except RuntimeError as e:
        if "out of memory" not in str(e).lower():
            if where(e):
                print(f"One of your exercises raised RuntimeError: {e}.{where(e)}")
                print("Run make check M=05 to see which test fails, and why.")
                return 1
            raise
        print("Your Mac ran out of memory for this run (it needs about 24 GB with the defaults).")
        print("Close other apps and the lesson notebook, then try again. With less memory, run")
        print(f"  uv run python modules/m05_full_finetune/train.py --{name} --batch 4")
        print("which needs about 17 GB.")
        return 1
    except OSError as e:
        print(f"Couldn't load {MODEL} ({e}).")
        print("Connect to the internet once so it can download (about 1 GB), or use --tiny.")
        return 1
    except Exception as e:  # a bug in an exercise: say so plainly, not with a traceback
        if not where(e):
            raise  # not from exercises.py: show the full error
        print(f"One of your exercises raised {type(e).__name__}: {e}.{where(e)}")
        print("Run make check M=05 to see which test fails, and why.")
        return 1

    print("\nThree held-out examples:")
    for t, b, f in list(zip(test_rows, base_rewrites, tuned_rewrites, strict=True))[:3]:
        print(f"  jargon:     {t['jargon']}\n  original:   {b}")
        print(f"  fine-tuned: {f}\n  reference:  {t['plain']}\n")
    print(f"{'':12}{'held-out loss':>14}{'chrF':>7}{'buzzwords':>11}{'general':>9}")
    for label, loss, s in [
        ("original", heldout_before, base),
        ("fine-tuned", heldout_after, tuned),
    ]:
        print(
            f"{label:12}{loss:14.3f}{s['chrf']:7.1f}{s['buzzwords']:11.2f}"
            f"{s['general_right']:6d}/{len(general)}"
        )

    summary = {
        "mode": name,
        "heldout_before": round(heldout_before, 4),
        "heldout_after": round(heldout_after, 4),
        "val_loss": round(run["val_loss"], 4),
        "initial_val_loss": round(run["initial_val_loss"], 4),
        "base": base,
        "tuned": tuned,
        "general_total": len(general),
        "peak_mem_gb": round(run["peak"], 2),
        "tokens_per_sec": round(run["tokens_per_sec"], 1),
        "optimizer_state_gb": round(run["optimizer_state_bytes"] / 1e9, 3),
        "params": params,
        "trainable_pct": round(100 * trainable / params, 2),
        "epochs": args.epochs,
        "lr": args.lr,
        "device": device,
        "seconds": round(run["seconds"], 1),
        "tiny": args.tiny,
        "used_solutions": bool(os.environ.get("FTCOURSE_SOLUTIONS")),
        "exercises_sha256": exercises_sha256,
        "history": run["history"],
        "examples": [
            {"jargon": t["jargon"], "original": b, "fine_tuned": f, "reference": t["plain"]}
            for t, b, f in zip(test_rows, base_rewrites, tuned_rewrites, strict=True)
        ],
    }
    out = Path(os.environ.get("FTCOURSE_OUTPUTS") or ROOT / "outputs") / "m05"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"run_{name}.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"\nPeak memory {summary['peak_mem_gb']} GB (AdamW's state alone: "
        f"{summary['optimizer_state_gb']} GB), {summary['tokens_per_sec']:.0f} tokens/s, "
        f"{summary['seconds']:.0f}s of training"
    )
    print(f"Saved {out / f'run_{name}.json'}. Next: make check M=05")
    return 0


if __name__ == "__main__":
    sys.exit(main())
