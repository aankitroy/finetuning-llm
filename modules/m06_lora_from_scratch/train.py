"""Tune Qwen2.5-0.5B-Instruct with your LoRA layers: the M5 task, under 1% of the weights.

    uv run python modules/m06_lora_from_scratch/train.py            # rank 8 on q, k, v, o
    uv run python modules/m06_lora_from_scratch/train.py --r 32     # a bigger adapter
    uv run python modules/m06_lora_from_scratch/train.py --targets all   # the MLP layers too
    uv run python modules/m06_lora_from_scratch/train.py --tiny     # a tiny model, seconds

Uses your code in exercises.py (the LoRA parts) and your M5 exercises (batches, loss and
scores). After training it merges the adapter into the weights with your merge_lora, checks the
merged model scores the same, and writes outputs/m06/run.json, which `make check M=06` reads.
Every run is also kept in outputs/m06/sweep/ for the stretch.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path

import torch
import torch.nn as nn

MODULE_DIR = Path(__file__).resolve().parent
ROOT = MODULE_DIR.parent.parent
M5_DIR = ROOT / "modules" / "m05_full_finetune"
sys.path.insert(0, str(ROOT))

from common.checkpoint import load_exercise  # noqa: E402

TARGETS = {
    "attn": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "all": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
}
M5_REFERENCE_LOSS = 0.94  # the reference M5 loop run's held-out loss
M5_PASS_DROP = 0.4  # M5's own bar: a run that didn't meet it isn't used for comparison
M5_CEILING = 1.0  # never compare with a full fine-tune worse than this


def load_m05_train():
    """M5's train.py, as a library: the model, the text tools, generation and the graded loss."""
    spec = importlib.util.spec_from_file_location("m05_train", M5_DIR / "train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def outputs_dir() -> Path:
    return Path(os.environ.get("FTCOURSE_OUTPUTS") or ROOT / "outputs")


def m05_loss() -> tuple[float, str]:
    """The full fine-tune to compare with: your own M5 loop run, or the reference one.

    If your M5 run did better than the reference (more epochs, say), the reference is used,
    so a strong M5 run never makes M6 harder to pass.
    """
    f = outputs_dir() / "m05" / "run_loop.json"
    try:
        s = json.loads(f.read_text())
        mine = float(s["heldout_after"])
        passed = s["heldout_before"] - mine >= M5_PASS_DROP
        usable = passed and not s.get("tiny") and not s.get("used_solutions")
    except (OSError, ValueError, KeyError, TypeError):
        usable = False
    if not usable:
        return M5_REFERENCE_LOSS, "the reference M5 run"
    if mine < M5_REFERENCE_LOSS:
        return M5_REFERENCE_LOSS, "the reference M5 run (yours did even better)"
    return min(mine, M5_CEILING), "your M5 run"


class ExerciseError(Exception):
    """An exercise returned the wrong kind of thing."""


def lora_layers(model):
    return [(n, m) for n, m in model.named_modules() if hasattr(m, "A") and hasattr(m, "B")]


def check_injected(model, replaced, device):
    if not isinstance(replaced, list):
        raise ExerciseError(
            f"inject_lora returned {type(replaced).__name__}, not a list. Return the list of "
            "names you replaced (exercise 3, step 8)."
        )
    if not replaced or not lora_layers(model):
        raise ExerciseError(
            "inject_lora replaced no layers. Run make check M=06: test_inject_lora shows what's "
            "wrong."
        )
    for name, layer in lora_layers(model):
        for part in ("A", "B"):
            p = getattr(layer, part)
            if not isinstance(p, nn.Parameter) or not p.requires_grad:
                raise ExerciseError(
                    f"{name}.{part} isn't a trainable parameter, so nothing would train. Make it "
                    "with nn.Parameter(torch.randn(..., device=...)) (put device= inside, not "
                    ".to() after: that returns a plain tensor), and in inject_lora freeze the "
                    "model first, then swap in the LoRA layers (exercise 3, steps 1 and 7)."
                )
        if layer.A.device.type != torch.device(device).type:
            raise ExerciseError(
                f"{name}.A is on {layer.A.device.type} but the model is on {device}. Create A and "
                "B with device=base.weight.device (exercise 1, step 3)."
            )
    frozen_wrong = [
        n for n, p in model.named_parameters() if p.requires_grad and not n.endswith((".A", ".B"))
    ]
    if frozen_wrong:
        raise ExerciseError(
            f"{frozen_wrong[0]} is still trainable: inject_lora should freeze every original "
            "weight first (exercise 3, step 1)."
        )


def merge_all(ex, model) -> int:
    """Swap every LoRA layer for the plain layer your merge_lora returns."""
    count = 0
    for name, layer in lora_layers(model):
        merged = ex.merge_lora(layer)
        if not isinstance(merged, nn.Linear):
            raise ExerciseError(
                f"merge_lora should return an nn.Linear, not {type(merged).__name__}. "
                "Run make check M=06: test_merge_lora shows what's wrong."
            )
        w = layer.base.weight
        if merged.weight.device != w.device or merged.weight.dtype != w.dtype:
            raise ExerciseError(
                f"merge_lora made its layer on {merged.weight.device.type} in {merged.weight.dtype}"
                f", but the model is on {w.device.type} in {w.dtype}. Pass device= and dtype= "
                "when you make the nn.Linear (exercise 5, step 2)."
            )
        parent_name, _, child = name.rpartition(".")
        setattr(model.get_submodule(parent_name), child, merged)
        count += 1
    return count


def base_outputs(m5, args, device, text_tools, test_prompts, general_prompts, max_new):
    """The original model's replies. Cached, since they don't depend on your code."""
    out_dir = outputs_dir() / "m06"
    key = hashlib.sha256(
        json.dumps([args.tiny, device, m5.MODEL, max_new, test_prompts, general_prompts]).encode()
    ).hexdigest()[:16]
    cache = out_dir / "base_outputs.json"
    if cache.exists():
        saved = json.loads(cache.read_text())
        if saved.get("key") == key:
            return saved["rewrites"], saved["general"]
    print("Scoring the original model first (once; cached for next time)...")
    model = m5.load_model(args.tiny, device)
    rewrites = m5.generate(model, text_tools, test_prompts, max_new, device)
    general = m5.generate(model, text_tools, general_prompts, max_new, device)
    del model
    out_dir.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"key": key, "rewrites": rewrites, "general": general}) + "\n")
    return rewrites, general


def where(e: Exception) -> str:
    frames = [
        f for f in traceback.extract_tb(e.__traceback__) if f.filename.endswith("exercises.py")
    ]
    if not frames:
        return ""
    module = "M5" if "m05_" in frames[-1].filename else "M6"
    return f" It happened in your {module} {frames[-1].name}, line {frames[-1].lineno}."


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--r", type=int, default=8, help="the adapter's rank")
    ap.add_argument("--alpha", type=float, default=None, help="the scale's top half (default 2r)")
    ap.add_argument("--targets", choices=sorted(TARGETS), default="attn")
    ap.add_argument("--tiny", action="store_true", help="a tiny random model on the CPU, seconds")
    ap.add_argument("--epochs", type=int, default=2, choices=range(1, 11), metavar="1-10")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-len", type=int, default=160)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.alpha is None:
        args.alpha = 2.0 * args.r
    if args.tiny and args.lr == ap.get_default("lr"):
        args.lr = 3e-2  # a tiny random model needs a much bigger step

    m5 = load_m05_train()
    ex = load_exercise(MODULE_DIR)
    ex5 = load_exercise(M5_DIR)
    exercises_file = MODULE_DIR / ("solutions" if os.environ.get("FTCOURSE_SOLUTIONS") else "")
    exercises_sha256 = hashlib.sha256((exercises_file / "exercises.py").read_bytes()).hexdigest()
    torch.manual_seed(args.seed)
    device = "mps" if torch.backends.mps.is_available() and not args.tiny else "cpu"
    data = M5_DIR / "data"
    buzzwords = [w for w in (data / "buzzwords.txt").read_text().splitlines() if w.strip()]
    train_rows, val_rows, test_rows = (
        m5.read_jsonl(data / f"{s}.jsonl") for s in ("train", "val", "test")
    )
    general = m5.read_jsonl(data / "general.jsonl")
    if args.tiny:
        train_rows, val_rows, test_rows, general = (
            train_rows[:32],
            val_rows[:8],
            test_rows[:4],
            general[:2],
        )
    max_new = 16 if args.tiny else 96
    targets = TARGETS[args.targets]
    m5_loss, m5_source = m05_loss()

    try:
        text_tools = m5.TinyText() if args.tiny else m5.RealText()
        test_prompts = [text_tools.prompt_ids(t["jargon"]) for t in test_rows]
        general_prompts = [text_tools.prompt_ids(g["question"], system=None) for g in general]
        base_rewrites, base_general = base_outputs(
            m5, args, device, text_tools, test_prompts, general_prompts, max_new
        )

        def examples(rows):
            return [
                ex5.build_example(
                    text_tools.prompt_ids(r["jargon"]),
                    text_tools.answer_ids(r["plain"]),
                    args.max_len,
                )
                for r in rows
            ]

        train, val = examples(train_rows), examples(val_rows)
        model = m5.load_model(args.tiny, device)
        total = sum(p.numel() for p in model.parameters())
        heldout_before = m5.heldout_loss(model, text_tools, val_rows, device)

        torch.manual_seed(args.seed)  # A's random start
        replaced = ex.inject_lora(model, targets, args.r, args.alpha)
        check_injected(model, replaced, device)
        heldout_injected = m5.heldout_loss(model, text_tools, val_rows, device)
        if abs(heldout_injected - heldout_before) > 1e-3:
            raise ExerciseError(
                f"Adding your LoRA layers changed the untrained model's loss "
                f"({heldout_before:.4f} -> {heldout_injected:.4f}). At the start, B is all zeros, "
                "so a LoRA layer must give exactly the base layer's output (exercise 1)."
            )
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())  # now with the adapters
        print(f"\nYour LoRA layers replaced {len(replaced)} layers ({', '.join(targets)}).")
        print(
            f"Training {trainable:,} of {total:,} parameters ({100 * trainable / total:.2f}%), "
            f"on {device}"
        )
        print(f"{len(train)} examples, {args.epochs} epochs, batches of {args.batch}\n")
        run = m5.train_loop(ex5, args, model, train, val, text_tools.pad_id, device)
        heldout_after = m5.heldout_loss(model, text_tools, val_rows, device)

        out = outputs_dir() / "m06"
        out.mkdir(parents=True, exist_ok=True)
        adapter = {}
        for name, layer in lora_layers(model):
            adapter[f"{name}.A"] = layer.A.detach().cpu()
            adapter[f"{name}.B"] = layer.B.detach().cpu()
        torch.save(adapter, out / "adapter.pt")
        adapter_mb = (out / "adapter.pt").stat().st_size / 1e6

        merged = merge_all(ex, model)
        heldout_merged = m5.heldout_loss(model, text_tools, val_rows, device)
        tuned_rewrites = m5.generate(model, text_tools, test_prompts, max_new, device)
        tuned_general = m5.generate(model, text_tools, general_prompts, max_new, device)
        base = m5.score(ex5, base_rewrites, test_rows, base_general, general, buzzwords)
        tuned = m5.score(ex5, tuned_rewrites, test_rows, tuned_general, general, buzzwords)
    except NotImplementedError as e:
        if "M5" in where(e):
            print(f"This run uses your M5 exercises too, and M5's {e} isn't written yet.")
            print("Finish M5 first (make check M=05), then run this again.")
        else:
            print(f"Not yet: {e} isn't written. Finish all five exercises, then run this again.")
        return 1
    except ExerciseError as e:
        print(f"Your exercises returned something unexpected: {e}")
        return 1
    except RuntimeError as e:
        if "out of memory" not in str(e).lower():
            if where(e):
                print(f"One of your exercises raised RuntimeError: {e}.{where(e)}")
                print("Run make check M=06 (or M=05) to see which test fails, and why.")
                return 1
            raise
        print("Your Mac ran out of memory for this run. Close other apps and the lesson")
        print("notebook, then try again, or add --batch 4.")
        return 1
    except OSError as e:
        print(f"Couldn't load {m5.MODEL} ({e}).")
        print("Connect to the internet once so it can download (about 1 GB), or use --tiny.")
        return 1
    except Exception as e:  # a bug in an exercise: say so plainly, not with a traceback
        if not where(e):
            raise  # not from an exercises.py: show the full error
        print(f"One of your exercises raised {type(e).__name__}: {e}.{where(e)}")
        print("Run make check M=06 (or M=05) to see which test fails, and why.")
        return 1

    print("\nThree held-out examples:")
    for t, b, f in list(zip(test_rows, base_rewrites, tuned_rewrites, strict=True))[:3]:
        print(f"  jargon:   {t['jargon']}\n  original: {b}")
        print(f"  LoRA:     {f}\n  reference: {t['plain']}\n")
    print(f"{'':12}{'held-out loss':>14}{'chrF':>7}{'buzzwords':>11}{'general':>9}")
    for label, loss, s in [("original", heldout_before, base), ("LoRA", heldout_after, tuned)]:
        print(
            f"{label:12}{loss:14.3f}{s['chrf']:7.1f}{s['buzzwords']:11.2f}"
            f"{s['general_right']:6d}/{len(general)}"
        )
    print(f"\nFull fine-tune ({m5_source}): held-out loss {m5_loss:.3f}")
    print(
        f"After merging {merged} layers into the weights: held-out loss {heldout_merged:.3f} "
        f"(unmerged {heldout_after:.3f})"
    )

    summary = {
        "heldout_before": round(heldout_before, 4),
        "heldout_after": round(heldout_after, 4),
        "heldout_merged": round(heldout_merged, 4),
        "m05_heldout": m5_loss,
        "m05_source": m5_source,
        "val_loss": round(run["val_loss"], 4),
        "base": base,
        "tuned": tuned,
        "general_total": len(general),
        "r": args.r,
        "alpha": args.alpha,
        "targets": args.targets,
        "layers_replaced": len(replaced),
        "trainable": trainable,
        "params": total,
        "trainable_pct": round(100 * trainable / total, 3),
        "adapter_mb": round(adapter_mb, 2),
        "peak_mem_gb": round(run["peak"], 2),
        "tokens_per_sec": round(run["tokens_per_sec"], 1),
        "optimizer_state_gb": round(run["optimizer_state_bytes"] / 1e9, 4),
        "epochs": args.epochs,
        "lr": args.lr,
        "device": device,
        "seconds": round(run["seconds"], 1),
        "tiny": args.tiny,
        "used_solutions": bool(os.environ.get("FTCOURSE_SOLUTIONS")),
        "exercises_sha256": exercises_sha256,
        "history": run["history"],
        "general": [
            {"question": g["question"], "original": b, "lora": f}
            for g, b, f in zip(general, base_general, tuned_general, strict=True)
        ],
        "examples": [
            {"jargon": t["jargon"], "original": b, "lora": f, "reference": t["plain"]}
            for t, b, f in zip(test_rows, base_rewrites, tuned_rewrites, strict=True)
        ],
    }
    text = json.dumps(summary, indent=2) + "\n"
    (out / "run.json").write_text(text)
    (out / "sweep").mkdir(exist_ok=True)
    tag = f"r{args.r}_{args.targets}{'_tiny' if args.tiny else ''}"
    (out / "sweep" / f"{tag}.json").write_text(text)
    print(
        f"\nPeak memory {summary['peak_mem_gb']} GB (AdamW's state: "
        f"{1000 * summary['optimizer_state_gb']:.1f} MB), {summary['tokens_per_sec']:.0f} "
        f"tokens/s, {summary['seconds']:.0f}s of training. The adapter file is "
        f"{summary['adapter_mb']:.1f} MB."
    )
    print(f"Saved {out / 'run.json'}. Next: make check M=06")
    return 0


if __name__ == "__main__":
    sys.exit(main())
