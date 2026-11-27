"""Train the same LoRA adapter on Qwen2.5-1.5B-Instruct with mlx-lm and with PEFT, side by side.

    uv run python modules/m07_lora_tools/train.py               # both tools, one after the other
    uv run python modules/m07_lora_tools/train.py --tool mlx    # just mlx-lm
    uv run python modules/m07_lora_tools/train.py --tool peft   # just PEFT
    uv run python modules/m07_lora_tools/train.py --tiny        # tiny random models, seconds

Both tools get the same batches in the same order, the same adapter (rank 8, alpha 16, on
q, k, v and o in all 28 blocks), the same learning-rate schedule and the same number of steps.
Each runs in its own process, so its peak memory is its own. After training, each adapter is
saved, the model is thrown away, and the adapter is loaded back onto a fresh model with your
exercises 5 and 6, which must give the same rewrites. Writes outputs/m07/{mlx,peft}/run.json and
outputs/m07/run.json, which `make check M=07` reads.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
ROOT = MODULE_DIR.parent.parent
M5_DIR = ROOT / "modules" / "m05_full_finetune"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(MODULE_DIR))

from common.checkpoint import load_exercise  # noqa: E402

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
SYSTEM = "Rewrite the user's text in plain English. Reply with the rewrite only."
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]
PAD_TO = 32  # pad every batch to a multiple of 32 tokens: fewer shapes for mlx to compile


def _mlx_works() -> bool:
    """mlx-lm is installed and MLX can use a GPU (Apple-silicon Macs only)."""
    if sys.platform != "darwin" or importlib.util.find_spec("mlx_lm") is None:
        return False
    try:
        import mlx.core as mx

        return bool(mx.metal.is_available())
    except Exception:
        return False


HAS_MLX = _mlx_works()


class ExerciseError(Exception):
    """An exercise returned the wrong kind of thing."""


def outputs_dir() -> Path:
    return Path(os.environ.get("FTCOURSE_OUTPUTS") or ROOT / "outputs") / "m07"


def load_m05_train():
    """M5's train.py, as a library: text tools, batches, generation and scoring."""
    spec = importlib.util.spec_from_file_location("m05_train", M5_DIR / "train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.MODEL = MODEL  # the 1.5B model has the same tokenizer and chat template as M5's 0.5B
    return mod


# --- memory: the same measure for both tools ---


class _RUsage(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [
        (n, ctypes.c_uint64)
        for n in (
            "user system pkg_idle_wkups interrupt_wkups pageins wired_size resident_size "
            "phys_footprint start_abstime exit_abstime child_user child_system "
            "child_pkg_idle_wkups child_interrupt_wkups child_pageins child_elapsed_abstime "
            "diskio_bytesread diskio_byteswritten qos_default qos_maintenance qos_background "
            "qos_utility qos_legacy qos_user_initiated qos_user_interactive billed_system_time "
            "serviced_system_time logical_writes lifetime_max_phys_footprint instructions cycles "
            "billed_energy serviced_energy interval_max_phys_footprint runnable_time "
            "spare1 spare2 spare3 spare4"
        ).split()
    ]


def process_peak_gb() -> float:
    """This process's largest memory footprint so far, as macOS counts it (GPU memory too).

    It's what Activity Monitor shows as "Memory", and it means the same for mlx and PyTorch.
    """
    if sys.platform == "darwin":
        info = _RUsage()
        lib = ctypes.CDLL("/usr/lib/libproc.dylib")
        if lib.proc_pid_rusage(os.getpid(), 4, ctypes.byref(info)) == 0:
            return info.lifetime_max_phys_footprint / 1e9
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # Linux: kilobytes


# --- data: your exercise 1, then your M5 build_example and collate ---


def chat_records(ex, rows):
    records = [ex.to_chat_record(r, SYSTEM) for r in rows]
    for rec in records[:1]:
        msgs = rec.get("messages") if isinstance(rec, dict) else None
        roles = [m.get("role") for m in msgs] if isinstance(msgs, list) else None
        if roles != ["system", "user", "assistant"]:
            raise ExerciseError(
                'to_chat_record should return {"messages": [system, user, assistant]}. Run make '
                "check M=07: test_to_chat_record shows what's wrong."
            )
    return records


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def build_batches(m5, ex5, text_tools, records, args, epochs, shuffle):
    """Padded batches, in the order both tools will see them."""
    import torch

    examples = []
    for rec in records:
        system, user, answer = (m["content"] for m in rec["messages"])
        examples.append(
            ex5.build_example(
                text_tools.prompt_ids(user, system=system),
                text_tools.answer_ids(answer),
                args.max_len,
            )
        )
    g = torch.Generator().manual_seed(args.seed)
    batches = []
    for _ in range(epochs):
        order = torch.randperm(len(examples), generator=g).tolist() if shuffle else None
        order = order or list(range(len(examples)))
        for i in range(0, len(order), args.batch):
            rows = [examples[j] for j in order[i : i + args.batch]]
            batch = m5.checked_batch(ex5, rows, text_tools.pad_id)
            batches.append(pad_batch(batch, text_tools.pad_id))
    return batches


def pad_batch(batch, pad_id):
    import torch.nn.functional as F

    extra = -batch["input_ids"].shape[1] % PAD_TO
    return {
        "input_ids": F.pad(batch["input_ids"], (0, extra), value=pad_id),
        "attention_mask": F.pad(batch["attention_mask"], (0, extra), value=0),
        "labels": F.pad(batch["labels"], (0, extra), value=-100),
    }


def lr_schedule(args, steps):
    """M5's schedule: warm up over the first tenth, then a cosine down to a tenth."""
    return max(1, steps // 10), steps


# --- PEFT ---


def peft_load_base(args, device):
    import torch

    if args.tiny:
        from tiny_models import make_tiny_hf_model

        return make_tiny_hf_model().to(device)
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(device)
    # Qwen's generation_config.json asks for a repetition penalty, which Hugging Face applies even
    # to greedy replies, and mlx-lm doesn't. Turn it off, so both tools decode the same way.
    model.generation_config.repetition_penalty = 1.0
    return model


def peft_train(ex, m5, args, model, batches, device):
    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, get_peft_model

    cfg = ex.peft_lora_config(args.r, args.alpha, TARGETS)
    if not isinstance(cfg, LoraConfig):
        raise ExerciseError(f"peft_lora_config should return a LoraConfig, not {type(cfg)}.")
    if cfg.lora_alpha != args.alpha:
        raise ExerciseError(
            f"peft_lora_config set lora_alpha={cfg.lora_alpha}, but PEFT wants alpha itself "
            f"({args.alpha:g}): it works out the scale on its own (exercise 3)."
        )
    model = get_peft_model(model, cfg)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    warmup, total = lr_schedule(args, len(batches))
    framework_peak, history = 0.0, []
    sync = torch.mps.synchronize if device == "mps" else lambda: None
    sync()
    start = time.perf_counter()
    for step, batch in enumerate(batches):
        for group in opt.param_groups:
            group["lr"] = m5.lr_at(step, args.lr, warmup, total)
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        loss = F.cross_entropy(
            logits[:, :-1].float().reshape(-1, logits.shape[-1]),
            batch["labels"][:, 1:].reshape(-1),
            ignore_index=-100,
        )
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if device == "mps":
            framework_peak = max(framework_peak, torch.mps.driver_allocated_memory() / 1e9)
            if step % 10 == 9:
                torch.mps.empty_cache()  # PyTorch keeps freed memory; hand it back now and then
        if step % 10 == 9 or step == len(batches) - 1:
            history.append({"step": step + 1, "train_loss": round(loss.item(), 4)})
            print(f"  step {step + 1:4d}/{len(batches)}  train loss {loss.item():.3f}", flush=True)
    sync()
    return model, time.perf_counter() - start, framework_peak, history


def peft_trainable(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


# --- mlx-lm ---


def mlx_load_base(args):
    from mlx_lm import load

    return load(str(tiny_mlx_dir()) if args.tiny else MODEL)


def tiny_mlx_dir() -> Path:
    from tiny_models import save_tiny_mlx_model

    folder = outputs_dir() / "tiny_mlx_model"
    if not (folder / "model.safetensors").exists():
        save_tiny_mlx_model(folder)
    return folder


def mlx_loss(model, ids, labels):
    """The same loss as PyTorch's: cross-entropy on the answer tokens only."""
    import mlx.core as mx
    import mlx.nn as nn

    logits = model(ids[:, :-1]).astype(mx.float32)
    targets = labels[:, 1:]
    mask = targets != -100
    ce = nn.losses.cross_entropy(logits, mx.where(mask, targets, 0)) * mask
    n = mask.sum()
    return ce.sum() / n, n


def mlx_train(ex, args, model, batches, adapter_dir):
    import mlx.core as mx
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten
    from mlx_lm.tuner import trainer
    from mlx_lm.tuner.utils import linear_to_lora_layers

    cfg = ex.mlx_lora_config(args.r, args.alpha, TARGETS, len(model.layers))
    lora = cfg.get("lora_parameters", {}) if isinstance(cfg, dict) else {}
    if not isinstance(cfg, dict) or "num_layers" not in cfg or "keys" not in lora:
        raise ExerciseError(
            'mlx_lora_config should return {"num_layers": ..., "lora_parameters": {...}}. Run '
            "make check M=07: test_mlx_lora_config shows what's wrong."
        )
    if lora.get("scale") != args.alpha / args.r:
        raise ExerciseError(
            f"mlx_lora_config set scale={lora.get('scale')}, but mlx-lm multiplies B @ A by the "
            f"scale itself: alpha / r = {args.alpha / args.r:g} (exercise 2)."
        )
    model.freeze()
    mx.random.seed(args.seed)  # A's random start, so a run can be repeated
    linear_to_lora_layers(model, cfg["num_layers"], lora)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps({"fine_tune_type": "lora", **cfg}, indent=2) + "\n"
    )

    warmup, total = lr_schedule(args, len(batches))
    schedule = optim.join_schedules(  # the same numbers as M5's lr_at
        [
            optim.linear_schedule(args.lr / warmup, args.lr * (1 + 1 / warmup), warmup),
            optim.cosine_decay(args.lr, max(1, total - warmup), 0.1 * args.lr),
        ],
        [warmup],
    )

    class ClippedAdamW(optim.AdamW):  # clip the gradients at 1.0, as the PyTorch loop does
        def apply_gradients(self, gradients, parameters):
            gradients, _ = optim.clip_grad_norm(gradients, 1.0)
            return super().apply_gradients(gradients, parameters)

    opt = ClippedAdamW(learning_rate=schedule, weight_decay=0.0)
    mlx_batches = [
        (mx.array(b["input_ids"].numpy()), mx.array(b["labels"].numpy())) for b in batches
    ]
    history = []

    class Report(trainer.TrainingCallback):
        def on_train_loss_report(self, info):
            history.append({"step": info["iteration"], "train_loss": round(info["train_loss"], 4)})

    def same_batches(**_):  # every tool sees the same batches, in the same order
        yield from mlx_batches

    mx.reset_peak_memory()
    start = time.perf_counter()
    trainer.train(
        model,
        opt,
        train_dataset=mlx_batches,
        args=trainer.TrainingArgs(
            batch_size=args.batch,
            iters=len(mlx_batches),
            steps_per_report=10,
            steps_per_eval=10**9,
            steps_per_save=10**9,
            max_seq_length=10**6,
            adapter_file=str(adapter_dir / "adapters.safetensors"),
        ),
        loss=mlx_loss,
        iterate_batches=same_batches,
        training_callback=Report(),
    )
    seconds = time.perf_counter() - start
    trainable = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    return model, seconds, mx.get_peak_memory() / 1e9, history, trainable


def mlx_heldout_loss(model, batches) -> float:
    """The held-out loss, averaged over sequences the way M5's heldout_loss does."""
    import mlx.core as mx

    total, count = 0.0, 0
    for b in batches:
        loss, _ = mlx_loss(model, mx.array(b["input_ids"].numpy()), mx.array(b["labels"].numpy()))
        total += loss.item() * b["input_ids"].shape[0]
        count += b["input_ids"].shape[0]
    return total / count


def mlx_generate(model, text_tools, prompts, max_new) -> list[str]:
    """Greedy replies, one prompt at a time."""
    import mlx.core as mx
    from mlx_lm.generate import generate_step

    outs = []
    for p in prompts:
        ids = []
        for token, _ in generate_step(mx.array(p), model, max_tokens=max_new):
            token = int(token)
            if token == text_tools.end_id:
                break
            ids.append(token)
        outs.append(text_tools.decode(ids))
    return outs


# --- one tool, in its own process ---


def held_out_batches(m5, ex5, text_tools, records, args):
    """Batches of 8 for the held-out loss, the same for both tools."""
    return build_batches(m5, ex5, text_tools, records, args, epochs=1, shuffle=False)


def torch_heldout_loss(model, batches, device) -> float:
    import torch
    import torch.nn.functional as F

    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for b in batches:
            b = {k: v.to(device) for k, v in b.items()}
            logits = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits
            loss = F.cross_entropy(
                logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                b["labels"][:, 1:].reshape(-1),
                ignore_index=-100,
            )
            total += loss.item() * b["input_ids"].shape[0]
            count += b["input_ids"].shape[0]
    model.train()
    return total / count


def free(device):
    gc.collect()
    if device == "mps":
        import torch

        torch.mps.empty_cache()
    if HAS_MLX:
        import mlx.core as mx

        mx.clear_cache()


def run_tool(tool, args) -> dict:
    import torch

    m5 = load_m05_train()
    ex = load_exercise(MODULE_DIR)
    ex5 = load_exercise(M5_DIR)
    torch.manual_seed(args.seed)
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
    out = outputs_dir() / tool
    adapter_dir = out / "adapter"

    text_tools = m5.TinyText() if args.tiny else m5.RealText()
    records = {s: chat_records(ex, r) for s, r in [("train", train_rows), ("valid", val_rows)]}
    for split, recs in records.items():  # the files mlx-lm's own command line would read
        write_jsonl(outputs_dir() / "data" / f"{split}.jsonl", recs)
    batches = build_batches(m5, ex5, text_tools, records["train"], args, args.epochs, True)
    val = held_out_batches(m5, ex5, text_tools, records["valid"], args)
    test_prompts = [text_tools.prompt_ids(t["jargon"]) for t in test_rows]
    general_prompts = [text_tools.prompt_ids(g["question"], system=None) for g in general]
    prompts = test_prompts + general_prompts
    n_test = len(test_prompts)

    device = "mps" if tool == "peft" and torch.backends.mps.is_available() else "cpu"
    if args.tiny:
        device = "cpu"
    else:
        print(f"Loading {MODEL} for {tool}...", flush=True)
    if tool == "peft":
        model = peft_load_base(args, device)
        total_before = sum(p.numel() for p in model.parameters())
        heldout_before = torch_heldout_loss(model, val, device)
        base_out = m5.generate(model, text_tools, prompts, max_new, device)
        model, seconds, framework_peak, history = peft_train(ex, m5, args, model, batches, device)
        peak = process_peak_gb()
        trainable, _ = peft_trainable(model)
        heldout_after = torch_heldout_loss(model, val, device)
        tuned_out = m5.generate(model, text_tools, prompts, max_new, device)
        model.save_pretrained(adapter_dir)
        del model
        free(device)
        loaded = ex.load_peft_adapter(peft_load_base(args, device), str(adapter_dir))
        from peft import PeftModel

        if not isinstance(loaded, PeftModel):
            raise ExerciseError(
                "load_peft_adapter should return the PeftModel that PeftModel.from_pretrained "
                "gives you (exercise 5)."
            )
        reload_out = m5.generate(loaded, text_tools, prompts, max_new, device)
        heldout_reloaded = torch_heldout_loss(loaded, val, device)
        adapter_file = adapter_dir / "adapter_model.safetensors"
    else:
        model, tok = mlx_load_base(args)
        from mlx.utils import tree_flatten

        total_before = sum(v.size for _, v in tree_flatten(model.parameters()))
        heldout_before = mlx_heldout_loss(model, val)
        base_out = mlx_generate(model, text_tools, prompts, max_new)
        model, seconds, framework_peak, history, trainable = mlx_train(
            ex, args, model, batches, adapter_dir
        )
        peak = process_peak_gb()
        heldout_after = mlx_heldout_loss(model, val)
        tuned_out = mlx_generate(model, text_tools, prompts, max_new)
        del model, tok
        free(device)
        loaded = ex.load_mlx_adapter(str(tiny_mlx_dir()) if args.tiny else MODEL, str(adapter_dir))
        if not isinstance(loaded, tuple) or len(loaded) != 2:
            raise ExerciseError(
                "load_mlx_adapter should return what mlx_lm.load returns: (model, tokenizer) "
                "(exercise 6)."
            )
        reload_out = mlx_generate(loaded[0], text_tools, prompts, max_new)
        heldout_reloaded = mlx_heldout_loss(loaded[0], val)
        adapter_file = adapter_dir / "adapters.safetensors"

    tps = ex.tokens_per_sec([b["attention_mask"] for b in batches], seconds)
    if not isinstance(tps, int | float):
        raise ExerciseError(f"tokens_per_sec should return a number, not {type(tps).__name__}.")
    padded = sum(b["attention_mask"].numel() for b in batches) / seconds
    base = m5.score(ex5, base_out[:n_test], test_rows, base_out[n_test:], general, buzzwords)
    tuned = m5.score(ex5, tuned_out[:n_test], test_rows, tuned_out[n_test:], general, buzzwords)
    matches = sum(a == b for a, b in zip(tuned_out, reload_out, strict=True))
    exercises_file = MODULE_DIR / ("solutions" if os.environ.get("FTCOURSE_SOLUTIONS") else "")
    summary = {
        "tool": tool,
        "heldout_before": round(heldout_before, 4),
        "heldout_after": round(heldout_after, 4),
        "heldout_reloaded": round(heldout_reloaded, 4),
        "final_loss": round(heldout_after, 4),
        "base": base,
        "tuned": tuned,
        "general_total": len(general),
        "reload_matches": matches,
        "reload_total": len(prompts),
        "trainable": int(trainable),
        "params": int(total_before),
        "trainable_pct": round(100 * trainable / total_before, 3),
        "adapter_mb": round(adapter_file.stat().st_size / 1e6, 2),
        "peak_mem_gb": round(peak, 2),
        "framework_peak_gb": round(framework_peak, 2),
        "tokens_per_sec": round(float(tps), 1),
        "padded_tokens_per_sec": round(padded, 1),
        "seconds": round(seconds, 1),
        "steps": len(batches),
        "r": args.r,
        "alpha": args.alpha,
        "lr": args.lr,
        "epochs": args.epochs,
        "dtype": "float32" if args.tiny and tool == "peft" else "bfloat16",
        "device": "gpu (mlx)" if tool == "mlx" else device,
        "tiny": args.tiny,
        "used_solutions": bool(os.environ.get("FTCOURSE_SOLUTIONS")),
        "exercises_sha256": hashlib.sha256(
            (exercises_file / "exercises.py").read_bytes()
        ).hexdigest(),
        "history": history,
        "examples": [
            {
                "jargon": t["jargon"],
                "original": b,
                "tuned": f,
                "reloaded": r,
                "reference": t["plain"],
            }
            for t, b, f, r in zip(test_rows, base_out, tuned_out, reload_out, strict=False)
        ],
        "general": [
            {"question": g["question"], "original": b, "tuned": f}
            for g, b, f in zip(general, base_out[n_test:], tuned_out[n_test:], strict=True)
        ],
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def where(e: Exception) -> str:
    frames = [
        f for f in traceback.extract_tb(e.__traceback__) if f.filename.endswith("exercises.py")
    ]
    if not frames:
        return ""
    module = "M5" if "m05_" in frames[-1].filename else "M7"
    return f" It happened in your {module} {frames[-1].name}, line {frames[-1].lineno}."


def run_one(tool, args) -> int:
    """Train with one tool, with plain messages instead of tracebacks for the usual problems."""
    try:
        s = run_tool(tool, args)
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
    except OSError as e:
        if where(e):
            print(f"One of your exercises raised {type(e).__name__}: {e}.{where(e)}")
            return 1
        print(f"Couldn't load {MODEL} ({e}).")
        print("Connect to the internet once so it can download (about 3 GB), or use --tiny.")
        return 1
    except Exception as e:  # a bug in an exercise: say so plainly, not with a traceback
        if "out of memory" in str(e).lower():
            print("Your Mac ran out of memory for this run. Close other apps and the lesson")
            print("notebook, then try again.")
            return 1
        if not where(e):
            raise
        print(f"One of your exercises raised {type(e).__name__}: {e}.{where(e)}")
        print("Run make check M=07 (or M=05) to see which test fails, and why.")
        return 1
    print(
        f"\n{tool}: held-out loss {s['heldout_before']:.3f} -> {s['heldout_after']:.3f}, "
        f"{s['tokens_per_sec']:.0f} tokens/s, peak {s['peak_mem_gb']:.1f} GB, "
        f"{s['seconds']:.0f}s of training. Reloaded adapter gave the same reply for "
        f"{s['reload_matches']}/{s['reload_total']} prompts."
    )
    return 0


ROWS = [
    ("held-out loss, before", "heldout_before", "{:.3f}"),
    ("held-out loss, after", "heldout_after", "{:.3f}"),
    ("buzzwords per rewrite", None, "{:.2f}"),
    ("general questions right", None, "{}"),
    ("training time (s)", "seconds", "{:.0f}"),
    ("tokens/sec (real tokens)", "tokens_per_sec", "{:.0f}"),
    ("tokens/sec (with padding)", "padded_tokens_per_sec", "{:.0f}"),
    ("peak memory (GB, macOS)", "peak_mem_gb", "{:.1f}"),
    ("peak memory (GB, its own count)", "framework_peak_gb", "{:.1f}"),
    ("trainable parameters (%)", "trainable_pct", "{:.3f}"),
    ("adapter file (MB)", "adapter_mb", "{:.1f}"),
    ("reloaded replies the same", None, "{}"),
]


def compare() -> None:
    """Put the runs side by side, and write outputs/m07/run.json for the checkpoint."""
    runs = {}
    for tool in ("mlx", "peft"):
        f = outputs_dir() / tool / "run.json"
        if f.exists():
            runs[tool] = json.loads(f.read_text())
    if not runs:
        return
    print(f"\n{'':34}" + "".join(f"{t:>12}" for t in runs))
    for label, key, fmt in ROWS:
        cells = []
        for s in runs.values():
            if label.startswith("buzzwords"):
                value = f"{s['base']['buzzwords']:.2f}->{s['tuned']['buzzwords']:.2f}"
            elif label.startswith("general"):
                value = f"{s['tuned']['general_right']}/{s['general_total']}"
            elif label.startswith("reloaded"):
                value = f"{s['reload_matches']}/{s['reload_total']}"
            else:
                value = fmt.format(s[key])
            cells.append(f"{value:>12}")
        print(f"{label:34}" + "".join(cells))
    main_run = runs.get("mlx") or runs["peft"]  # the dashboard follows mlx-lm, which M8 uses
    keys = ["final_loss", "heldout_after", "peak_mem_gb", "tokens_per_sec", "trainable_pct"]
    summary = {k: main_run[k] for k in keys} | {"shown": main_run["tool"], "tools": runs}
    (outputs_dir() / "run.json").write_text(json.dumps(summary, indent=2) + "\n")
    (outputs_dir() / "compare.json").write_text(
        json.dumps({t: {k: s[k] for _, k, _ in ROWS if k} for t, s in runs.items()}, indent=2)
        + "\n"
    )
    print(f"\nSaved {outputs_dir() / 'run.json'}. Next: make check M=07")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tool", choices=["mlx", "peft", "both"], default="both")
    ap.add_argument("--tiny", action="store_true", help="tiny random models, seconds")
    ap.add_argument("--epochs", type=int, default=2, choices=range(1, 11), metavar="1-10")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--r", type=int, default=8, help="the adapter's rank")
    ap.add_argument("--alpha", type=float, default=16.0)
    ap.add_argument("--max-len", type=int, default=160)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.tiny and args.lr == ap.get_default("lr"):
        args.lr = 3e-2  # a tiny random model needs a much bigger step

    if args.tool != "both":
        if args.tool == "mlx" and not HAS_MLX:
            print("mlx-lm runs only on Apple-silicon Macs. Use --tool peft here.")
            return 1
        code = run_one(args.tool, args)
        compare()
        return code

    tools = ["mlx", "peft"] if HAS_MLX else ["peft"]
    if not HAS_MLX:
        print("mlx-lm runs only on Apple-silicon Macs: training with PEFT only.")
    passthrough = [f"--{k.replace('_', '-')}={v}" for k, v in vars(args).items() if k != "tool"]
    passthrough = [a.split("=")[0] for a in passthrough if a.endswith("=True")] + [
        a for a in passthrough if not a.endswith(("=True", "=False"))
    ]
    for tool in tools:  # each in its own process, so each peak memory is its own
        print(f"\n=== {tool} ===", flush=True)
        code = subprocess.run([sys.executable, __file__, "--tool", tool, *passthrough]).returncode
        if code != 0:
            return code
    return 0


if __name__ == "__main__":
    sys.exit(main())
