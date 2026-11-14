# LLM Fine-Tuning, Step by Step — Course Structure

A hands-on course that teaches LLM fine-tuning by building it yourself, one step at a time, on a single Apple Silicon laptop.

**Reference hardware:** MacBook Pro, Apple M5 Max, 36 GB unified memory
**Philosophy:** first build each idea from scratch on a tiny model so you understand it, then use the production tool for the same job on a real model.

---

## 1. Who this is for

- Comfortable with Python and basic NumPy
- Has used an LLM API or chat app, but has never trained or fine-tuned a model
- Wants intuition *and* working code, not just a script copied from somewhere

No prior PyTorch is required. Module 1 covers what you need.

---

## 2. What fits on a 36 GB Mac

Unified memory is shared by the CPU, the GPU and macOS, so plan on roughly **26–28 GB** being usable for training. The figures below are rough guides. Sequence length and batch size move them a lot.

| Technique | Comfortable | Possible, but tight | Notes |
|---|---|---|---|
| Full fine-tune (bf16 + AdamW) | 0.5B | ~1.5B | About 12–16 bytes per parameter, plus activations |
| LoRA (bf16 base) | 1B–4B | 7B–8B | Base weights take ~2 bytes per parameter; adapters are tiny |
| QLoRA (4-bit base) | 7B–8B | 14B | Base weights take ~0.5–0.6 bytes per parameter |
| Inference (4-bit) | ≤ 14B | ~30B | Useful for running the larger "teacher" models used in eval |

**Toolchain choice:**

- **MLX + `mlx-lm`** is the main engine for real fine-tuning runs. It's Apple-native, fast, and supports LoRA, QLoRA and full fine-tuning on Apple Silicon.
- **PyTorch (MPS backend)** is used for the from-scratch modules, because it's the most common framework and the clearest way to learn.
- **Hugging Face `transformers` / `peft` / `trl` / `datasets`** cover the industry-standard APIs. They run on MPS, but `bitsandbytes` 4-bit quantization is CUDA-only, so QLoRA happens in MLX.
- **llama.cpp / Ollama** handle export and local serving.

Default models are small open-weight models from the Qwen, Llama, Gemma, SmolLM and Phi families. Every lesson names a "tiny" model (≤ 0.5B) for fast iteration and a "real" model (1B–8B) for the full run.

---

## 3. Course map

```
Part A  Foundations          M0 Setup → M1 Tensors & PyTorch → M2 How LLMs work → M3 Tokenizers
Part B  Training from zero   M4 Training loop from scratch → M5 Full fine-tune of a small model
Part C  Efficient tuning     M6 LoRA from scratch → M7 LoRA with MLX/PEFT → M8 QLoRA
Part D  Making it useful     M9 Data curation → M10 Instruction & chat tuning → M11 Evaluation
Part E  Alignment            M12 Preference tuning (DPO/ORPO) → M13 RL with verifiable rewards (GRPO)
Part F  Shipping             M14 Merge, quantize, export → M15 Serve & integrate
Capstone                     M16 End-to-end domain specialist
```

Estimated total time: **40–60 hours**. Each module takes about 2–4 hours.

---

## 4. Modules

Each module lists **Goal**, **You'll build**, **Checkpoint** (the auto-graded check that you got it right), and **Stretch**.

### Part A — Foundations

#### M0 · Environment & Apple Silicon setup
- **Goal:** get a reproducible environment and understand unified memory.
- **You'll build:** a `uv`-managed environment, plus a `doctor.py` script that checks MPS and MLX availability, free memory, and the Python version.
- **Concepts:** unified memory vs VRAM, MPS vs MLX, dtypes (fp32/bf16/fp16), watching memory with `asitop` / Activity Monitor.
- **Checkpoint:** `make doctor` passes.
- **Stretch:** benchmark matmul throughput on the CPU, MPS and MLX.

#### M1 · Tensors, autograd & PyTorch essentials
- **Goal:** know just enough PyTorch to read and write training code.
- **You'll build:** linear regression, then a 2-layer MLP, trained with a handwritten loop and no high-level trainer.
- **Concepts:** tensors, broadcasting, `requires_grad`, `.backward()`, optimizers, `nn.Module`, device placement.
- **Checkpoint:** the MLP reaches 95% or better on a toy classification set.

#### M2 · How an LLM actually works
- **Goal:** a working mental model of a decoder-only transformer.
- **You'll build:** load a tiny pretrained model, inspect its layers, extract logits by hand, and write greedy, temperature and top-p sampling yourself.
- **Concepts:** embeddings, attention, the KV cache, next-token prediction, logits → probabilities, what "weights" and "parameters" mean.
- **Checkpoint:** your handwritten generation loop matches `model.generate()` token for token in greedy mode.
- **Stretch:** visualize attention patterns for one prompt.

#### M3 · Tokenizers & chat templates
- **Goal:** understand what the model really sees.
- **You'll build:** a toy BPE tokenizer; then inspect a real tokenizer's special tokens and render conversations through `apply_chat_template`.
- **Concepts:** BPE, special tokens, BOS/EOS, padding sides, chat templates, why template mismatches silently ruin fine-tunes.
- **Checkpoint:** correctly predict the token count and the template output for five tricky inputs.

### Part B — Training from zero

#### M4 · The training loop, from scratch
- **Goal:** demystify "training" completely.
- **You'll build:** a character-level GPT (about 1–10M parameters) trained from zero on a small text corpus.
- **Concepts:** causal LM loss, train/val split, learning-rate schedules (warmup + cosine), gradient clipping, gradient accumulation, overfitting, loss curves.
- **Checkpoint:** validation loss drops below the target threshold and samples look like plausible text.
- **Stretch:** add mixed precision and measure the speedup and the memory saved.

#### M5 · Your first real fine-tune (full fine-tune)
- **Goal:** fine-tune a pretrained ~0.5B model by updating every weight.
- **You'll build:** a full fine-tune for a narrow style task (for example, rewriting text in a fixed tone), done twice: once with your own loop and once with the HF `Trainer`.
- **Concepts:** pretrained vs fine-tuned weights, loss masking on prompt tokens, catastrophic forgetting, measuring the memory cost of optimizer state.
- **Checkpoint:** the fine-tuned model beats the base model on a held-out style test, and you can report the peak memory you measured.
- **Lesson to take away:** you'll see why full fine-tuning doesn't scale on a laptop, which motivates Part C.

### Part C — Parameter-efficient fine-tuning

#### M6 · LoRA from scratch
- **Goal:** understand low-rank adaptation down to the math.
- **You'll build:** a `LoRALinear` layer from scratch; inject it into the M5 model's attention projections; freeze the base; train.
- **Concepts:** the update ΔW = BA, rank `r`, `alpha` scaling, which modules to target, trainable-parameter counts, merging adapters back into the weights.
- **Checkpoint:** a unit test confirms the layer equals the base layer at initialization, the trainable parameter count is under 1%, and quality comes within a few percent of the M5 full fine-tune.
- **Stretch:** sweep `r ∈ {2, 8, 32}` and plot quality against parameter count.

#### M7 · LoRA with real tools (MLX & PEFT)
- **Goal:** run LoRA at a realistic scale with production tooling.
- **You'll build:** the same task on a 1B–4B model, once with `mlx_lm.lora` and once with `peft`, with a side-by-side comparison of speed and memory.
- **Concepts:** adapter configs, saving and loading adapters, the effect of sequence length on memory, tokens/sec as the metric to watch.
- **Checkpoint:** an adapter that loads and produces the expected style on a fixed prompt set.

#### M8 · QLoRA: bigger models, same laptop
- **Goal:** fine-tune a 7B–8B model on 36 GB.
- **You'll build:** quantize a base model to 4-bit in MLX and train LoRA adapters on top of it.
- **Concepts:** quantization (int8/int4, group size), the quality vs memory trade-off, why gradients still flow through the adapters, gradient checkpointing.
- **Checkpoint:** a QLoRA run on a 7B-class model finishes within the memory budget and beats an untuned 4-bit baseline on the eval set.

### Part D — Making it useful

#### M9 · Data: the part that actually matters
- **Goal:** learn to build training data that works.
- **You'll build:** a data pipeline covering loading, cleaning, deduplication (exact and MinHash), length filtering, formatting to chat JSONL, and train/val/test splits. You'll also generate synthetic data with a larger local model and filter it.
- **Concepts:** quality over quantity, data contamination, diversity, licensing, synthetic data pitfalls, how many examples are "enough".
- **Checkpoint:** the dataset validator passes (schema, no leakage between splits, length distribution within bounds).

#### M10 · Instruction & chat tuning (SFT)
- **Goal:** turn a base model into an assistant for your task.
- **You'll build:** supervised fine-tuning with `trl`'s `SFTTrainer` (or MLX) on the M9 dataset, with loss masked to assistant turns only.
- **Concepts:** base vs instruct models, multi-turn formatting, completion-only loss, packing, system prompts, stop-token problems.
- **Checkpoint:** the model follows the output format on 90% or more of held-out prompts.

#### M11 · Evaluation you can trust
- **Goal:** stop guessing whether your fine-tune got better.
- **You'll build:** an eval harness with exact-match / regex metrics, an LLM-as-judge using a local model, a regression suite for general capability (to catch forgetting), and a comparison report across runs.
- **Concepts:** perplexity vs task metrics, judge bias, sample sizes and noise, experiment tracking (MLflow, W&B, or plain JSON logs).
- **Checkpoint:** a single command produces a report comparing the base, M7, M8 and M10 models.

### Part E — Alignment

#### M12 · Preference tuning: DPO & ORPO
- **Goal:** teach the model *which* answer is better, not just *an* answer.
- **You'll build:** a preference dataset of (prompt, chosen, rejected) triples; implement the DPO loss by hand on a tiny model, then train with `trl`'s `DPOTrainer`.
- **Concepts:** RLHF in brief, reward models, the DPO objective, β, reference models, ORPO as a single-stage alternative.
- **Checkpoint:** your handwritten DPO loss matches the library's to within tolerance, and the win rate against the SFT model is above 50% on held-out prompts.

#### M13 · RL with verifiable rewards (GRPO)
- **Goal:** improve reasoning on tasks where correctness can be checked.
- **You'll build:** GRPO on a small model (≤ 1.5B) for arithmetic or simple logic puzzles, with a programmatic reward function.
- **Concepts:** policy optimization, group-relative advantages, reward hacking, KL penalties, why RL is expensive.
- **Checkpoint:** accuracy on the held-out problem set improves measurably over the SFT baseline.

### Part F — Shipping

#### M14 · Merge, quantize & export
- **Goal:** turn adapters into deployable artifacts.
- **You'll build:** merge LoRA into the base weights (`mlx_lm.fuse` / `merge_and_unload`), convert to GGUF, quantize (Q4_K_M, Q8_0), and compare quality and size.
- **Concepts:** merged vs separate adapters, GGUF, quantization formats, model cards, licensing for redistribution.
- **Checkpoint:** the GGUF model runs in llama.cpp and passes the M11 regression suite with little degradation.
- **Stretch:** model merging (linear, TIES, SLERP) of two adapters.

#### M15 · Serve & integrate
- **Goal:** use your model like a real product.
- **You'll build:** serve the model through Ollama or `mlx_lm.server` behind an OpenAI-compatible API, plus a small chat UI or CLI that calls it.
- **Concepts:** throughput vs latency, context length limits, prompt caching, when to fine-tune vs when to use RAG or prompting.
- **Checkpoint:** an end-to-end smoke test hits the local API and validates its responses.

### Capstone

#### M16 · Build a domain specialist end to end
Pick one track, or propose your own:
1. **Support bot:** answers in a company's voice from its docs (SFT + DPO).
2. **Structured extractor:** reliable JSON output from messy text (SFT + format eval).
3. **Code helper:** a small model that's good at one library or language (SFT + unit-test reward via GRPO).
4. **Personal writing style:** a model that writes like you (data curation focus).

**Deliverables:** dataset card, training config, eval report against the base model, exported GGUF, a short write-up of what worked and what didn't.

---

## 5. How each module is laid out

```
modules/
  m06_lora_from_scratch/
    README.md          # concept explainer + learning objectives
    lesson.ipynb       # guided, runnable walkthrough
    exercises.ipynb    # TODO cells for you to complete
    solutions/         # reference solutions (try first!)
    src/               # reusable code the lesson builds up
    tests/
      test_checkpoint.py   # the auto-graded checkpoint
    config.yaml        # model names, hyperparams, memory notes
```

**What makes it interactive:**
- **Checkpoint tests:** `make check M=06` runs that module's pytest suite against *your* exercise code and gives specific hints when a test fails.
- **Progress tracker:** `make progress` shows which modules have passed.
- **Memory meter:** a shared helper prints peak memory and tokens/sec for every training run, so you build intuition for the hardware.
- **Predict-then-run cells:** before key cells, the notebook asks you to predict the loss, memory or output, then shows the actual result.
- **Tiny-first defaults:** every config has a `--tiny` flag that finishes in under 2 minutes for fast iteration.

---

## 6. Proposed repo layout

```
finetuning-llm/
  README.md              # quick start + course pitch
  COURSE.md              # this document
  pyproject.toml         # uv-managed deps
  Makefile               # doctor, check, progress, clean
  common/                # shared utils: memory meter, data loaders, eval harness
  data/                  # small bundled datasets (+ download scripts for larger ones)
  modules/
    m00_setup/
    m01_pytorch_essentials/
    ...
    m16_capstone/
  scripts/
    doctor.py
    progress.py
  outputs/               # gitignored: checkpoints, adapters, logs
```

---

## 7. Build order (suggested roadmap for the repo)

1. **Skeleton:** `pyproject.toml`, `Makefile`, `doctor.py`, `common/memory.py`, the M0 checkpoint.
2. **Core path:** M2 → M4 → M5 → M6 → M7. This is the minimum viable course and teaches the essence.
3. **Scale and data:** M8, M9, M10, M11.
4. **Advanced:** M12, M13.
5. **Shipping:** M14, M15, then capstone templates.
6. **Polish:** M1 and M3 deep dives, stretch goals, diagrams, troubleshooting FAQ.

---

## 8. Common pitfalls to cover (troubleshooting appendix)

- Chat template mismatch between training and inference
- Forgetting to mask prompt tokens, so the model learns to echo the prompt
- EOS token missing from training data, so generation never stops
- MPS operations that silently fall back to the CPU (`PYTORCH_ENABLE_MPS_FALLBACK`)
- Memory pressure and swap ruining throughput (watch the memory pressure graph, not just "used")
- Learning rate too high for LoRA, so loss spikes and the model collapses
- Evaluating on data that leaked into training
