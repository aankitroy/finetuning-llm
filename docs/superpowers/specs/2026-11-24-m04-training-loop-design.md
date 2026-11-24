# M4 · The training loop, from scratch — design

Approved in chat on 2026-11-24 (design and the Tiny Shakespeare download).

## Why
The learner finished M3 and wants M4 in the same beginner-first style. COURSE.md: "demystify
training completely"; build a character-level GPT trained from zero; checkpoint "validation loss
drops below the target threshold and samples look like plausible text". M5 reuses this exact loop
on a pretrained model, so the loop must be the learner's own.

## Goals / non-goals
- Concept-first lesson, a **What each line does** box after every code cell, bold terms,
  glossary + syntax table in the README (as M1–M3).
- The learner writes the training loop's pieces; the model is given (`model.py`), built from parts
  they know from M2 (embeddings, causal attention, MLP, layer norm).
- Not mixed precision (the stretch), not distributed training, not the HF `Trainer` (M5).

## Approaches
1. **Given model + learner-written loop, on Tiny Shakespeare, on MPS (chosen).** Short enough to
   run in a few minutes; samples visibly turn into Shakespeare-like text.
2. Learner also writes the model: repeats M2 and doubles the module's length.
3. Use a library loop (`Trainer`, Lightning): hides exactly what this module is meant to show.

## Design
- **Data:** `shakespeare.txt` — Tiny Shakespeare (1,115,394 bytes, 65 distinct characters,
  public-domain plays), copied once from karpathy/char-rnn
  (sha256 `86c4e6aa…c565ed`). Excluded from pre-commit's whitespace fixers (two lines end in a
  space). First 90% train, last 10% validation.
- **Model (`model.py`, given):** `TinyGPT(vocab_size, block_size=256, n_embd=256, n_head=4,
  n_layer=6, dropout=0.1)`, about 4.8M parameters; `forward(idx) -> logits (B, T, V)`;
  `generate(idx, max_new_tokens, temperature=0.8, generator=None)`.
- **Lesson sections:** 0 What training really does · 1 Characters as tokens, and the train/val
  split · 2 Batches of shifted windows · 3 The causal LM loss · 4 One training step · 5 Learning-rate
  schedules · 6 Gradient clipping · 7 Gradient accumulation · 8 Reading loss curves · 9 Your task.
- **Exercises:**
  1. `get_batch(data, block_size, batch_size, generator=None) -> (x, y)`: random windows; `y` is `x`
     shifted one to the left.
  2. `causal_lm_loss(logits, targets) -> Tensor`: cross-entropy over every position.
  3. `lr_at(step, max_lr, min_lr, warmup_steps, total_steps) -> float`: linear warmup
     `max_lr * (step + 1) / warmup_steps`, then cosine down to `min_lr`; `min_lr` from
     `total_steps` on.
  4. `clip_grad_norm(params, max_norm) -> float`: by hand; returns the norm before clipping;
     skips params with no grad; scales only when the norm is above `max_norm`.
  5. `train_step(model, batches, optimizer, max_norm) -> float`: zero grads, backward each
     micro-batch's loss divided by `len(batches)`, clip, step; returns the mean loss.
  6. `estimate_loss(model, data, block_size, batch_size, iters, generator=None) -> float`: eval mode,
     no gradients, average loss, back to train mode.
- **Questions (6):** loss before training (≈ ln 65 ≈ 4.17), what overfitting looks like, a learning
  rate that's too high, effective batch size with accumulation (number), why warmup, what clipping
  does to a gradient.
- **Run step:** `train.py` (MPS when available) trains with the learner's code: micro-batch 16 ×
  2 accumulation steps, 2000 steps, AdamW, max_lr 1e-3 → 1e-4, warmup 100, clip 1.0. Measured on the
  reference Mac with the reference loop: about 106 s, val loss 1.53, 96% real words, peak memory 1.25 GB (bf16: ~25% faster). Writes
  `outputs/m04/run_summary.json` (`val_loss`, `train_loss`, `initial_val_loss`, `real_word_rate`,
  `peak_mem_gb`, `steps`, `params`, `device`, `seconds`, `tiny`, `used_solutions`, `history`),
  `loss_curve.png` and `samples.txt`. `--tiny` (CPU, small model, seconds) is for tests/CI.
- **Checkpoint:** six exercise tests, six answers, val loss < **1.70**, real-word rate ≥ **0.80**
  (share of words in the sample that occur in the training text; raised from the 0.60 floated in
  chat after measuring: val 2.15 → 39%, 1.79 → 75%, 1.53 → 96%, so 0.80 tracks the loss target), a full-size run, with the
  learner's own code. Metrics: `final_loss` and `peak_mem_gb` (both lower is better).
- **Stretch:** `--precision bf16` (autocast), comparing time and peak memory.
