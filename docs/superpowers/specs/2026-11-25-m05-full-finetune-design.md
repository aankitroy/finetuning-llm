# M5 · Your first real fine-tune (full fine-tune) — design

Approved in chat on 2026-11-25 (design, the style task, and the Qwen2.5-0.5B-Instruct download).

## Why
The learner finished M4 (their own training loop on a model trained from zero). COURSE.md M5:
fine-tune every weight of a ~0.5B pretrained model on a narrow style task, once with your own loop
and once with the HF `Trainer`; concepts are loss masking on prompt tokens, catastrophic
forgetting and the memory cost of optimizer state; checkpoint "beats the base model on a held-out
style test, and you can report the peak memory you measured". It should leave the learner seeing
why full fine-tuning doesn't scale on a laptop (the motivation for Part C).

## Goals / non-goals
- Same beginner-first shape as M1–M4: concept-first lesson, a **What each line does** box after
  every code cell, bold terms, glossary + syntax table, numbered steps in every exercise.
- The learner writes the pieces that make fine-tuning differ from M4: masking, padding/collation,
  the shifted masked loss, measuring optimizer memory, and the two scores.
- Not LoRA (M6), not dataset curation (M9), not LLM-judge evaluation (M11).

## Approaches (style task)
1. **Jargon → plain English, ~500 original pairs, scored by chrF + buzzword count (chosen).** The
   most realistic use of fine-tuning (a house style from examples), visibly better after tuning.
2. Rule-made style (telegram style): exact scoring, but a few lines of code could do it, so it
   doesn't show why you'd fine-tune.
3. A public parallel corpus (Shakespeare → modern): another download, unclear licences.

## Design
- **Model:** `Qwen/Qwen2.5-0.5B-Instruct` (494M params, tied embeddings, Apache-2.0, ~1 GB). ChatML
  template, as in M3. Measured on the reference Mac: a full fp32 AdamW step on 8 x 96 tokens takes
  ~0.95 s and ~13.3 GB. With a "rewrite in plain English" system prompt, the base model rewrites but
  keeps jargon ("capitalize on our complementary strengths", "bandwidth", "deliverables").
- **Data (`data/`):** `train.jsonl` 400, `val.jsonl` 50, `test.jsonl` 50 lines of
  `{"jargon", "plain"}`, original, CC BY 4.0; `buzzwords.txt`; `general.jsonl` — 10 ordinary
  questions with an expected substring, for the forgetting check.
- **Prompt:** system "Rewrite the user's text in plain English. Reply with the rewrite only.",
  user = jargon, assistant = plain. The base model is evaluated with the same prompt.
- **Exercises:**
  1. `build_example(prompt_ids, answer_ids, max_len=None) -> (input_ids, labels)`: labels are -100
     on the prompt; truncation keeps the start.
  2. `collate(examples, pad_id) -> dict`: right-pad to the longest; `attention_mask` 1/0; padded
     labels -100; long tensors.
  3. `masked_lm_loss(logits, labels) -> Tensor`: shift by one, cross-entropy with
     `ignore_index=-100`; equals `model(input_ids, labels=labels).loss`.
  4. `optimizer_state_bytes(optimizer) -> int`: bytes in every tensor of `optimizer.state`.
  5. `chrf(hypothesis, reference, max_n=6, beta=2.0) -> float` (0–100): spaces removed;
     for n = 1..max_n, n-gram precision and recall from clipped counts; average P and R over the
     orders where both texts have n-grams; F-beta. (Close to, not identical with, sacrebleu.)
  6. `count_buzzwords(text, buzzwords) -> int`: lowercase, non-letters → spaces, count each phrase
     as whole words.
- **Run:** `train.py --loop` (the learner's pieces in an M4-style loop: AdamW, warmup + cosine,
  clip 1.0) and `train.py --trainer` (HF `Trainer`, `data_collator` = the learner's `collate`).
  Then greedy generation on the 50 test inputs for base (cached) and tuned; mean chrF, buzzwords per
  output, the forgetting check, peak memory, tokens/s, optimizer-state GB. `--tiny`: a tiny random
  Qwen2 model on the CPU, seconds, no download. Writes `outputs/m05/run_<loop|trainer>.json`.
- **Checkpoint:** six exercise tests; six answers; for both runs: held-out loss on the reference
  rewrites falls by ≥ 0.4 from the base model's, buzzwords per rewrite ≤ base, and chrF ≥ base − 3;
  full-size runs with own code. *Review fixes:* the graded held-out loss is measured by train.py
  with course-owned masking (`heldout_before` / `heldout_after`), and each run stores a sha256 of
  the exercises it used, which must match the current `exercises.py`. *Ruling after measuring (replaces "chrF ≥ base + 10, buzzwords ≤ half
  of base"):* with the rewrite system prompt the base model already scores chrF 54.4 and 0.16
  buzzwords per rewrite, and tuned runs score chrF 53–58 although their rewrites are clearly in the
  house style; chrF rewards one reference wording among many valid ones. The held-out loss
  (1.55 → 0.84–0.94) measures the style directly and is deterministic.
- **Defaults (measured):** 2 epochs, lr 1e-5, batch 8: val loss 0.94, chrF 56.2, 0 buzzwords,
  10/10 general questions, ~40 s training, peak ~24 GB (AdamW state 3.95 GB). 3 epochs at 2e-5
  overfits (val 0.89 → 1.20) and forgets (7/10): used in the lesson and the stretch.
  Forgetting is reported (dashboard + summary), not graded. Metrics: `final_loss`, `peak_mem_gb`,
  `tokens_per_sec`, `trainable_pct` (100 here; M6 makes it < 1).
- **New dependency:** `accelerate` (required by `Trainer`).
