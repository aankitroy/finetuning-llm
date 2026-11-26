# M6 · LoRA from scratch — design

Approved in chat on 2026-11-26 ("go").

## Why
The learner finished M5: a full fine-tune of Qwen2.5-0.5B-Instruct that peaked at ~24 GB, 3.95 GB of
it AdamW's state, and that would need ~112 GB for a 7B model. COURSE.md M6: build a `LoRALinear`
layer by hand, inject it into the M5 model's attention projections, freeze the base and train.
Concepts: ΔW = BA, rank `r`, `alpha` scaling, which modules to target, trainable-parameter counts,
merging adapters back. Checkpoint: the layer equals the base layer at initialization, trainable
parameters are under 1%, and quality comes within a few percent of the M5 full fine-tune.

## Goals / non-goals
- Same beginner-first shape as M1–M5: narrated video first, concept-first lesson, a **What each line
  does** box after every code cell, bold terms, glossary + syntax table, numbered steps in every
  exercise, Predict cells.
- The learner writes LoRA itself: the layer, the parameter count, injection + freezing, counting
  trainable parameters, merging.
- Not PEFT / mlx-lm (M7), not quantization (M8), not saving/loading adapter files as a tool workflow
  (M7) — M6 saves the adapter only to show its size.

## Approaches
1. **Same task, data and model as M5; LoRA on the attention projections (chosen).** The only thing
   that changes is *how* the model is tuned, so the comparison with M5 (quality, memory, size) is
   fair, and no download is needed.
2. A new task: more variety, but the "within a few percent of M5" check would have nothing to compare
   with.
3. Build LoRA on the M4 TinyGPT: simpler, but tiny models don't show the memory win.

## Design
- **Model / data:** `Qwen/Qwen2.5-0.5B-Instruct` in fp32 (as M5), M5's `data/` read in place (not
  copied). The learner's own M5 `build_example`, `collate`, `masked_lm_loss`,
  `optimizer_state_bytes`, `chrf` and `count_buzzwords` build the batches and the scores (their code
  carries forward); the graded held-out loss uses M5's course-owned `heldout_loss`.
- **Exercises (`exercises.py`):**
  1. `class LoRALinear(nn.Module)`: `__init__(base, r, alpha)` stores the frozen `base`, `A` =
     `randn(r, in) / sqrt(in)` and `B` = `zeros(out, r)` as parameters on the base's device and
     dtype, `scale = alpha / r`; `forward(x) = base(x) + scale * (x @ A.T @ B.T)`.
  2. `lora_params(in_features, out_features, r) -> int` = `r * (in + out)`.
  3. `inject_lora(model, targets, r, alpha) -> list[str]`: freeze every parameter, then replace every
     `nn.Linear` whose own name is in `targets` with a `LoRALinear`; return the replaced names.
  4. `count_trainable(model) -> (trainable, total)`.
  5. `merge_lora(layer) -> nn.Linear`: a plain layer with `W + scale * B @ A` and the same bias.
- **Run:** `train.py` (one mode, the learner's own loop as in M5: AdamW on the trainable parameters
  only, warmup + cosine, clip 1.0). Options `--r`, `--alpha`, `--targets attn|all`, `--tiny`.
  After training it merges every layer with the learner's `merge_lora` and measures the held-out loss
  again (the merge must not change it), saves the adapter (`adapter.pt`, A and B only) to show its
  size, scores the 50 test rewrites and the 10 general questions like M5. Writes
  `outputs/m06/run.json` (latest run, graded) and `outputs/m06/sweep/r<r>_<targets>.json` (stretch).
- **Defaults:** r = 8, alpha = 16, targets q/k/v/o (1.08M of 494M = 0.22%). The learning rate and
  epochs are set after measuring (LoRA needs a far bigger step than M5's 1e-5).
- **Checkpoint:** five exercise tests (the layer equals the base at init; it computes
  base + scale·BA·x; injection keeps the outputs identical and freezes everything else; counts;
  merging gives the same outputs). Six answers. The run: trainable % < 1, held-out loss within a
  margin of the learner's own M5 loop run (`outputs/m05/run_loop.json`, or the reference 0.94 if
  absent), merge changes the held-out loss by < 0.01, full size, own code (sha256 of `exercises.py`).
  The margin is set after measuring.
- **Video:** narrated, Kokoro `af_heart`, step 1 of the module, same player as M1–M5.
- **Stretch:** sweep r ∈ {2, 8, 32} (and `--targets all`), plot held-out loss against parameter count.

## Measured (reference Mac, solutions, 2 epochs, batch 8, alpha = 2r)
| run | trainable | held-out loss | chrF | general | AdamW state | adapter |
|---|---|---|---|---|---|---|
| r 2, attn | 0.05% | 0.940 | 47.2 | 9/10 | 2.2 MB | 1.1 MB |
| **r 8, attn, lr 2e-4 (default)** | 0.22% | **0.903** | 50.9 | 10/10 | 8.7 MB | 4.4 MB |
| r 32, attn | 0.88% | 0.907 | 51.9 | 9/10 | 34.6 MB | 17.4 MB |
| r 8, all | 0.89% | 0.913 | 52.0 | 9/10 | 35.2 MB | 17.7 MB |
| r 8, lr 1e-4 | 0.22% | 0.925 | 49.3 | 10/10 | | |
| r 8, lr 5e-4 | 0.22% | 0.889 | 51.3 | 9/10 | | |
| r 8, 3 epochs | 0.22% | 0.921 | | | | |

Base model: 1.554, chrF 54.4. Training ~22 s; peak memory (same measure as M5) 19–21 GB vs M5's ~24 GB.
**Rulings:** default lr 2e-4 (best loss without losing a general question); `MARGIN` = 5% of the
learner's M5 loop loss, but never below the reference 0.94 (a learner whose M5 run was better,
e.g. with more epochs, compares against 0.94, so M6 is never harder because M5 went well);
chrF isn't graded (LoRA rewrites are shorter, chrF 50.9 vs base 54.4, while loss and buzzwords
improve — M5 already showed chrF penalises valid rewordings); the trainable share is graded from
train.py's own count, not the learner's `count_trainable`; the tiny run skips the < 1% check (a
tiny model is mostly attention).

## Left to the builder
Lesson wording and numbers (from real runs), video scenes, the exact margin and learning rate.
