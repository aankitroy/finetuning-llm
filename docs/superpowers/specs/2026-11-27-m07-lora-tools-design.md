# M7 · LoRA with real tools (MLX & PEFT) — design

Model chosen in chat on 2026-11-27 (Qwen2.5-1.5B-Instruct, "I will go with your recommendation").

## Why
The learner built LoRA by hand in M6 (0.5B, rank 8 on q/k/v/o, held-out loss 0.903, 0.22% trainable).
COURSE.md M7: run LoRA at a realistic scale with production tooling — the same task on a 1B–4B model,
once with `mlx_lm.lora` and once with `peft`, compared side by side on speed and memory. Concepts:
adapter configs, saving and loading adapters, sequence length vs memory, tokens/sec as the metric to
watch. Checkpoint: an adapter that loads and produces the expected style on a fixed prompt set. It
sets up M8 (QLoRA on a 7B model in MLX), so the learner must leave M7 comfortable with mlx-lm.

## Goals / non-goals / anti-goals
- Goals: same beginner-first shape as M1–M6 (narrated video first, concept-first lesson, a **What each
  line does** box after every code cell, bold terms, glossary + syntax table, numbered steps and "In
  plain words" in every exercise, Predict cells, hints, `make solution`). The learner sees that the
  tools do exactly what they built in M6 (same A, B, rank, scale), configures both tools, measures
  them fairly, saves an adapter and loads it back.
- Non-goals: quantization/QLoRA (M8), new data (M9), trl/SFTTrainer (M10), serving (M15).
- Anti-goals: a module where the learner only runs commands and learns nothing (the tools must be
  connected back to M6's layer); a comparison that's unfair (different data, steps, rank or sequence
  length between tools); runs that need more than ~20 GB or more than ~10 minutes each.

## Constraints
- Mac M5 Max, 36 GB unified memory, shared with the learner's other apps (Cursor, Chrome, OrbStack ≈
  20 GB). Keep each run's peak under ~18 GB.
- Downloads need the user's explicit permission: `mlx-lm` and `peft` (added to `pyproject.toml`, only
  the `mlx-lm` extra on macOS arm64) and `Qwen/Qwen2.5-1.5B-Instruct` (~3.1 GB, Apache-2.0).
- CI runs on Ubuntu and macOS without the model: everything graded on CI uses tiny models or is
  skipped, as in M5/M6. mlx doesn't exist on Linux, so mlx-specific tests skip there.
- Never touch the learner's `exercises.py`, `answers.toml`, `lesson.ipynb`, `.progress`, `outputs`.

## Approaches
1. **Course script drives both tools through their Python APIs; the learner writes the parts that
   matter (chosen).** The learner writes the data conversion, both configs, the tokens/sec measure and
   the two adapter loaders; `train.py` calls `mlx_lm` (its LoRA trainer) and PEFT + the learner's M5
   loop. Fair by construction (one script sets the same data, steps, rank, length for both) and the
   learner still touches every concept.
2. Learner runs the `mlx_lm.lora` CLI with a YAML they write, and a PEFT script. Most realistic, but
   timing and memory are measured differently by each tool, so the side-by-side isn't fair, and the
   checkpoint can't tell whose code ran. Kept as a stretch (run the CLI with the learner's own config).
3. Only one tool. Fails COURSE.md, and M8 needs MLX while M10 needs the HF stack.

## Design
- **Model / data:** `Qwen/Qwen2.5-1.5B-Instruct`, the one HF download used by both tools (mlx-lm
  loads HF safetensors directly). M5's `data/` read in place (400 train / 50 val / 50 test, plus
  `general.jsonl` and `buzzwords.txt`), same system prompt, same chat template. Scored with M5's
  course-owned scorer (`score`) and the learner's M5 `chrf`/`count_buzzwords`, like M6.
- **Same settings for both tools:** LoRA rank 8, alpha 16 (scale 2.0), on q/k/v/o in all 28 layers,
  dropout 0, batch 8, the same number of optimizer steps, the same learning rate and max sequence
  length, loss only on the answer tokens. Fair by construction: `train.py` builds the padded batches once (M5's
  `build_example`/`collate`, the learner's code) in a fixed order and feeds the same batches to both
  tools — to mlx-lm through its custom batch-iterator hook rather than its own length-sorted batching —
  so tokens/sec (exercise 4, from those batches' attention masks) means the same thing for both.
  (Builder's call.) The learning rate, steps and dtype are set after measuring.
  Preference: bf16 for both, if PEFT trains stably in bf16 on MPS; otherwise PEFT in fp32 and the
  lesson says so (and the README explains that part of the speed/memory gap is the dtype).
- **Exercises (`exercises.py`):**
  1. `to_chat_record(pair, system) -> dict`: one M5 pair as mlx-lm's chat format,
     `{"messages": [{"role":"system",…},{"role":"user",…},{"role":"assistant",…}]}`.
  2. `mlx_lora_config(r, alpha, targets, num_layers) -> dict`: the LoRA part of an mlx-lm config —
     `{"num_layers": num_layers, "lora_parameters": {"rank": r, "scale": alpha / r, "dropout": 0.0,
     "keys": [...]}}` with keys as mlx-lm names them (`self_attn.q_proj`, …). Teaches that mlx-lm
     wants the *scale*, not alpha (M6 §3). (Builder's call: `num_layers` sits next to
     `lora_parameters`, as in mlx-lm's own YAML, so the CLI stretch can use the dict as it is.)
  3. `peft_lora_config(r, alpha, targets) -> LoraConfig`: PEFT's config (`r`, `lora_alpha`,
     `target_modules`, `lora_dropout=0`, `task_type="CAUSAL_LM"`). Teaches that PEFT wants *alpha*.
  4. `tokens_per_sec(attention_masks, seconds) -> float`: real (non-pad) tokens processed per second
     — the metric to watch, and why padding inflates a naive count.
  5. `load_peft_adapter(base_model, adapter_dir)` and `load_mlx_adapter(model_name, adapter_dir)`:
     load a saved adapter back on a fresh base (`PeftModel.from_pretrained`; `mlx_lm.load(…,
     adapter_path=…)`). Five exercise numbers, six functions; `make solution EX=5` shows both.
- **Run:** `train.py --tool mlx|peft|both` (default both, mlx first), `--tiny` for CI and the smoke
  test. Each tool: writes the converted JSONL (learner's exercise 1) to `outputs/m07/data/`, trains,
  saves its adapter (`outputs/m07/{mlx,peft}/adapter/`), measures held-out loss before/after, peak
  memory (same measure for both: process peak from the OS, plus each framework's own number for the
  lesson), tokens/sec with the learner's exercise 4, trainable %; then **reloads the adapter from disk
  with the learner's loader in a fresh model** and generates greedily on the fixed prompt set. Writes
  `outputs/m07/{mlx,peft}/run.json` and prints a side-by-side table (`outputs/m07/compare.json`).
- **Fixed prompt set:** the 50 M5 test pairs plus M5's 10 general questions. "Expected style" = the
  reloaded adapter's rewrites use fewer buzzwords than the base 1.5B model's and score held-out loss
  well below base; plus the reloaded outputs must equal the outputs the trained model gave before saving
  (proves save/load worked).
- **Checkpoint:** exercise tests on tiny models (mlx ones skip off macOS); six understanding questions
  (scale vs alpha, adapter size on disk, tokens/sec vs padding, sequence length vs memory, why the two
  tools' numbers differ, merging vs keeping the adapter separate); the run, for each tool: full size,
  own code (sha256 of `exercises.py`), held-out loss below a bar set after measuring, buzzwords fewer
  than base, reload matches. Metrics on the dashboard: `final_loss`, `peak_mem_gb`, `tokens_per_sec`,
  `trainable_pct` (from the mlx run, the one M8 builds on; both shown in the lesson table).
- **Lesson sections:** 0 why tools (what M6 left out: saving, speed, bigger models); 1 the two tools
  and what each is; 2 the data format; 3 the mlx-lm config (rank, scale, keys) and its LoRA layer next
  to M6's; 4 the PEFT config and its LoRA layer next to M6's; 5 training a small run each way;
  6 tokens/sec and padding; 7 sequence length vs memory (measured at 128/256/512/1024); 8 saving and
  loading an adapter (what's in the files, how big); 9 your task.
- **Video:** narrated, Kokoro `af_heart`, step 1 of the module, same player as M1–M6, about 4 minutes.
- **Stretch:** run the `mlx_lm.lora` CLI with a YAML built from the learner's config; try rank 16 or
  `--max-seq-length 512`; fuse the mlx adapter (`mlx_lm.fuse`) and compare outputs.

## Left to the builder
Learning rate, steps, dtype, the loss bar and margins (from real runs); lesson wording and numbers;
video scenes; exact question wording; how peak memory is read for each tool.
