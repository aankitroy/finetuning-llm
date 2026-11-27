# M7 · LoRA with real tools (MLX & PEFT)

**Builds on M6.** You built LoRA by hand: a frozen layer plus `scale · B @ A`, rank 8 on the attention projections. Here you use the two tools people actually train with, and see that each one is doing exactly what you built. Every new idea is explained where it first appears, and every term is in the [glossary](#glossary) below.

**Goal:** train the M5 plain-English task on a model three times bigger, **Qwen2.5-1.5B-Instruct**, twice: once with Apple's **mlx-lm** and once with Hugging Face's **PEFT**. Same adapter, same batches, same steps. Compare their speed, memory and quality side by side, save each adapter and load it back.

**Time:** about 3 hours.

**Download:** [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) (about 3 GB, Apache-2.0 licence), once; both tools read the same copy. `mlx-lm` and `peft` come with the course's packages (`uv sync`). The checkpoint's exercise tests use tiny random models and need no download.

**Memory:** on the reference Mac, mlx-lm peaked at about 9 GB and PEFT at about 13 GB. Each tool runs in its own process, one after the other. The lesson notebook holds both tools' models at once and peaks at about 16 GB. Close other big apps, and close the lesson notebook (or restart its kernel) before you run `train.py`.

**Mac only for mlx-lm:** MLX runs only on Apple-silicon Macs. Elsewhere `train.py` trains with PEFT only, and the mlx parts of the checkpoint are skipped.

**Data:** M5's [`data/`](../m05_full_finetune/data), read in place: the same 400 training pairs, 50 validation and 50 test pairs.

## Steps

0. **Watch the video first.** Open the dashboard (`make progress`) and press **Play the video** on M7's first step: a 4-minute narrated animation of the whole module, with captions.

1. **Read and run the lesson.**
   ```bash
   uv run jupyter lab modules/m07_lora_tools/lesson.ipynb
   ```

   | § | Section | You learn |
   |---|---|---|
   | 0 | Why use a tool? | what M6 left out: saving, speed, bigger models |
   | 1 | Two tools | what mlx-lm and PEFT are, and where each runs |
   | 2 | The data format | chat records, the files mlx-lm reads |
   | 3 | LoRA in mlx-lm | its config (rank, scale, keys) and its LoRA layer next to yours |
   | 4 | LoRA in PEFT | its LoraConfig (r, lora_alpha) and its LoRA layer next to yours |
   | 5 | A short run with each | the same batches through both, and two ways to count memory |
   | 6 | Tokens per second | the metric to watch, and why padding fools it |
   | 7 | Sequence length and memory | what grows when sequences get longer |
   | 8 | Saving and loading an adapter | what's in the files, how big, and loading them back |
   | 9 | Your task | the exercises and the run |

2. **Do the exercises** in [`exercises.py`](exercises.py). Each one says in plain words what it is and lists numbered steps as comments.

   | # | You write | What it teaches | Lesson § |
   |---|---|---|---|
   | 1 | `to_chat_record(pair, system)` | the chat format the tools read | 2 |
   | 2 | `mlx_lora_config(r, alpha, targets, num_layers)` | mlx-lm's adapter settings: it wants the scale | 3 |
   | 3 | `peft_lora_config(r, alpha, targets)` | PEFT's adapter settings: it wants alpha | 4 |
   | 4 | `tokens_per_sec(attention_masks, seconds)` | counting real tokens, not padding | 6 |
   | 5 | `load_peft_adapter(base_model, adapter_dir)` | loading a saved PEFT adapter | 8 |
   | 6 | `load_mlx_adapter(model_path, adapter_dir)` | loading a saved mlx-lm adapter | 8 |

   Check your progress any time with `make check M=07`. The exercise tests use tiny models and run in seconds.

3. **Answer the understanding checks** in [`answers.toml`](answers.toml): six questions on the scale, file sizes, tokens/sec, sequence length, memory and keeping adapters separate. Predict first, then check yourself in the notebook.

4. **Train with both tools:**
   ```bash
   uv run python modules/m07_lora_tools/train.py
   ```
   It turns the data into chat records with your exercise 1, builds the batches once with your M5 `build_example` and `collate`, and gives the same batches to both tools: rank 8, alpha 16, on `q_proj`, `k_proj`, `v_proj` and `o_proj` in all 28 blocks (0.141% of the weights), 2 epochs (100 steps) at learning rate 2e-4. Each tool trains in its own process (about 90 seconds of training each on the reference Mac), saves its adapter (about 8.7 MB), throws the model away, loads the adapter back onto a fresh model with your exercises 5 and 6, and checks it gives the same replies. Then it prints the two runs side by side. About 8 minutes in all. `--tool mlx` or `--tool peft` runs just one.

5. **Pass the checkpoint:** run `make check M=07`. M7 turns green when every exercise passes, your answers are right, and each tool's latest run, with your current code (re-run `train.py` after you change `exercises.py`):
   - reached a held-out loss of at most **0.8** (about 0.68 in the reference runs; 1.16 before training)
   - rewrote the test set at least **3** chrF points closer to the reference rewrites than the untuned model did (about +7 to +9 in the reference runs)
   - loaded back from disk and gave the same reply for at least **95%** of the 60 fixed prompts (all 60 in the reference runs)
   - peaked under **18 GB**

   The general-questions check from M5 is reported, not graded.

## What the reference runs measured

| | mlx-lm | PEFT |
|---|---|---|
| held-out loss | 1.159 → 0.682 | 1.163 → 0.685 |
| chrF against the reference rewrites | 54.8 → 63.8 | 55.1 → 61.9 |
| general questions right | 10/10 | 10/10 |
| training time | 94 s | 86 s |
| tokens/sec (real tokens) | 605 | 658 |
| peak memory (macOS) | 9.3 GB | 12.8 GB |
| trainable parameters | 0.141% | 0.141% |
| adapter file | 8.7 MB | 8.8 MB |

## Glossary

| Term | In plain words | Lesson § |
|---|---|---|
| **mlx-lm** | Apple's library for running and fine-tuning language models with MLX, on the Mac's GPU. | 1 |
| **PEFT** | Hugging Face's library of parameter-efficient fine-tuning methods (LoRA among them), for PyTorch models. | 1 |
| **adapter config** | The settings that describe an adapter: rank, scale or alpha, dropout, which layers. Saved next to the adapter. | 3 |
| **keys** | mlx-lm's name for the target modules, written from inside a block, like `self_attn.q_proj`. | 3 |
| **chat record** | One training example as a list of messages, each with a role and its content. | 2 |
| **JSONL** | A text file with one JSON object per line. | 2 |
| **tokens per second** | How many real tokens training gets through each second: the speed to compare runs by. | 6 |
| **padding** | Filler tokens that make every sequence in a batch the same length. The loss ignores them (M3). | 6 |
| **sequence length** | How many tokens one example has. Longer sequences mean more activations to keep. | 7 |
| **bf16** | bfloat16: a 2-byte number format with float32's range and less precision. Both tools train in it here. | 5 |
| **adapter file** | The saved A and B for every wrapped layer: megabytes, next to a model of gigabytes. | 8 |

## Syntax at a glance

| Code | What it does | When you use it |
|---|---|---|
| `{"role": "user", "content": text}` | One chat message. | Building a chat record. |
| `json.dumps(record)` | A dict as one line of JSON text. | Writing JSONL. |
| `alpha / r` | The scale. | mlx-lm's `lora_parameters`. |
| `LoraConfig(r=..., lora_alpha=..., target_modules=[...])` | PEFT's adapter settings. | Exercise 3. |
| `get_peft_model(model, config)` | Wraps the target layers with LoRA and freezes the rest. | Section 4, train.py. |
| `model.print_trainable_parameters()` | PEFT's count of what trains. | Checking 0.141%. |
| `linear_to_lora_layers(model, num_layers, lora)` | mlx-lm's version of your M6 `inject_lora`. | Section 3, train.py. |
| `int(mask.sum())` | How many 1s are in a mask. | Counting real tokens. |
| `model.save_pretrained(folder)` | Saves a PEFT adapter (not the base model). | Section 8. |
| `PeftModel.from_pretrained(base, folder)` | Loads a PEFT adapter onto a base model. | Exercise 5. |
| `load(name, adapter_path=folder)` | Loads a model in mlx-lm, with an adapter on it. | Exercise 6. |

## Stretch

- Run mlx-lm's own command line on the files train.py wrote: put the output of your `mlx_lora_config` in a YAML file with `model`, `data: outputs/m07/data`, `train: true`, `iters: 100`, `batch_size: 8` and `learning_rate: 2e-4`, then `uv run mlx_lm.lora --config your.yaml`. How close is its loss to train.py's? (It sorts batches by length, so its tokens/sec will differ: why?)
- Try `--r 16` (with `--alpha 32`) and `--max-len 512`. Which changes the memory more?
- Fuse the mlx adapter into the weights with `uv run mlx_lm.fuse --model Qwen/Qwen2.5-1.5B-Instruct --adapter-path outputs/m07/mlx/adapter`, and check the fused model gives the same rewrites. That's M6's `merge_lora`, done for you.

## Stuck?

- `make hint M=07 EX=3` gives a nudge for exercise 3 (`EX=q2` for question 2).
- `make solution M=07 EX=3` shows the reference answer for just that exercise.

Both are logged on your dashboard. Try for a while first. To redo the module from scratch, run `make reset M=07`, which also puts the lesson notebook back to unrun.
