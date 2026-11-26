# M6 · LoRA from scratch

**Builds on M5.** You'll reuse its model, its data and your own M5 exercises: `train.py` builds its batches and computes its loss and scores with your M5 code. Every new idea is explained where it first appears, and every term is in the [glossary](#glossary) below.

**Goal:** build **LoRA** by hand, the trick almost every fine-tune on a laptop uses. You'll write a layer that keeps the original weights frozen and trains a small add-on next to them, put it inside Qwen, and teach the M5 house style with 0.22% of the weights, about as well as M5's full fine-tune did.

**Time:** about 3 hours.

**Download:** none. The lesson and `train.py` use the [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) model you downloaded in M5. The checkpoint's exercise tests use a tiny random model.

**Memory:** `train.py` peaked at about 20 GB on the reference Mac (M5's full fine-tune: about 24 GB). Close other big apps, and close the lesson notebook (or restart its kernel) before you run it. With less memory, add `--batch 4`.

**Data:** M5's [`data/`](../m05_full_finetune/data), read in place: the same 400 training pairs, 50 validation and 50 test pairs, so the comparison with M5 is fair.

## Steps

0. **Watch the video first.** Open the dashboard (`make progress`) and press **Play the video** on M6's first step: a 4-minute narrated animation of the whole module, with captions.

1. **Read and run the lesson.**
   ```bash
   uv run jupyter lab modules/m06_lora_from_scratch/lesson.ipynb
   ```

   | § | Section | You learn |
   |---|---|---|
   | 0 | Do we need to change every weight? | the update ΔW, and freezing the original weights |
   | 1 | Low rank: a big matrix from two thin ones | rank, B @ A, and what it saves |
   | 2 | A LoRA layer, by hand | why B starts at zero and A starts random |
   | 3 | The scale: alpha / r | the volume knob on the update |
   | 4 | Putting LoRA inside Qwen | target modules, swapping layers, counting what trains |
   | 5 | Training only the adapters | the real run, and where the memory still goes |
   | 6 | Merging the adapter back | folding B @ A into the weights, and the adapter's size |
   | 7 | Which layers, and how big a rank? | the reference sweep, and forgetting |
   | 8 | Your task | the exercises and the run |

2. **Do the exercises** in [`exercises.py`](exercises.py). Each one says in plain words what it is and lists numbered steps as comments.

   | # | You write | What it teaches | Lesson § |
   |---|---|---|---|
   | 1 | `LoRALinear(base, r, alpha)` | the LoRA layer: frozen base + scale · B @ A | 2 |
   | 2 | `lora_params(in_features, out_features, r)` | what an adapter costs | 1 |
   | 3 | `inject_lora(model, targets, r, alpha)` | freezing a model and swapping its layers | 4 |
   | 4 | `count_trainable(model)` | trainable vs total parameters | 4 |
   | 5 | `merge_lora(layer)` | folding the adapter back into one plain layer | 6 |

   Check your progress any time with `make check M=06`. The exercise tests use a tiny model and run in seconds.

3. **Answer the understanding checks** in [`answers.toml`](answers.toml): six questions on counting, starting values, the scale, memory and merging. Predict first, then check yourself in the notebook.

4. **Train your adapter:**
   ```bash
   uv run python modules/m06_lora_from_scratch/train.py
   ```
   It puts your `LoRALinear` on `q_proj`, `k_proj`, `v_proj` and `o_proj` in all 24 layers (rank 8, alpha 16), trains only those for 2 epochs at learning rate 2e-4 (about 22 seconds of training on the reference Mac), merges them into the weights with your `merge_lora`, and scores the merged model like M5 did. It also saves the adapter on its own (`outputs/m06/adapter.pt`, about 4.4 MB) and writes `outputs/m06/run.json`. About 3 minutes in all, the first time (it scores the original model once, then caches that).

5. **Pass the checkpoint:** run `make check M=06`. M6 turns green when every exercise passes, your answers are right, and your latest run, with your current code (re-run `train.py` after you change `exercises.py`):
   - trained under **1%** of the weights (0.22% with the defaults)
   - gave the same held-out loss after merging as before, within **0.01**
   - reached a held-out loss within **5%** of your M5 loop run's (about 0.90 against 0.94 in the reference runs). If your M5 run did better than the reference 0.94, the reference is used instead, so a strong M5 run never makes M6 harder. If you have no M5 loop run, or it didn't pass M5's own bar, the reference is used too.

   The general-questions check from M5 is reported, not graded.

## Glossary

| Term | In plain words | Lesson § |
|---|---|---|
| **LoRA** | Low-rank adaptation: freeze the model and train a small low-rank update next to some of its layers. | intro |
| **adapter** | The small add-on LoRA trains: the A and B matrices for every target layer. | intro |
| **update** | The change fine-tuning makes to a weight matrix: new W = W + ΔW. | 0 |
| **frozen** | A weight with `requires_grad` off: it's used, but never trained. | 0 |
| **low-rank** | A matrix built from a few column-times-row pieces, so it takes far fewer numbers to describe. | 1 |
| **rank** | How many of those pieces: `r`. B is out x r and A is r x in. | 1 |
| **alpha** | The number you choose that sets the scale, `alpha / r`, of the update. | 3 |
| **target modules** | The layers that get an adapter. Here, the four attention projections. | 4 |
| **trainable parameters** | The numbers the optimizer will change: those with `requires_grad` on. | 4 |
| **activations** | The in-between results of the forward pass, kept so `.backward()` can use them (M5). | 5 |
| **merging** | Folding `scale · B @ A` into W, so the model is a plain one again. | 6 |

## Syntax at a glance

| Code | What it does | When you use it |
|---|---|---|
| `nn.Parameter(t)` | Marks a tensor as a weight to train. | A and B. |
| `p.requires_grad_(False)` | Freezes a parameter. | The base layer; the whole model. |
| `torch.zeros(out, r)`, `torch.randn(r, in)` | A tensor of zeros / of random numbers. | B starts at zero, A starts random. |
| `device=w.device, dtype=w.dtype` | Make a tensor where, and in the format, another lives. | A and B next to the base weight. |
| `x @ A.T @ B.T` | x through A, then through B. | The LoRA update, without building B @ A. |
| `model.named_modules()` | Every layer, with its dotted name. | Finding the target layers. |
| `name.rpartition(".")` | Splits at the last dot: (before, ".", after). | The parent's name and the layer's own name. |
| `model.get_submodule(name)` | The layer with that dotted name. | Finding the parent. |
| `setattr(obj, name, value)` | `obj.name = value`, with the name as a string. | Putting the LoRA layer in place. |
| `isinstance(m, nn.Linear)` | `True` if `m` is a linear layer. | Only wrapping plain linear layers, once. |
| `sum(p.numel() for p in model.parameters() if p.requires_grad)` | How many numbers train. | Counting trainable parameters. |
| `with torch.no_grad():` | Runs without recording gradients. | Changing weights by hand. |
| `.copy_(t)` | Fills a tensor with `t`'s numbers, in place. | The merged layer's weight and bias. |

## Stretch

- Run `train.py --r 2`, `--r 32` and `--targets all`. Every run is saved in `outputs/m06/sweep/`. Plot held-out loss against trainable parameters (matplotlib, from M1). Does yours match the table in lesson section 7?
- Try `--lr 5e-4` (the reference run reached 0.889) and `--epochs 3`. Which would you ship, given the general-questions score?
- Load `outputs/m06/adapter.pt` into a fresh copy of the model with your `inject_lora`, copying each A and B in, and check that you get the same rewrites. That's what M7's tools do for you.

## Stuck?

- `make hint M=06 EX=3` gives a nudge for exercise 3 (`EX=q2` for question 2).
- `make solution M=06 EX=3` shows the reference answer for just that exercise.

Both are logged on your dashboard. Try for a while first. To redo the module from scratch, run `make reset M=06`, which also puts the lesson notebook back to unrun.
