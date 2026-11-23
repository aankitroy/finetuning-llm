# M4 · The training loop, from scratch

**Builds on M1 to M3.** If you finished them, you know everything this module assumes: the four-beat training loop (M1), logits, softmax and causal attention (M2), and tokens (M3). Every new idea is explained where it first appears, and every term is in the [glossary](#glossary) below.

**Goal:** take "training" apart completely. You'll write every piece of a real training loop yourself, then use it to train a small GPT from random weights until it writes Shakespeare-like text. In M5 you'll run the same loop on a real pretrained model.

**Time:** about 3.5 hours.

**Data:** [`shakespeare.txt`](shakespeare.txt) is Tiny Shakespeare (1.1 MB: 40,000 lines from several of Shakespeare's plays, which are in the public domain), from [karpathy/char-rnn](https://github.com/karpathy/char-rnn). It's already in this folder; nothing to download.

**The model** is in [`model.py`](model.py): a 4.8M-parameter GPT that reads one character at a time, built from the parts you met in M2. It's given to you. This module is about training it.

## Steps

1. **Read and run the lesson.**
   ```bash
   uv run jupyter lab modules/m04_training_loop/lesson.ipynb
   ```

   | § | Section | You learn |
   |---|---|---|
   | 0 | What training really does | the four beats again, and an untrained model's gibberish |
   | 1 | Characters as tokens, and the train/validation split | a 65-character vocabulary, and why you keep text back |
   | 2 | Batches of shifted windows | how one window of text holds many next-character examples |
   | 3 | The causal LM loss | cross-entropy over every position, and why it starts near ln 65 |
   | 4 | One training step | the optimizer, and the loss falling for real |
   | 5 | Learning-rate schedules | too small, too big, and warmup with cosine decay |
   | 6 | Gradient clipping | the gradient norm, and capping it without changing direction |
   | 7 | Gradient accumulation | a big batch's gradients with a small batch's memory |
   | 8 | Reading loss curves | measuring honestly, and what overfitting looks like |
   | 9 | Your task | the exercises and the run |

2. **Do the exercises** in [`exercises.py`](exercises.py). Each one says in plain words what it is and lists numbered steps as comments. Together they're a complete training loop.

   | # | You write | What it teaches | Lesson § |
   |---|---|---|---|
   | 1 | `get_batch(data, block_size, batch_size, ...)` | turning text into training examples | 2 |
   | 2 | `causal_lm_loss(logits, targets)` | measuring how wrong the guesses are | 3 |
   | 3 | `lr_at(step, ...)` | warmup, then cosine decay | 5 |
   | 4 | `clip_grad_norm(params, max_norm)` | keeping one bad batch from wrecking the model | 6 |
   | 5 | `train_step(model, batches, optimizer, max_norm)` | one full step, with accumulation and clipping | 4, 7 |
   | 6 | `estimate_loss(model, data, ...)` | an honest measurement of the loss | 8 |

   Check your progress any time with `make check M=04`. The exercise tests use a tiny model and run in seconds.

3. **Answer the understanding checks** in [`answers.toml`](answers.toml): six questions about the starting loss, overfitting, learning rates, accumulation, warmup and clipping. Predict first, then check yourself in the notebook.

4. **Train your tiny GPT:**
   ```bash
   uv run python modules/m04_training_loop/train.py
   ```
   It trains the 4.8M-parameter model on your Mac's GPU with your six functions: 2,000 steps of 2 micro-batches of 16 windows, 256 characters each. That takes about 2 minutes and about 1.3 GB of memory. It prints the train and validation losses every 200 steps, then a sample, and saves `outputs/m04/run_summary.json`, `loss_curve.png` and `samples.txt`.

5. **Pass the checkpoint:** run `make check M=04`. M4 turns green when every exercise passes, your answers are right, and your run reached a validation loss below **1.70** with at least **80%** real words in its sample (words that appear somewhere in the training text), using your own code.

## Glossary

| Term | In plain words | Lesson § |
|---|---|---|
| **training** | Guess, measure the loss, work out the gradients, nudge the weights; repeated thousands of times. | 0 |
| **step** | One round of that: one optimizer update. | 0 |
| **loss** | One number for how wrong the model's guesses were. Training makes it smaller. | 0 |
| **training set** | The text the model learns from: here the first 90%. | 1 |
| **validation set** | Text kept back and never trained on, to measure the model honestly: the last 10%. | 1 |
| **context length** | How many tokens the model sees at once (also called the block size): 256 characters here. | 2 |
| **batch** | Several training windows processed together, each from a random place. | 2 |
| **logits** | The model's raw score for every possible next token, at every position. | 3 |
| **cross-entropy** | The loss for picking one answer out of many: `-ln(probability of the right one)`. | 3 |
| **causal LM loss** | Cross-entropy for the next token, averaged over every position in the batch. | 3 |
| **optimizer** | The part that uses the gradients to nudge the weights. | 4 |
| **AdamW** | The optimizer most language models use: it scales each weight's nudge by its recent gradients. | 4 |
| **learning rate** | How big each nudge is. Too small learns slowly; too big bounces or blows up. | 4 |
| **learning-rate schedule** | A plan for changing the learning rate during training. | 5 |
| **warmup** | Climbing from a small learning rate to the peak over the first steps. | 5 |
| **cosine decay** | Gliding from the peak down to a minimum along half a cosine wave. | 5 |
| **gradient norm** | The length of all the gradients together: the square root of every value squared, summed. | 6 |
| **gradient clipping** | Shrinking all gradients by the same factor when their norm is above a limit. | 6 |
| **gradient accumulation** | Adding up the gradients of several small batches before one optimizer step. | 7 |
| **micro-batch** | One of those small batches. | 7 |
| **effective batch size** | Micro-batch size times the number of micro-batches: what each step learns from. | 7 |
| **dropout** | Switching off a random share of the model's numbers at each training step, so it can't lean on any one. | 8 |
| **loss curve** | The loss plotted against the step, for the training and validation sets. | 8 |
| **overfitting** | Getting better at the training text and worse at new text: the two loss curves split. | 8 |

## Syntax at a glance

| Code | What it does | When you use it |
|---|---|---|
| `torch.randint(high, (n,), generator=g)` | `n` random whole numbers from 0 up to `high`. | Random starts for a batch. |
| `torch.Generator().manual_seed(0)` | A random-number source that repeats exactly. | Runs you can reproduce. |
| `torch.stack([...])` | Piles same-shaped tensors into one, with a new first dimension. | Rows of a batch. |
| `.reshape(a, b)` | The same numbers in a new shape; `-1` means "whatever fits". | Flattening for cross-entropy. |
| `F.cross_entropy(logits, targets)` | The average cross-entropy loss. | The causal LM loss. |
| `math.log`, `math.cos`, `math.pi` | Natural log, cosine and π. | The starting loss; cosine decay. |
| `torch.optim.AdamW(model.parameters(), lr=...)` | An AdamW optimizer for every weight. | Every training run. |
| `optimizer.zero_grad()` | Clears the gradients. | Once at the start of each step. |
| `loss.backward()` | Works out every gradient, adding to `.grad`. | Each micro-batch. |
| `optimizer.step()` | Nudges every weight. | Once at the end of each step. |
| `p.grad` | A weight's gradient (or `None` if it has none). | Measuring and clipping gradients. |
| `.mul_(x)` | Multiplies a tensor by `x` in place (the `_` means in place). | Clipping. |
| `.item()` | A one-number tensor as a plain Python number. | Logging the loss. |
| `model.eval()` / `model.train()` | Evaluation mode (dropout off) / training mode (dropout on). | Measuring the loss honestly. |
| `torch.no_grad()` | A block that records nothing for gradients. | Measuring without training. |
| `plt.plot`, `plt.legend` | Draw a line chart, and label its lines. | Loss curves. |

## Stretch

- Run `train.py --precision bf16`. It does most of the maths in bfloat16, a 16-bit number format (M0), instead of 32-bit floats. Compare the time, the peak memory and the final loss with your first run. On the reference Mac it was about 25% faster, with almost the same loss.
- Try `--lr 3e-3` and `--lr 1e-2`, or `--accum 1` and `--accum 4`, and compare the loss curves in `outputs/m04/loss_curve.png`. Each run overwrites the last, so note the numbers down.

## Stuck?

- `make hint M=04 EX=5` gives a nudge for exercise 5 (`EX=q2` for question 2).
- `make solution M=04 EX=5` shows the reference answer for just that exercise.

Both are logged on your dashboard. Try for a while first. To redo the module from scratch, run `make reset M=04`, which also puts the lesson notebook back to unrun.
