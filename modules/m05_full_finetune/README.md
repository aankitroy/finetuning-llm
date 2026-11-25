# M5 · Your first real fine-tune (full fine-tune)

**Builds on M1 to M4.** If you finished them, you know everything this module assumes: chat templates and padding (M3), and a training loop with a learning-rate schedule, clipping and validation (M4). Every new idea is explained where it first appears, and every term is in the [glossary](#glossary) below.

**Goal:** fine-tune a real 494-million-parameter model, updating every weight, so it rewrites jargon in plain English. You'll do it twice, with your own loop and with Hugging Face's `Trainer`, measure what it costs in memory, score the result, and check what the model forgot along the way.

**Time:** about 4 hours.

**Download:** the lesson and `train.py` download [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) (about 1 GB, Apache-2.0 licence) once and cache it. The checkpoint's exercise tests use a tiny random model and need no download.

**Memory:** a full fine-tune of this model peaks at about 24 GB. Close other big apps first, and close the lesson notebook (or restart its kernel) before you run `train.py`.

**Data:** [`data/`](data) holds 500 original jargon → plain English pairs (400 to train on, 50 to validate, 50 to test), written for this course and shared under CC BY 4.0 like the rest of its content, plus [`buzzwords.txt`](data/buzzwords.txt) and 10 ordinary questions for the forgetting check.

## Steps

1. **Read and run the lesson.**
   ```bash
   uv run jupyter lab modules/m05_full_finetune/lesson.ipynb
   ```

   | § | Section | You learn |
   |---|---|---|
   | 0 | What fine-tuning changes | pretrained vs fine-tuned weights, and what's new since M4 |
   | 1 | The original model tries the task | what an instruction-tuned model does before any training |
   | 2 | Formatting the pairs, and padding a batch | prompt + answer as one sequence, padding and the attention mask |
   | 3 | Loss masking: only the answer is graded | -100 labels, and the shift-by-one in the loss |
   | 4 | Where the memory goes | weights, gradients, optimizer state and activations, measured |
   | 5 | Fine-tuning for real | one epoch on the real model, and the difference it makes |
   | 6 | The same run with `Trainer` | `TrainingArguments` and a data collator |
   | 7 | Scoring the rewrites | chrF by hand, and counting buzzwords fairly |
   | 8 | Forgetting | what the model lost, and how to limit it |
   | 9 | Your task | the exercises and the runs |

2. **Do the exercises** in [`exercises.py`](exercises.py). Each one says in plain words what it is and lists numbered steps as comments.

   | # | You write | What it teaches | Lesson § |
   |---|---|---|---|
   | 1 | `build_example(prompt_ids, answer_ids, max_len)` | loss masking: grade only the answer | 3 |
   | 2 | `collate(examples, pad_id)` | padding a batch, with the attention mask and -100 labels | 2 |
   | 3 | `masked_lm_loss(logits, labels)` | the shifted, masked loss every fine-tune uses | 3 |
   | 4 | `optimizer_state_bytes(optimizer)` | what AdamW costs in memory | 4 |
   | 5 | `chrf(hypothesis, reference)` | scoring a rewrite against a reference | 7 |
   | 6 | `count_buzzwords(text, buzzwords)` | a simple, fair style check | 7 |

   Check your progress any time with `make check M=05`. The exercise tests use a tiny model and run in seconds.

3. **Answer the understanding checks** in [`answers.toml`](answers.toml): six questions on memory, masking, padding, the shift and forgetting. Predict first, then check yourself in the notebook.

4. **Fine-tune twice:**
   ```bash
   uv run python modules/m05_full_finetune/train.py --loop
   uv run python modules/m05_full_finetune/train.py --trainer
   ```
   Each one trains all 494M weights for 2 epochs (about 40 seconds of training on the reference Mac), then generates rewrites of the 50 test sentences with the original and the fine-tuned model, and asks both the 10 ordinary questions. The first run also scores the original model, which is cached for later runs. Each takes about 3 minutes in all, and saves `outputs/m05/run_loop.json` or `run_trainer.json`, with every test rewrite in it.

5. **Pass the checkpoint:** run `make check M=05`. M5 turns green when every exercise passes, your answers are right, and in both runs, with your own code:
   - the loss on the held-out rewrites fell by at least **0.4** (about 1.55 → 0.94 in the reference run): the model has learned the style
   - the buzzwords per rewrite didn't go up
   - chrF stayed within **3** points of the original model's: the rewrites still say the same thing

   The forgetting check is reported, not graded.

## Glossary

| Term | In plain words | Lesson § |
|---|---|---|
| **fine-tuning** | Training an already-trained model a little more, on your own examples. | 0 |
| **pretrained weights** | The weights as the model's makers trained them. | 0 |
| **fine-tuned weights** | The same weights after your extra training. | 0 |
| **full fine-tune** | Fine-tuning that updates every weight of the model. | 0 |
| **loss masking** | Labels of -100 on the prompt (and padding), so only the answer is graded. | 3 |
| **optimizer state** | What the optimizer remembers for each weight: for AdamW, two running averages. | 4 |
| **activations** | The in-between results of the forward pass, kept so `.backward()` can use them. | 4 |
| **epoch** | One pass over every training example. | 5 |
| **`Trainer`** | Hugging Face's ready-made training loop. | 6 |
| **`TrainingArguments`** | The settings for a `Trainer` run: batch size, epochs, learning rate, logging. | 6 |
| **data collator** | The function that turns a list of examples into one padded batch. | 6 |
| **chrF** | Character n-gram F-score: how much a rewrite's character pieces overlap the reference's, from 0 to 100. | 7 |
| **precision** | The share of the rewrite's pieces that are also in the reference. | 7 |
| **recall** | The share of the reference's pieces that the rewrite has. | 7 |
| **F-score** | Precision and recall combined into one number; chrF weights recall twice as much. | 7 |
| **buzzwords** | Jargon words and phrases from a fixed list, counted as whole words. | 7 |
| **catastrophic forgetting** | Losing old skills because fine-tuning on one narrow task changed every weight. | 8 |

## Syntax at a glance

| Code | What it does | When you use it |
|---|---|---|
| `list_a + list_b` | Joins two lists into one. | Prompt ids then answer ids. |
| `[-100] * n` | A list of `n` copies of -100. | Masking the prompt, padding the labels. |
| `items[:max_len]` | The first `max_len` items. | Cutting an example to length. |
| `max(len(x) for x in rows)` | The length of the longest row. | How far to pad. |
| `torch.tensor(rows, dtype=torch.long)` | Equal-length lists → a tensor of whole numbers. | Building a batch. |
| `logits[:, :-1, :]`, `labels[:, 1:]` | All but the last position / all but the first. | Lining each guess up with the next token. |
| `F.cross_entropy(..., ignore_index=-100)` | Cross-entropy that skips labels of -100. | The masked loss. |
| `model(input_ids=..., labels=...).loss` | The model's own shifted, masked loss. | Checking yours; training in the lesson. |
| `optimizer.state` | A dict of what the optimizer stores for each weight. | Measuring its memory. |
| `torch.is_tensor(x)` | `True` if `x` is a tensor. | Skipping the plain numbers in optimizer state. |
| `.numel()`, `.element_size()` | How many numbers / bytes per number. | Bytes = one times the other. |
| `TrainingArguments(...)`, `Trainer(...).train()` | Describe a run, then run it. | Training without writing the loop. |
| `.replace(" ", "")` | Removes every space. | chrF compares characters, not words. |
| `c.isalpha()` | `True` for a letter. | Splitting text into words fairly. |
| `words[i : i + n] == phrase` | Do these `n` words match the phrase, in order? | Counting multi-word buzzwords. |
| `continue` | Skips to the next time round the loop. | chrF sizes a text is too short for. |

## Stretch

- Run `train.py --loop --epochs 3 --lr 2e-5`. Watch the validation loss after each epoch (it goes back up), and the ordinary-questions score (it drops). Then try `--epochs 1`. Which would you ship?
- Compare `run_loop.json` and `run_trainer.json`: speed, peak memory and scores. They run the same recipe, so the differences come from details such as how `Trainer` averages the loss and schedules the learning rate.
- Read the `examples` in a run summary and find a rewrite where chrF is low but the rewrite is good. That's why M11 is about evaluation.

## Stuck?

- `make hint M=05 EX=5` gives a nudge for exercise 5 (`EX=q2` for question 2).
- `make solution M=05 EX=5` shows the reference answer for just that exercise.

Both are logged on your dashboard. Try for a while first. To redo the module from scratch, run `make reset M=05`, which also puts the lesson notebook back to unrun.
