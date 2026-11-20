# M2 · How an LLM actually works

**Builds on M1.** If you finished M1, you know everything this module assumes: tensors, logits, softmax, argmax and `nn.Linear`. Every new idea is explained where it first appears, and every term is in the [glossary](#glossary) below.

**Goal:** see inside a real language model, and write the loop that makes it produce text. That loop is the same one behind every chatbot. The checkpoint is that your loop writes exactly the same tokens as the library's `model.generate()`.

**Time:** about 3 hours.

**Download:** the first time you run the lesson or `generate.py`, it downloads [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) (about 270 MB, Apache-2.0 licence) and caches it. The checkpoint's tests need no download.

## Steps

1. **Read and run the lesson.**
   ```bash
   uv run jupyter lab modules/m02_how_llms_work/lesson.ipynb
   ```

   | § | Section | You learn |
   |---|---|---|
   | 0 | What a language model does | next-token prediction, tokens, the generation loop (with a toy model) |
   | 1 | Load a real model | the tokenizer, token ids, counting parameters and memory |
   | 2 | Inside the model | transformers, embeddings, the 30 stacked blocks |
   | 3 | Attention | queries, keys, values, the causal mask, worked by hand |
   | 4 | From logits to the next token | the model's output, top-5 guesses, greedy decoding |
   | 5 | A generation loop, by hand | predict, append, repeat; matching `model.generate()` |
   | 6 | Temperature and top-p | sampling, and how to make it safer or more adventurous |
   | 7 | The KV cache | why `generate()` is faster than your loop |
   | 8 | Your task | the exercises and the run |

2. **Do the exercises** in [`exercises.py`](exercises.py). Each one says in plain words what it is and lists numbered steps as comments.

   | # | You write | What it teaches | Lesson § |
   |---|---|---|---|
   | 1 | `next_token_logits(model, input_ids)` | what a model actually outputs | 4 |
   | 2 | `greedy(logits)` | the simplest way to pick a token | 4 |
   | 3 | `softmax_with_temperature(logits, t)` | sharper or flatter probabilities | 6 |
   | 4 | `top_p_filter(probs, p)` | cutting off the unlikely tail | 6 |
   | 5 | `causal_attention(q, k, v)` | one attention head, by hand | 3 |
   | 6 | `generate(model, input_ids, ...)` | the generation loop, greedy and sampled | 5, 6 |

   Check your progress any time with `make check M=02`. The tests use a tiny random model built in code, so they run in a second.

3. **Answer the understanding checks** in [`answers.toml`](answers.toml): six short questions on what a model outputs, attention, temperature, top-p, the causal mask and the KV cache.

4. **Run your loop on the real model:**
   ```bash
   uv run python modules/m02_how_llms_work/generate.py
   uv run python modules/m02_how_llms_work/generate.py --prompt "Once upon a time"
   ```
   It runs your `generate()` next to `model.generate()`, checks they write the same 20 tokens, compares their speed, and saves `outputs/m02/run_summary.json`.

5. **Pass the checkpoint:** run `make check M=02`. M2 turns green when every exercise passes, your answers are right, and your loop matched the library on the real model with your own code.

## Glossary

| Term | In plain words | Lesson § |
|---|---|---|
| **language model** | A model that, given some text, scores what could come next. Repeating that writes text. | 0 |
| **token** | A small piece of text (a word or part of one) that the model works with. Each has a number, its id. | 0, 1 |
| **vocabulary** | The full list of tokens a model knows: 49,152 for SmolLM2. | 0, 1 |
| **tokenizer** | Turns text into token ids and back. Each model has its own. | 1 |
| **transformer** | The design behind modern language models: embeddings, then a stack of attention + MLP blocks, then a layer that outputs logits. | 2 |
| **embedding** | The list of numbers (576 for SmolLM2) that a token id is turned into before the blocks. | 2 |
| **attention** | How each token gathers information from the tokens before it. | 3 |
| **query** | A token's "what am I looking for?" numbers. | 3 |
| **key** | A token's "what do I contain?" numbers, matched against queries. | 3 |
| **value** | The information a token hands over when another token attends to it. | 3 |
| **causal mask** | Hides later tokens, so each token only sees itself and earlier ones. | 3 |
| **logits** | Raw scores, one per vocabulary token, for what comes next. | 4 |
| **greedy** | Always picking the single most likely token. Predictable, but can get repetitive. | 4 |
| **end-of-text token** | A special token meaning "I'm finished". Generation stops when it's picked. | 5 |
| **sampling** | Picking the next token at random, weighted by its probability. | 6 |
| **temperature** | Divides the logits before softmax: below 1 is sharper and safer, above 1 is flatter and more surprising. | 6 |
| **top-p** | Keeps only the most likely tokens whose probabilities add up to at least p, then samples from them. | 6 |
| **KV cache** | Keeps earlier tokens' keys and values between steps, so each step only processes the newest token. Same output, much faster. | 7 |

## Syntax at a glance

| Code | What it does | When you use it |
|---|---|---|
| `AutoTokenizer.from_pretrained` | Downloads (or reuses) a model's tokenizer. | Before you can turn text into ids. |
| `AutoModelForCausalLM.from_pretrained` | Downloads (or reuses) a model's weights and builds it. | Loading any next-token model. |
| `tokenizer.encode` | Text → list of token ids. | Preparing a prompt. |
| `tokenizer.decode` | Token ids → text. | Reading what the model wrote. |
| `model(input_ids).logits` | Runs the model; logits of shape (batch, tokens, vocabulary). | Getting the next-token scores. |
| `torch.topk` | The k biggest values and their positions. | Showing the model's top guesses. |
| `torch.cat` | Joins tensors along a dimension. | Appending a new token to the sequence. |
| `torch.sort` | Sorts values and also returns where each came from. | Top-p filtering. |
| `torch.cumsum` | Running total. | Top-p filtering. |
| `torch.multinomial` | Draws random positions, weighted by probability. | Sampling the next token. |
| `torch.Generator().manual_seed(n)` | A private random-number source with a fixed seed. | Making sampling repeatable. |
| `torch.triu` | Keeps the upper triangle of a grid. | Building the causal mask. |
| `.masked_fill` | Replaces values wherever a mask is True. | Putting −∞ on future positions. |
| `k.T` | Transpose: swaps rows and columns. | Scoring queries against keys. |
| `model.generate` | The library's built-in generation loop (with a KV cache). | Real use, and checking your loop. |

## Stretch

- Try other prompts with `generate.py --prompt "..."`, and change the sampling settings in the notebook.
- In the notebook, load the model with `attn_implementation="eager"` and run it with `output_attentions=True` to see which earlier tokens each token attends to.

## Stuck?

- `make hint M=02 EX=4` gives a nudge for exercise 4 (`EX=q2` for question 2).
- `make solution M=02 EX=4` shows the reference answer for just that exercise.

Both are logged on your dashboard. Try for a while first. To redo the module from scratch, run `make reset M=02`, which also puts the lesson notebook back to unrun.
