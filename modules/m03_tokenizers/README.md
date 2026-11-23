# M3 · Tokenizers & chat templates

**Builds on M2.** If you finished M2, you know everything this module assumes: tokens, token ids, the vocabulary and the end-of-text token. Every new idea is explained where it first appears, and every term is in the [glossary](#glossary) below.

**Goal:** see exactly what a model receives. You'll turn text into bytes and then tokens with a tokenizer you build yourself, then look at what a real chat model is sent when you give it a conversation. Getting this right is what keeps fine-tunes from failing silently in M5 and M10.

**Time:** about 2.5 hours.

**Download:** section 4 of the lesson and `tokenize_check.py` download the [SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) tokenizer (about 3 MB, Apache-2.0 licence; no model weights) and cache it. The checkpoint's tests need no download.

## Steps

1. **Read and run the lesson.**
   ```bash
   uv run jupyter lab modules/m03_tokenizers/lesson.ipynb
   ```

   | § | Section | You learn |
   |---|---|---|
   | 0 | What the model sees | token ids, and why neither characters nor words make good tokens |
   | 1 | Characters, bytes and UTF-8 | how any text becomes bytes, and why a cut-off character breaks |
   | 2 | BPE by hand | counting pairs, merging, and learning a tokenizer from text |
   | 3 | Encoding and decoding | turning text into tokens and back with your merges |
   | 4 | A real tokenizer | spaces, capitals, numbers and other languages in SmolLM2 |
   | 5 | Special tokens | BOS, EOS, padding, the attention mask and the padding side |
   | 6 | Chat templates | what a chat model really receives, and the generation prompt |
   | 7 | When templates go wrong | three silent mistakes, and the rule for fine-tuning |
   | 8 | Your task | the exercises and the run |

2. **Do the exercises** in [`exercises.py`](exercises.py). Each one says in plain words what it is and lists numbered steps as comments. Exercises 1 to 5 build a byte-level BPE tokenizer; exercise 6 formats a chat.

   | # | You write | What it teaches | Lesson § |
   |---|---|---|---|
   | 1 | `pair_counts(ids)` | finding the most common neighbouring tokens | 2 |
   | 2 | `merge(ids, pair, new_id)` | replacing a pair with one new token | 2 |
   | 3 | `train_bpe(text, num_merges)` | how a tokenizer is learned | 2 |
   | 4 | `encode(text, merges)` | turning text into tokens, merges in the right order | 3 |
   | 5 | `decode(ids, merges)` | turning tokens back into text, safely | 3 |
   | 6 | `render_chatml(messages, ...)` | exactly what a chat model is sent | 6 |

   Check your progress any time with `make check M=03`. The tests are plain Python and run in under a second.

3. **Answer the understanding checks** in [`answers.toml`](answers.toml): six questions where you predict a token count or a template's output for a tricky input. Predict first, then check yourself in the notebook.

4. **Run the check:**
   ```bash
   uv run python modules/m03_tokenizers/tokenize_check.py
   ```
   It trains your BPE on [`corpus.txt`](corpus.txt), checks that tricky text (emoji, Hindi, tabs, empty text) survives encode then decode, compares your `render_chatml` with the real template on 5 conversations, and saves `outputs/m03/run_summary.json`. It also reports how many bytes each of your tokens covers, next to SmolLM2's.

5. **Pass the checkpoint:** run `make check M=03`. M3 turns green when every exercise passes, your answers are right, and your run matched all 5 conversations with your own code.

## Glossary

| Term | In plain words | Lesson § |
|---|---|---|
| **token** | A small piece of text (a word, part of one, or a byte) that the model works with. | 0 |
| **token id** | A token's number: its position in the vocabulary. The model only sees these. | 0 |
| **vocabulary** | The full list of tokens a model knows: 49,152 for SmolLM2. | 0 |
| **byte** | A number from 0 to 255. All stored text is bytes. | 1 |
| **UTF-8** | The standard way to turn text into bytes: 1 byte for English letters, up to 4 for emoji. | 1 |
| **BPE** | Byte pair encoding: start from bytes and keep merging the most common pair into a new token. | 2 |
| **merge** | One BPE step: a pair of tokens becomes one new token everywhere it appears. | 2 |
| **encode** | Text → token ids. | 3 |
| **decode** | Token ids → text. | 3 |
| **special token** | A marker with its own id that is never split, like `<\|im_start\|>`. | 5 |
| **BOS** | Beginning of sequence: the special token that marks where a text starts. | 5 |
| **EOS** | End of sequence: the special token that marks where it ends, and stops generation. | 5 |
| **padding** | Filler tokens that make texts in a batch the same length. | 5 |
| **attention mask** | 1 for real tokens and 0 for padding, so the model ignores the padding. | 5 |
| **padding side** | Whether padding goes before (left) or after (right) the text. Left for generation. | 5 |
| **chat template** | The recipe that turns a list of messages into the one string a chat model receives. | 6 |
| **ChatML** | The chat format SmolLM2 and Qwen use: each message wrapped in `<\|im_start\|>` and `<\|im_end\|>`. | 6 |
| **system message** | The instruction at the start of a chat that sets how the model behaves. | 6 |
| **generation prompt** | The opening of the assistant's turn (`<\|im_start\|>assistant`), added so the model replies. | 6 |

## Syntax at a glance

| Code | What it does | When you use it |
|---|---|---|
| `ord` | A character's number. | Seeing that text is numbers. |
| `.encode("utf-8")` | Text → bytes. | The first step of byte-level BPE. |
| `.decode("utf-8", errors="replace")` | Bytes → text, showing � for broken characters. | Decoding tokens safely. |
| `bytes([...])` | Builds bytes from a list of numbers. | The vocabulary of single bytes. |
| `zip(ids, ids[1:])` | Each item next to the one after it. | Neighbouring pairs. |
| `dict.get` | A value, or a default if the key is missing. | Counting: `counts.get(pair, 0) + 1`. |
| `max(d, key=d.get)` | The key with the biggest value. | The most common pair. |
| `lambda` | A tiny one-line function. | `min(..., key=lambda p: ...)` when encoding. |
| `AutoTokenizer.from_pretrained` | Downloads (or reuses) a model's tokenizer. | Loading a real tokenizer. |
| `tokenizer.tokenize` | Text → token pieces, as text. | Seeing where the tokenizer cuts. |
| `tokenizer.convert_ids_to_tokens` | Ids → token pieces. | Reading what the model receives. |
| `padding=True` | Pads a batch of texts to the same length. | Running several texts at once. |
| `tokenizer.padding_side` | `"left"` or `"right"`. | `"left"` before batched generation. |
| `tokenizer.apply_chat_template` | Formats messages with the model's template. | Every time you talk to, or train, a chat model. |
| `split_special_tokens=True` | Treats special-token text as ordinary characters. | Encoding text from users safely. |

## Stretch

- Run `tokenize_check.py --merges 2000` and watch your bytes per token go up. In the notebook, print your 20 longest tokens: they're the corpus's most common words.
- Train your BPE on a page of text in another language and compare the bytes per token.

## Stuck?

- `make hint M=03 EX=4` gives a nudge for exercise 4 (`EX=q2` for question 2).
- `make solution M=03 EX=4` shows the reference answer for just that exercise.

Both are logged on your dashboard. Try for a while first. To redo the module from scratch, run `make reset M=03`, which also puts the lesson notebook back to unrun.
