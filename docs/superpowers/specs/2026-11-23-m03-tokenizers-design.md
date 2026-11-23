# M3 · Tokenizers & chat templates — design

Approved in chat on 2026-11-23 (design and the tokenizer-only download).

## Why
The learner finished M2 and wants M3 in the same beginner-first style. COURSE.md: "understand
what the model really sees"; checkpoint "correctly predict the token count and the template
output for five tricky inputs". It sets up M5/M10, where a wrong chat template silently ruins a
fine-tune.

## Goals / non-goals
- Concept-first lesson, a **What each line does** box after every code cell, bold terms,
  glossary + syntax table in the README (as M1/M2).
- The learner writes a byte-level BPE from scratch and renders ChatML by hand.
- Not loss masking or packing (M5/M10), not training a tokenizer at scale.

## Approaches
1. **Toy BPE in pure Python + the real SmolLM2-Instruct tokenizer for inspection (chosen).**
   Tests are pure Python with fixed expected strings, so no download in tests or CI.
2. Use the `tokenizers` library to train BPE: faster, but hides the algorithm, which is the point.
3. Only inspect real tokenizers: no from-scratch understanding.

## Design
- **Download:** tokenizer files of `HuggingFaceTB/SmolLM2-135M-Instruct` (~3 MB, no weights).
  Its template is ChatML and injects a default system prompt when the first message isn't one.
- **Lesson sections:** 0 What the model sees · 1 Characters, bytes and UTF-8 · 2 BPE by hand ·
  3 Encoding and decoding · 4 A real tokenizer · 5 Special tokens (BOS/EOS, padding, padding side,
  attention mask) · 6 Chat templates · 7 When templates go wrong · 8 Your task.
- **Exercises:** `pair_counts(ids)`, `merge(ids, pair, new_id)`, `train_bpe(text, num_merges)`
  (new ids from 256; most frequent pair, ties → first seen), `encode(text, merges)` (lowest merge
  id first), `decode(ids, merges)` (UTF-8, `errors="replace"`),
  `render_chatml(messages, add_generation_prompt=False, default_system=None)`.
- **Questions (6):** token count of "12345" (5), " hello" vs "hello", token count of
  "<|im_start|>" (1), output with no system message (default system added), what
  `add_generation_prompt` adds, padding side for batched generation (left).
- **Run step:** `tokenize_check.py` trains your BPE on the bundled `corpus.txt`, checks round-trips
  on tricky strings (emoji, Hindi, whitespace, empty), compares `render_chatml` with the real
  template on 5 conversations, and writes `outputs/m03/run_summary.json` (`templates_matched`,
  `templates_compared`, `roundtrip_ok`, `your_bytes_per_token`, `smollm_bytes_per_token`, `source`,
  `used_solutions`). `--offline` renders a stored copy of the template with jinja2 (tests/CI).
- **Checkpoint:** six exercise tests, six answers, run matched 5/5 with round-trips OK, run used
  the real tokenizer and your own code. Metric: `bytes_per_token` (higher is better).
