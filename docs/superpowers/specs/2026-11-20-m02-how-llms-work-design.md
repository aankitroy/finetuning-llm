# M2 · How an LLM actually works — design

Approved in chat on 2026-11-20 (design and both downloads).

## Why
The learner finished M1 (no prior NN/PyTorch background) and wants M2 in the same
beginner-first style. COURSE.md: "a working mental model of a decoder-only transformer";
checkpoint "your handwritten generation loop matches `model.generate()` token for token in
greedy mode".

## Goals / non-goals
- Explain every idea in plain words before code; a **What each line does** box after every code
  cell; every term bold where defined; glossary + syntax table in the README (as M1).
- The learner writes: next-token logits, greedy pick, temperature, top-p, causal attention by
  hand, and the generation loop.
- Not tokenizers in depth (that's M3), not training (M4).

## Approaches
1. **Real small model + tiny random model for tests (chosen).** Lesson and `generate.py` use
   `HuggingFaceTB/SmolLM2-135M` (~270 MB, Apache-2.0); tests build a tiny random
   `LlamaForCausalLM` in code, so tests and CI need no download.
2. Real model in tests too: simplest, but a 270 MB download in CI and slower checks.
3. A from-scratch toy transformer only: no download, but never touches a real LLM, which is the
   point of M2.

## Design
- **Dependency:** `transformers` added to the `course` group.
- **Lesson sections:** 0 What a language model does (next-token prediction, repeat) · 1 Load a
  real model (tokens in brief, the tokenizer, parameters and memory) · 2 Inside the model
  (embeddings, layers, the architecture printout) · 3 Attention (scores, causal mask, softmax,
  weighted mix; a tiny worked example) · 4 From logits to the next token (greedy) · 5 A generation
  loop by hand · 6 Temperature and top-p · 7 The KV cache (why `generate()` is faster) · 8 Your task.
- **Exercises:** `next_token_logits(model, input_ids)`, `greedy(logits)`,
  `softmax_with_temperature(logits, temperature)`, `top_p_filter(probs, p)`,
  `causal_attention(q, k, v)`, `generate(model, input_ids, max_new_tokens, temperature=0.0,
  top_p=1.0, eos_token_id=None, generator=None)`. Step comments, "In plain words" docstrings.
- **Questions (6):** what a model outputs per step, attention score shape, temperature effect,
  top-p meaning, why the causal mask, what the KV cache does.
- **Run step:** `generate.py` runs your loop and `model.generate()` (greedy, 20 tokens, CPU
  float32 for exact comparison) on SmolLM2 and writes `outputs/m02/run_summary.json`
  (`matched`, `model`, tokens/sec for both). `--tiny` uses the random model.
- **Checkpoint:** six exercise tests (greedy loop vs `generate()` on the tiny model, attention vs
  `F.scaled_dot_product_attention`, etc.), answers, `matched` true, run used the real model and
  your own code.
- **Folded in (M1 review minors):** shorter hints 3 and 5, q5 hint points at the lesson, drop
  `model.train()` from the reference solution, gitignore `.ipynb_checkpoints/`, skip message only
  on a real diff.
