# LLM Fine-Tuning, Step by Step

Learn to fine-tune large language models by building it yourself, one module at a time, on a single Apple Silicon Mac.

[![CI](../../actions/workflows/ci.yml/badge.svg)](../../actions/workflows/ci.yml)
[![Code: MIT](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE)
[![Content: CC BY 4.0](https://img.shields.io/badge/content-CC%20BY%204.0-lightgrey.svg)](LICENSE-CONTENT)

You'll work in the terminal: read a lesson, run the code, pass the checkpoint. A local dashboard keeps track of what you've done, shows your results over time, and tells you what to do next.

## What you'll learn

| Part | Modules |
|---|---|
| A · Foundations | Setup, PyTorch essentials, how LLMs work, tokenizers & chat templates |
| B · Training from zero | A training loop from scratch, your first full fine-tune |
| C · Efficient tuning | LoRA from scratch, LoRA with MLX & PEFT, QLoRA |
| D · Making it useful | Data curation, instruction tuning (SFT), evaluation |
| E · Alignment | Preference tuning (DPO/ORPO), RL with verifiable rewards (GRPO) |
| F · Shipping | Merge, quantize & export to GGUF, serve locally |
| Capstone | Build a domain specialist end to end |
| G · Beyond text (optional) | Vision-language models, speech-to-text (Whisper), image generation (diffusion LoRA) |

The full syllabus is in [COURSE.md](COURSE.md).

## Hardware

Built and tested on Apple Silicon Macs. The reference machine is an M5 Max with 36 GB of unified memory. On that much memory you can comfortably do:

- a full fine-tune of a model up to about 1.5B parameters
- LoRA on models up to about 8B
- QLoRA (4-bit) on models up to about 14B

Smaller Macs work for most modules using each lesson's `--tiny` settings.

## Quick start

```bash
git clone <this repo>
cd finetuning-llm
uv sync            # install dependencies (needs https://docs.astral.sh/uv/)
make doctor        # check your machine; this completes M0
make progress      # open the dashboard in your browser
```

Keep the dashboard open beside your terminal. It shows the steps for the module you're on, with the command for each one, and ticks them off as you go: cells run in the lesson, exercises written and passing, questions answered, training done, checkpoint passed. Nothing is ticked by hand. If you'd rather stay in the terminal, run `make progress TEXT=1`.

## How progress works

Each module has three parts, and you do all of them:

1. **Lesson** (`lesson.ipynb`): run it cell by cell. Cells marked *Predict* ask you to guess before you run.
2. **Exercises** (`exercises.py`): write the code yourself. It starts as empty stubs.
3. **Understanding checks** (`answers.toml`): short questions about *why* things work, not just *whether* they do.

Then run the checkpoint with `make check M=NN`, for example `make check M=01`. It checks your code, your training results and your answers. Every run is saved to `.progress/history.jsonl` on your machine, so the dashboard can chart your attempts. Add `LABEL="r=8"` to tag one.

A module counts as done only when its **whole** checkpoint passed: no failures, no skipped tests, no filtering with `-k`, and nothing run against the reference solutions. Failed attempts are kept and charted, but never mark a module done. Nothing is ever sent off your machine.

### When you're stuck

```bash
make hint M=01 EX=3       # a nudge for exercise 3 (or EX=q2 for a question)
make solution M=01 EX=3   # the reference answer for just that exercise
```

Both are fine to use, and both are logged: the dashboard shows "2 hints, 1 solution viewed" next to the module, so you can see how much you did unaided.

### Starting over

```bash
make reset          # the whole course
make reset M=01     # one module
```

It lists exactly what will be deleted (your progress records, hint log, training outputs, and your `exercises.py` / `answers.toml`, which are restored to the empty starters) and does nothing until you type `reset`. Lesson notebooks and reference solutions are kept.

## Repo layout

```
COURSE.md            the syllabus
modules/mNN_*/       one folder per module: README, notebooks, checkpoint tests, module.toml
modules/_template/   start here when adding a module
common/              shared code: progress tracking, metric reporting
scripts/             doctor.py, progress_server.py
dashboard/           the progress page (plain HTML/JS, works offline)
tests/               tests for the course tooling itself
```

## Contributing

Contributions are welcome, from typo fixes to whole new modules. See [CONTRIBUTING.md](CONTRIBUTING.md) and our [Code of Conduct](CODE_OF_CONDUCT.md).

## License

- Code: [MIT](LICENSE)
- Course content (`COURSE.md`, module READMEs and notebooks): [CC BY 4.0](LICENSE-CONTENT)
