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

Keep the dashboard open beside your terminal. It updates on its own as you pass checkpoints. If you'd rather stay in the terminal, run `make progress TEXT=1`.

## How progress works

Each module ends with a checkpoint you run with `make check M=NN`, for example `make check M=06`. Every run is saved to `.progress/history.jsonl` on your machine, so the dashboard can chart your results across attempts. Add `LABEL="r=8"` to tag an attempt.

A module counts as done only when its **whole** checkpoint suite ran and passed: no failures, no skipped tests, no filtering with `-k`. Failed attempts are kept and charted, but never mark a module done. Nothing is ever sent off your machine.

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
