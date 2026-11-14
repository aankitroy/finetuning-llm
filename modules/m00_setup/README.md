# M0 · Environment & Apple Silicon setup

**Goal:** a reproducible environment, and a feel for how your Mac's memory works.

## Steps

1. Install [uv](https://docs.astral.sh/uv/) if you don't have it, then run `uv sync` from the repo root.
2. Run `make doctor`. It checks your Python version, chip and memory, then runs this module's checkpoint.
3. Open the dashboard with `make progress` and watch M0 tick over.

## Concepts

- **Unified memory.** On Apple Silicon the CPU and GPU share one pool of RAM. A 36 GB Mac has no separate "VRAM". Training uses the same memory as macOS and your apps, so plan for about 26–28 GB being usable.
- **MPS vs MLX.** PyTorch talks to the GPU through *MPS*. Apple's *MLX* is a separate framework built for this hardware. The course uses both.
- **dtypes.** fp32 uses 4 bytes per number, bf16 and fp16 use 2. Halving bytes roughly halves memory.

PyTorch and MLX aren't installed yet, so those two checks show as ○. That's expected: you'll add them in M1 and M7.

## Checkpoint

`make doctor`. It passes when Python 3.11+ and your memory are detected.
