# M1 · Tensors, autograd & PyTorch essentials

**Goal:** know just enough PyTorch to read and write a training loop. Every fine-tune in this course is a training loop.

**Time:** about 3 hours.

## Steps

1. **Read and run the lesson.**
   ```bash
   uv run jupyter lab modules/m01_pytorch_essentials/lesson.ipynb
   ```
   It covers tensors and broadcasting, autograd, a hand-written linear regression, `nn.Module` and optimizers, and the Mac's GPU (MPS).

2. **Do the exercises** in [`exercises.py`](exercises.py). Replace each `raise NotImplementedError`:

   | # | You write | What it teaches |
   |---|---|---|
   | 1 | `linear_forward(x, w, b)` | what a linear layer really computes |
   | 2 | `mse_loss(pred, target)` | a loss is just a number to make smaller |
   | 3 | `sgd_step(params, lr)` | gradient descent by hand, and why gradients need resetting |
   | 4 | `MLP` | building a network with `nn.Module` |
   | 5 | `train_one_epoch(...)` | the five-beat training loop over mini-batches |
   | 6 | `accuracy(model, X, y)` | evaluating without tracking gradients |

   Check your progress any time with `make check M=01`. Each test's failure message says what to fix.

3. **Train your network** on the two-moons dataset:
   ```bash
   uv run python modules/m01_pytorch_essentials/train.py          # about 5 seconds on MPS
   uv run python modules/m01_pytorch_essentials/train.py --tiny   # even faster
   ```
   It uses *your* `exercises.py` and saves `outputs/m01/run_summary.json`.

4. **Pass the checkpoint:** run `make check M=01`. M1 turns green on the dashboard when every exercise passes and your validation accuracy reaches **95%**.

## Concepts

- **Tensors** have a shape, a dtype and a device. Bytes per number × number count = memory, which is how you'll size models later.
- **Autograd** records operations so `.backward()` can fill in `.grad`. Gradients accumulate, so reset them every step.
- **The training loop** is the same everywhere: forward → loss → `zero_grad()` → `backward()` → `step()`.
- **MPS** is how PyTorch uses the Apple GPU. The model and its data must be on the same device.

## Stretch

Try `train.py --hidden 4` and `--hidden 256`. Tag each check with `make check M=01 LABEL="h=4"` and compare the attempts on the dashboard.

## Stuck?

Reference solutions are in [`solutions/exercises.py`](solutions/exercises.py). Try for a while first: the struggle is where the learning happens.
