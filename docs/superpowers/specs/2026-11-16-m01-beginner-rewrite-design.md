# M1 for complete beginners — design

## Why

The course owner took M1 themselves and had to ask ChatGPT what a tensor, autograd,
logits and cross-entropy are, and what each piece of PyTorch syntax
(`requires_grad`, `torch.no_grad()`, `nn.Module`, `.backward()`, `optimizer.step()`…)
is for. The lesson names these things without explaining them. A learner who has never
seen a neural network or PyTorch should be able to finish M1 with only the repo open.

## Goals

- Every idea is explained in plain English, at the point it is first used, before the code.
- Every line of PyTorch syntax in the lesson is explained right after the cell that uses it.
- The exercises are approachable from zero: step-by-step comments to fill in, and a worked
  example in the notebook that each harder exercise mirrors.
- A one-page glossary in the README for looking things up again later.

## Non-goals / anti-goals

- Don't lower the bar: the goal stays "your own network reaches 95% on two moons", and the
  strict checkpoint, tests and pass rules are unchanged.
- Don't give the answers away: the step comments say *what* each line must do, not the code.
- No new module and no separate primer notebook (option 1 was chosen over those).

## Approaches considered

1. **Rewrite the lesson concept-first, in place (chosen).** Explanations sit next to the
   code they explain. Costs: the most writing; hints and questions need light updates.
2. Separate primer notebook before the lesson. Faster, but explanations live apart from the
   code, so learners still flip back and forth.
3. Glossary/cheat sheet only. Cheapest, but terms still arrive unexplained mid-cell.

## The design

**Lesson (`lesson.ipynb`), new order:**

0. What is a model, and what does training mean? (A model is a function with adjustable
   numbers; training = guess, measure how wrong, nudge the numbers, repeat.) Picture: the
   loop as a diagram.
1. Tensors — "a grid of numbers", with 0-d/1-d/2-d examples, shape/dtype/device, memory maths,
   matrix multiply `@` explained with a worked 2×3 @ 3×2 example, broadcasting.
2. Gradients and autograd — gradient as "which way is downhill, and how steep", a plot of
   (w−5)² with the slope marked; then `requires_grad`, `.backward()`, `.grad`,
   `torch.no_grad()`, why gradients accumulate (demonstrated).
3. Linear regression by hand — unchanged idea, with every line annotated.
4. `nn.Module`, `nn.Linear` and optimizers — what each replaces from section 3.
5. Classification: logits → softmax → cross-entropy → argmax. Plain words + a tiny worked
   example with three numbers, then `F.cross_entropy`.
6. Batches and epochs — what they are and why.
7. A worked example: a tiny network trained end to end on a different toy problem, written
   the same way `MLP` and `train_one_epoch` must be (so exercises 4–5 have a model to follow
   without being the answer).
8. MPS (the Mac GPU), then the two-moons task.

After each code cell, a **"What each line does"** box. Each new term is **bold** where it is
defined. Predict prompts stay.

**Exercises (`exercises.py` and `starter/exercises.py`):** same six functions, same
signatures and tests. Each docstring gains a plain-English "what this is" line, and the body
gets numbered step comments above the `raise NotImplementedError`, e.g. for
`train_one_epoch`: "1. shuffle the row order  2. for each slice of batch_size rows …".
Exercises 4–5 point to the notebook section they mirror.

**Questions (`answers.toml`):** keep q1–q5; add q6 on logits/cross-entropy (e.g. which class
is predicted from given logits), with hash, hint and solution. Reword q5 to lean on the lesson.

**README:** plain-language intro, glossary (tensor, shape, dtype, device, matrix multiply,
broadcasting, gradient, autograd, loss, MSE, logits, softmax, cross-entropy, argmax, ReLU,
layer, `nn.Module`, parameter, optimizer, learning rate, batch, epoch, MPS), and the
"syntax at a glance" table (call → what it does → when you use it).

**Hints:** reworded to match the new step comments.

**Reset fix (folded in):** add `starter/lesson.ipynb` (a clean, unrun copy) so `make reset`
restores the notebook too. `reset_plan`/`do_reset` already restore any file in `starter/`,
so this needs a test, not new code.

**Est. time** in `module.toml` goes from 180 to 240 minutes.

## Left to the builder

Exact wording, analogies and pictures; which toy problem the worked example in section 7
uses (must not be two moons, and must not be copy-paste-able into the exercises).
