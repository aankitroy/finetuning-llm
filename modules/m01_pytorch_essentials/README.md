# M1 · Tensors, autograd & PyTorch essentials

**No experience needed.** You don't need to know what a neural network is, and you don't need to have used PyTorch. If you can read a little Python, you can do this module. Every idea is explained where it first appears, and every term is in the [glossary](#glossary) below.

**Goal:** understand, line by line, the code that trains a small neural network. Every fine-tune in this course is that same code with a bigger model.

**Time:** about 4 hours. Take breaks; there's a lot of new vocabulary.

## Steps

1. **Read and run the lesson.**
   ```bash
   uv run jupyter lab modules/m01_pytorch_essentials/lesson.ipynb
   ```
   Run each cell with Shift-Enter and read the **What each line does** box under it. It goes from "what is a model?" to a working neural network in ten short sections:

   | § | Section | You learn |
   |---|---|---|
   | 0 | What is a model, and what does training mean? | the four-step training loop, with no code |
   | 1 | Tensors | grids of numbers, shapes, matrix multiply, broadcasting |
   | 2 | Gradients and autograd | which way is downhill, and how PyTorch works it out |
   | 3 | Linear regression, by hand | a whole model and training loop in ten lines |
   | 4 | nn.Module, layers and optimizers | PyTorch's ready-made pieces |
   | 5 | Classification | logits, softmax, cross-entropy, argmax, accuracy, ReLU |
   | 6 | Batches and epochs | training on a little data at a time |
   | 7 | A worked example | a small network end to end: the pattern for exercises 4 and 5 |
   | 8 | The Mac's GPU | MPS and `.to(device)` |
   | 9 | Your task | the two-moons dataset |

2. **Do the exercises** in [`exercises.py`](exercises.py). Each one says in plain words what it is and lists numbered steps as comments. Write a line or two under each step, then delete the `raise NotImplementedError`.

   | # | You write | What it teaches | Lesson § |
   |---|---|---|---|
   | 1 | `linear_forward(x, w, b)` | what a linear layer really computes | 1, 3 |
   | 2 | `mse_loss(pred, target)` | a loss is one number that says how wrong you are | 3 |
   | 3 | `sgd_step(params, lr)` | gradient descent by hand, and why gradients need resetting | 2 |
   | 4 | `MLP` | building a network with `nn.Module` | 5, 7 |
   | 5 | `train_one_epoch(...)` | the five-beat training loop over batches | 5, 6, 7 |
   | 6 | `accuracy(model, X, y)` | checking how often the model is right | 5 |

   Check your progress any time with `make check M=01`. Each failing test says what to fix.

3. **Answer the understanding checks** in [`answers.toml`](answers.toml): six short questions on shapes, memory, gradients, ReLU and logits. You can work them out in the notebook.

4. **Train your network** on the two-moons dataset:
   ```bash
   uv run python modules/m01_pytorch_essentials/train.py          # about 5 seconds on your Mac's GPU
   uv run python modules/m01_pytorch_essentials/train.py --tiny   # even faster
   ```
   It uses *your* `exercises.py` and saves `outputs/m01/run_summary.json`.

5. **Pass the checkpoint:** run `make check M=01`. M1 turns green on the dashboard when every exercise passes, your answers are right, and your network reaches **95%** validation accuracy with your own code.

## Glossary

| Term | In plain words | Lesson § |
|---|---|---|
| **model** | A function with adjustable numbers inside it. Given an input, it produces a prediction. | 0 |
| **parameter** | One of a model's adjustable numbers (also called a *weight*). Training changes these. | 0 |
| **bias** | A number a layer adds on after multiplying by the weights (the `b` in `x @ w + b`). | 0, 1 |
| **training** | Repeating guess → measure → find the direction → nudge until the predictions are good. | 0 |
| **loss** | One number saying how wrong the model's guesses are. Training tries to make it smaller. | 0, 3 |
| **tensor** | A grid of numbers: one number, a list, a table, or more dimensions. Everything in PyTorch is a tensor. | 1 |
| **shape** | How many numbers a tensor has along each dimension, e.g. `(2, 3)` is 2 rows × 3 columns. | 1 |
| **dtype** | The number format, e.g. `float32` (4 bytes per number) or `bfloat16` (2 bytes). | 1 |
| **device** | Which chip holds a tensor: `cpu` or `mps` (the Mac's GPU). | 1, 8 |
| **matrix multiply** | `A @ B`: each result is a row of `A` times a column of `B`, summed. (a, b) @ (b, c) gives (a, c). | 1 |
| **broadcasting** | PyTorch stretching a smaller tensor to fit, e.g. adding one bias row to every row of a batch. | 1 |
| **gradient** | For each parameter: which way, and how steeply, the loss changes if you nudge it. Step the opposite way to lower the loss. | 2 |
| **autograd** | PyTorch's automatic gradient calculator: it records the maths, and `.backward()` works out the gradients. | 2 |
| **learning rate** | How big a step each nudge takes. Too small is slow; too big overshoots. | 2 |
| **MSE** | Mean squared error: average of (prediction − truth)². The usual loss when predicting numbers. | 3 |
| **layer** | A reusable chunk of a model. `nn.Linear` computes `x @ w + b`. | 4 |
| **nn.Module** | The base for every PyTorch model. It keeps track of all the model's parameters. | 4, 7 |
| **optimizer** | The tool that nudges every parameter using its gradient, e.g. `SGD` or `Adam`. | 4 |
| **logits** | A classifier's raw scores, one per class. Any number; bigger means more likely. | 5 |
| **softmax** | Turns logits into probabilities between 0 and 1 that add up to 1. | 5 |
| **cross-entropy** | The usual loss for picking a class: small when the model gives the right class a high probability. | 5 |
| **argmax** | The position of the biggest value. On logits, that's the predicted class. | 5 |
| **ReLU** | Keeps positive numbers, turns negatives into 0. Placed between layers so the network can learn curved boundaries. | 5 |
| **batch** | A small group of examples used for one training step. | 6 |
| **epoch** | One full pass through all the training data, batch by batch. | 6 |
| **MPS** | Metal Performance Shaders: how PyTorch uses your Mac's GPU. | 8 |

## Syntax at a glance

| Code | What it does | When you use it |
|---|---|---|
| `torch.tensor([...])` | Makes a tensor from a number or a (nested) list. | Creating small inputs by hand. |
| `x.shape` | The tensor's size along each dimension. | Checking that layers fit together. |
| `A @ B` | Matrix multiply. | Inside every linear layer. |
| `requires_grad=True` | Tells autograd to record what happens to this tensor. | On parameters you want to train. |
| `.backward()` | Works out the gradient of a loss for every recorded tensor. | Once per training step, after the loss. |
| `.grad` | Where a tensor's gradient is stored after `.backward()`. | Updating parameters by hand. |
| `torch.no_grad()` | Switches off recording inside a `with` block (or as `@torch.no_grad()` on a function). | Updating parameters, or just making predictions. |
| `nn.Module` | The base class for a model; define layers in `__init__` and the maths in `forward`. | Building any network. |
| `nn.Linear` | A ready-made linear layer (`x @ w + b`) that owns its own `w` and `b`. | The building block of most networks. |
| `torch.relu` / `F.relu` | ReLU: negatives become 0. | Between two linear layers. |
| `model(x)` | Runs the model's `forward` on `x`. | The forward beat of training. |
| `model.parameters()` | Every parameter in the model. | Handing them to an optimizer. |
| `optimizer.zero_grad()` | Resets every parameter's gradient. | Before each `.backward()`. |
| `optimizer.step()` | Nudges every parameter using its gradient. | After each `.backward()`. |
| `F.cross_entropy` | Cross-entropy loss from logits and the correct class numbers. | Training a classifier. |
| `a == b` | Compares two tensors position by position, giving True/False for each. | Checking predictions against labels. |
| `.float()` | Turns a tensor into decimal numbers; True/False become 1.0/0.0. | Averaging True/False to get accuracy. |
| `.argmax(dim=1)` | The position of the biggest value in each row. | Turning logits into predicted classes. |
| `torch.randperm(n)` | The numbers 0 to n−1 in random order. | Shuffling the data each epoch. |
| `.to(device)` | Moves a model or tensor to a device (`"mps"` or `"cpu"`). | Using the Mac's GPU. |
| `.item()` | Turns a one-number tensor into a plain Python number. | Printing or collecting a loss. |

## Stretch

Try `train.py --hidden 4` and `--hidden 256`. Tag each check with `make check M=01 LABEL="h=4"` and compare the attempts on the dashboard.

## Stuck?

- `make hint M=01 EX=3` gives a nudge for exercise 3 (`EX=q2` for question 2).
- `make solution M=01 EX=3` shows the reference answer for just that exercise.

Both are logged on your dashboard. Try for a while first: the struggle is where the learning happens. To redo the module from scratch, run `make reset M=01`. That also puts the lesson notebook back to unrun.
