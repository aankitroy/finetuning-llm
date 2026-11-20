"""M1 exercises. Replace each `raise NotImplementedError` with your code.

Each exercise says in plain words what it is, then lists numbered steps as comments.
Write one or two lines of code under each step, then delete the `raise` line.
The lesson section named in each exercise shows every piece you need.

Check your work any time with:  make check M=01
Stuck?  make hint M=01 EX=1  (a nudge, logged on your dashboard)
"""

import torch
from torch import nn
from torch.nn import functional as F  # noqa: F401  (you'll want F.relu / F.cross_entropy)


def linear_forward(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Exercise 1. A linear layer, by hand.

    In plain words: multiply the inputs by the weights, then add the bias. This is what
    `nn.Linear` does inside. See lesson section 3 (`pred = X @ w + b`).

    Shapes: x is (batch, in), w is (in, out), b is (out,). The result is (batch, out).
    """
    # 1. Matrix-multiply x by w (the `@` operator, section 1).
    # 2. Add b. Broadcasting adds it to every row for you.
    # 3. Return the result.
    raise NotImplementedError("Exercise 1: linear_forward")


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Exercise 2. Mean squared error, by hand (don't use nn.MSELoss).

    In plain words: how wrong are the predictions, on average? For each position take
    (prediction - target), square it, then average them all into one number.
    See lesson section 3.
    """
    # 1. Subtract target from pred.
    # 2. Square every difference.
    # 3. Average them into a single number (a 0-d tensor) and return it.
    raise NotImplementedError("Exercise 2: mse_loss")


def sgd_step(params: list[torch.Tensor], lr: float) -> None:
    """Exercise 3. One step of gradient descent, by hand.

    In plain words: nudge each parameter a little way downhill, then throw away its
    gradient so the next backward() starts fresh. See lesson section 2, the loop
    with `w -= lr * w.grad`.

    Nothing is returned: the parameters are changed in place.
    """
    # 1. Switch off gradient recording for the update (a `with` block).
    # 2. Inside it, go through each parameter p in params.
    # 3.   Subtract lr times p's gradient from p, in place (so p stays the same tensor).
    # 4.   Reset p's gradient.
    raise NotImplementedError("Exercise 3: sgd_step")


class MLP(nn.Module):
    """Exercise 4. A small neural network: Linear -> ReLU -> Linear.

    In plain words: the first layer turns each input into `hidden` numbers, ReLU zeroes
    the negative ones (so the network can bend its boundary), and the second layer
    turns those into one score (logit) per class. Follow the pattern of `SineNet` in
    lesson section 7, with ReLU (section 5) instead of tanh.

    forward() returns raw logits (no softmax) of shape (batch, out_features).
    """

    def __init__(self, in_features: int, hidden: int, out_features: int):
        super().__init__()
        # 1. Create a linear layer from in_features to hidden, and store it on self.
        # 2. Create a linear layer from hidden to out_features, and store it on self.
        raise NotImplementedError("Exercise 4: build the layers in MLP.__init__")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Pass x through the first layer.
        # 2. Apply ReLU.
        # 3. Pass the result through the second layer and return it (no softmax).
        raise NotImplementedError("Exercise 4: MLP.forward")


def train_one_epoch(model, X, y, optimizer, batch_size: int) -> float:
    """Exercise 5. Train for one epoch, in shuffled batches.

    In plain words: go through all the training data once, a batch at a time, doing
    the five beats for each batch. Lesson section 7 shows the five beats; section 6
    shows shuffling and slicing into batches; section 5 shows the classification loss.

    X is (examples, features), y holds one class number per example.
    Return the average of the batch losses as a plain Python float.
    """
    # 1. Make a shuffled order of the row positions 0..len(X)-1, on X's device.
    # 2. Make an empty list to collect each batch's loss.
    # 3. Loop over that order in slices of batch_size. For each slice:
    # 4.   Forward: get the model's logits for those rows of X.
    # 5.   Loss: cross-entropy between those logits and the same rows of y.
    # 6.   Clear the old gradients (through the optimizer).
    # 7.   Backward: compute the new gradients.
    # 8.   Step: let the optimizer nudge the parameters.
    # 9.   Add this batch's loss to your list as a plain Python number.
    # 10. Return the average of the list.
    raise NotImplementedError("Exercise 5: train_one_epoch")


@torch.no_grad()
def accuracy(model, X, y) -> float:
    """Exercise 6. What fraction of examples does the model get right?

    In plain words: run the model, take the class with the biggest logit for each
    example, and count how often it equals the true label. See lesson section 5
    (argmax). `@torch.no_grad()` above means nothing here is recorded for gradients.
    """
    # 1. Get the logits for all of X.
    # 2. For each example, take the position of the biggest logit (argmax along the
    #    class dimension, dim=1). That's the predicted class.
    # 3. Compare the predictions with y (gives True/False for each example).
    # 4. Turn True/False into 1.0/0.0, average, and return it as a Python float.
    raise NotImplementedError("Exercise 6: accuracy")
