"""Train your M1 MLP on the two-moons dataset and save a run summary.

    uv run python modules/m01_pytorch_essentials/train.py           # full run
    uv run python modules/m01_pytorch_essentials/train.py --tiny    # quick check

Uses your code in exercises.py. Writes outputs/m01/run_summary.json, which
`make check M=01` reads.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

MODULE_DIR = Path(__file__).resolve().parent
ROOT = MODULE_DIR.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(MODULE_DIR))

from data import make_moons  # noqa: E402

from common.checkpoint import load_exercise  # noqa: E402


def pick_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tiny", action="store_true", help="fewer epochs, finishes in seconds")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    epochs = args.epochs or (40 if args.tiny else 150)

    ex = load_exercise(MODULE_DIR)
    torch.manual_seed(args.seed)
    device = pick_device()

    X, y = make_moons(n=2000, noise=0.2, seed=args.seed)
    split = int(0.8 * len(X))
    X_train, y_train = X[:split].to(device), y[:split].to(device)
    X_val, y_val = X[split:].to(device), y[split:].to(device)

    model = ex.MLP(in_features=2, hidden=args.hidden, out_features=2).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    print(f"Training on {device} for {epochs} epochs")
    start = time.time()
    for epoch in range(1, epochs + 1):
        loss = ex.train_one_epoch(model, X_train, y_train, opt, batch_size=64)
        if epoch == 1 or epoch % max(1, epochs // 10) == 0 or epoch == epochs:
            val = ex.accuracy(model, X_val, y_val)
            print(f"  epoch {epoch:4d}  loss {loss:.4f}  val accuracy {val:.1%}")

    summary = {
        "val_accuracy": ex.accuracy(model, X_val, y_val),
        "final_loss": loss,
        "epochs": epochs,
        "hidden": args.hidden,
        "lr": args.lr,
        "device": device,
        "seconds": round(time.time() - start, 2),
    }
    out = Path(os.environ.get("FTCOURSE_OUTPUTS") or ROOT / "outputs") / "m01"
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nValidation accuracy: {summary['val_accuracy']:.1%}  (need 95%)")
    print(f"Saved {out / 'run_summary.json'}. Next: make check M=01")
    return 0


if __name__ == "__main__":
    sys.exit(main())
