"""Capstone (YOUR exercise): end-to-end neural-network classifier as a script.

This is a SKELETON for you to implement. Apply the recipe from
neural_networks.ipynb to a NEW task — *multiclass* classification of the Wine
dataset (3 classes, 13 features) — which adds one variation the notebook only
described: a multiclass output layer + CrossEntropyLoss + integer labels.

Time budget: ~2 hours (hard cap). The base version is sized for this — you've
already seen every piece in the notebook. The numbered EXTENSION CHALLENGES below
are OPTIONAL and beyond the budget; do not start them within the 2 hours.

Fill in every function marked `# TODO` (each currently raises
NotImplementedError) and wire them together in main(). When it's working you
should be able to run:

    uv run python capstone_tabular_nn.py
    uv run python capstone_tabular_nn.py --epochs 150 --lr 5e-4 --hidden 64 32

Target to beat: ~0.95+ test accuracy is very achievable on Wine.

------------------------------------------------------------------------------
THE RECIPE (your checklist — same as the notebook):
    seed -> load -> split (stratified) -> scale (fit on train only) ->
    tensors/loaders -> model -> train (with validation) -> evaluate on test ->
    save artifacts.

MULTICLASS DIFFERENCES vs the notebook's binary classifier:
    - output layer: n_classes logits (not 1), no activation
    - loss: nn.CrossEntropyLoss (not BCEWithLogitsLoss)
    - labels: int64 class indices (not float 0/1), shape (N,) not (N,1)
    - prediction: logits.argmax(dim=1) gives the predicted class

EXTENSION CHALLENGES (once the base version works):
    1. Add dropout + weight_decay; compare test accuracy. Does it help?
    2. Add early stopping (stop when val loss stalls; restore best weights).
    3. Add a --task flag to switch to regression on load_diabetes (1 output,
       MSELoss, report RMSE/R2).
    4. Swap the single val split for k-fold CV; report mean +/- std.
    5. Add an LR scheduler and watch the loss curve.
------------------------------------------------------------------------------
"""
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import joblib
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score


def set_seed(seed: int) -> None:
    """Make the run reproducible across numpy and torch."""
    # TODO: seed numpy and torch.
    raise NotImplementedError


def load_data(seed: int):
    """Load Wine, split into train/val/test, and scale features.

    Returns: (X_tr, y_tr, X_va, y_va, X_te, y_te), scaler, n_features, n_classes

    Requirements:
      - Use load_wine() for X (178x13) and y (labels 0/1/2).
      - Split into THREE sets (train/val/test) with train_test_split, stratified
        on the labels so class ratios are preserved in each split.
      - Fit a StandardScaler on the TRAINING features ONLY (avoid leakage), then
        transform train, val, and test with it.
      - n_features = number of columns; n_classes = number of distinct labels.
    """
    # TODO: implement per the docstring.
    raise NotImplementedError


def make_loaders(X_tr, y_tr, X_va, y_va, batch_size: int):
    """Wrap the train/val arrays in tensors and DataLoaders.

    Reminders:
      - Features -> float32 tensors. Labels -> int64 (long) tensors of shape (N,)
        because CrossEntropyLoss expects class indices, not one-hot.
      - shuffle=True for the train loader, shuffle=False for validation.
    Returns: train_loader, val_loader
    """
    # TODO: implement per the docstring.
    raise NotImplementedError


def build_model(n_features: int, n_classes: int, hidden: list[int]) -> nn.Module:
    """Build an MLP: [n_features] -> hidden... -> [n_classes] logits.

    Reminders:
      - One nn.Linear per hidden width, each followed by a non-linearity (ReLU).
      - Final layer outputs n_classes logits with NO activation (softmax is
        applied inside CrossEntropyLoss).
      - nn.Sequential(*layers) is a convenient way to assemble it.
    """
    # TODO: build and return the model.
    raise NotImplementedError


def train(model, train_loader, val_loader, loss_fn, optimizer, epochs, device):
    """Standard training loop with a validation pass each epoch.

    For each epoch:
      - TRAIN: set train mode; for each batch, clear gradients, forward, compute
        loss, backprop, step; accumulate the running train loss.
      - VALIDATE: set eval mode; under torch.no_grad(), accumulate the val loss
        (no weight updates).
      - Record mean train and val loss per epoch; optionally print progress.
    Returns: history dict with 'train' and 'val' loss lists.
    """
    # TODO: implement the loop (this is the core thing the capstone tests).
    raise NotImplementedError


def evaluate(model, X_te, y_te, device):
    """Predict on the test set and return (accuracy, macro_f1).

    Reminders:
      - eval mode + torch.no_grad().
      - Predicted class = logits.argmax(dim=1).
      - Use accuracy_score and f1_score(..., average='macro').
    """
    # TODO: implement per the docstring.
    raise NotImplementedError


def main():
    p = argparse.ArgumentParser(description="End-to-end MLP classifier on the Wine dataset.")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--hidden", type=int, nargs="+", default=[32, 16],
                   help="hidden layer widths, e.g. --hidden 64 32")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="wine_model.pt")
    args = p.parse_args()

    # TODO: wire the pieces together in the right order:
    #   1. set_seed(args.seed)
    #   2. choose device (cuda if available else cpu)
    #   3. data = load_data(args.seed)  -> unpack arrays, scaler, n_features, n_classes
    #   4. build train/val loaders with make_loaders(...)
    #   5. model = build_model(...).to(device)
    #   6. loss_fn = nn.CrossEntropyLoss(); optimizer = Adam(model.parameters(), lr=args.lr)
    #   7. train(...)
    #   8. acc, f1 = evaluate(...); print them
    #   9. save model state_dict (torch.save) AND the scaler (joblib.dump) --
    #      you need both to predict on new raw data later.
    raise NotImplementedError


if __name__ == "__main__":
    main()
