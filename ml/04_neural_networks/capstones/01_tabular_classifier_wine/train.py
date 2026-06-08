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
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import joblib
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

@dataclass
class DataSet:
    X_tr: np.array
    X_va: np.array
    X_te: np.array
    y_tr: np.array
    y_va: np.array
    y_te: np.array

def set_seed(seed: int) -> None:
    """Make the run reproducible across numpy and torch."""
    # TODO: seed numpy and torch.
    np.random.seed(seed); torch.manual_seed(seed)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    X, y = load_wine(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    X_tr, X_va, y_tr, y_va = train_test_split(X_tr, y_tr, test_size=0.2, random_state=seed, stratify=y_tr)
    
    sc = StandardScaler().fit(X_tr)
    X_tr, X_va, X_te = sc.transform(X_tr), sc.transform(X_va), sc.transform(X_te)
    
    ds = DataSet(X_tr=X_tr, X_va=X_va, X_te=X_te, y_tr=y_tr, y_va=y_va, y_te=y_te)
        
    return ds, sc, X.shape[1], len(np.unique(y))
    


def make_loaders(X_tr, y_tr, X_va, y_va, batch_size: int):
    """Wrap the train/val arrays in tensors and DataLoaders.

    Reminders:
      - Features -> float32 tensors. Labels -> int64 (long) tensors of shape (N,)
        because CrossEntropyLoss expects class indices, not one-hot.
      - shuffle=True for the train loader, shuffle=False for validation.
    Returns: train_loader, val_loader
    """
    # TODO: implement per the docstring.
    ds_tr = TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.int64))
    ds_va = TensorDataset(torch.tensor(X_va, dtype=torch.float32), torch.tensor(y_va, dtype=torch.int64))
    
    tr_loader = DataLoader(ds_tr, batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(ds_va, batch_size=batch_size, shuffle=False)
    
    return tr_loader, va_loader
    

def build_model(n_features: int, n_classes: int, hidden: list[int]) -> nn.Module:
    """Build an MLP: [n_features] -> hidden... -> [n_classes] logits.

    Reminders:
      - One nn.Linear per hidden width, each followed by a non-linearity (ReLU).
      - Final layer outputs n_classes logits with NO activation (softmax is
        applied inside CrossEntropyLoss).
      - nn.Sequential(*layers) is a convenient way to assemble it.
    """
    # TODO: build and return the model.
    layers = []
    prev = n_features
    for h in hidden:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.ReLU())
        prev = h
    layers.append(nn.Linear(prev, n_classes))
    
    return nn.Sequential(*layers)


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
    
    history = {"train": [], "val": []}

    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)   # move batch to device location: no op if CPU
            optimizer.zero_grad()                   # gradients accumulate - must clear them each step
            loss = loss_fn(model(xb), yb)           # predict + compute loss
            loss.backward()                         # backprop
            optimizer.step()                        # update weights
            running_train_loss += loss.item() * xb.size(0)
        train_loss = running_train_loss / len(train_loader.dataset)
        
        # Validation pass
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                loss = loss_fn(model(xb), yb)       # compute val loss but do NOT backprop or step
                running_val_loss += loss.item() * xb.size(0)
        val_loss = running_val_loss / len(val_loader.dataset)
        
        history["train"].append(train_loss)
        history["val"].append(val_loss)
        if (epoch + 1) % max(1, epochs // 10) == 0:
            print(f"epoch {epoch+1:3d} | train {train_loss:.4f} | val {val_loss:.4f}")
    
    return history


def evaluate(model, X_te, y_te, device):
    """Predict on the test set and return (accuracy, macro_f1).

    Reminders:
      - eval mode + torch.no_grad().
      - Predicted class = logits.argmax(dim=1).
      - Use accuracy_score and f1_score(..., average='macro').
    """
    # TODO: implement per the docstring.
    model.eval()
    with torch.no_grad():
        X_te_t = torch.tensor(X_te, dtype=torch.float32).to(device)
        y_pred = model(X_te_t).argmax(1).cpu().numpy()  # move back to CPU and convert to numpy for sklearn
        acc = accuracy_score(y_te, y_pred)
        f1 = f1_score(y_te, y_pred, average="macro")
    
    print(f"Test accuracy: {acc:.4%} | Test macro F1: {f1:.4%}")
    
    return acc, f1


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
    
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load data
    ds, scaler, n_features, n_classes = load_data(args.seed)
    # Tensor + data loaders
    tr_loader, va_loader = make_loaders(ds.X_tr, ds.y_tr, ds.X_va, ds.y_va, batch_size=args.batch_size)
    # Build model
    model = build_model(n_features, n_classes, args.hidden).to(device)
    # Loss fn and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Train
    history = train(model, tr_loader, va_loader, loss_fn, optimizer, args.epochs, device)
    # Evaluate
    acc, f1 = evaluate(model, ds.X_te, ds.y_te, device)
    # Save model and scaler
    torch.save(model.state_dict(), args.out)
    joblib.dump(scaler, args.out.replace(".pt", "_scaler.joblib"))
    print(f"Model and scaler saved to {args.out} and {args.out.replace('.pt', '_scaler.joblib')}")


if __name__ == "__main__":
    main()
