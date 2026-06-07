"""Capstone (YOUR exercise): NN regression with EARLY STOPPING.

A direct sibling of capstone_tabular_nn.py — same skeleton style, same training
loop pattern. Two genuinely new things:

  1. REGRESSION on a continuous target (California Housing — predict median house
     value from 8 features). That changes:
        - loss:   nn.MSELoss (not CrossEntropy)
        - output: 1 neuron with NO activation (any real number)
        - target: float32 tensor of shape (N, 1) — matches model output shape
        - target scaling: scale y too (helps MSE converge); inverse-transform
                          predictions for reporting in real ($100k) units
        - metrics: RMSE and R²    (not accuracy/F1)

  2. EARLY STOPPING in the training loop. Instead of always running all `epochs`
     and reporting the FINAL weights, you track the BEST val loss epoch-by-epoch,
     snapshot those weights, and STOP after `patience` epochs of no improvement.
     At the end you RESTORE the snapshotted best weights. This is the canonical
     "real" training loop — no more "did 200 epochs help or hurt?" guesswork.

Time budget: ~2 hours (hard cap). Same as the previous capstone — the additions
are small per piece; the new conceptual bit is keeping a copy.deepcopy of
state_dict and restoring it.

Fill in every function marked `# TODO` (each currently raises
NotImplementedError) and wire them together in main(). When it's working you
should be able to run:

    uv run python capstone_regression_nn.py
    uv run python capstone_regression_nn.py --epochs 300 --patience 20 --hidden 128 64

Target to beat: ~0.79+ test R² (≈$53k RMSE) is very achievable on California Housing
with a small MLP — the same number the notebook achieved with the basic loop.
Early stopping should let you set --epochs generously without worrying about
overfitting.

------------------------------------------------------------------------------
THE RECIPE (your checklist):
    seed -> load -> split -> scale X AND y (fit on train only) ->
    tensors/loaders -> model (1 output, no activation) ->
    train WITH EARLY STOPPING (track best val, save best_state, restore at end) ->
    evaluate on test (inverse-transform predictions; report RMSE & R²) ->
    save model state_dict + x_scaler + y_scaler.

REGRESSION DIFFERENCES vs the classifier capstone:
    - target dtype: float32 (not int64)
    - target shape: (N, 1) so it matches model output (N, 1)
    - loss: nn.MSELoss
    - output layer: nn.Linear(prev, 1) with NO activation
    - metrics: from sklearn.metrics import root_mean_squared_error, r2_score
               (root_mean_squared_error exists in sklearn 1.4+; alternative is
                np.sqrt(mean_squared_error(...)))
    - target scaling: fit a SECOND StandardScaler on y_train.reshape(-1, 1).
                      Transform y_va, y_te with the SAME scaler. At eval time,
                      inverse_transform predictions back to $100k units for
                      reporting.
    - save TWO scalers (x_scaler + y_scaler) plus the model state_dict.

EARLY STOPPING (the new training-loop technique):
    - Track best val_loss across epochs (init: float("inf")).
    - When val_loss < best_val: best_val = val_loss; best_state =
      copy.deepcopy(model.state_dict()); patience_counter = 0.
    - Else: patience_counter += 1.
    - If patience_counter >= patience: stop (break the epoch loop).
    - After the loop: model.load_state_dict(best_state) so you return the
      BEST model, not the last one.
    - copy.deepcopy is essential — state_dict() returns references to the live
      tensors, so without deepcopy your "best_state" mutates with the next step.

EXTENSION CHALLENGES (optional, beyond 2h):
    1. Add an LR scheduler (torch.optim.lr_scheduler.ReduceLROnPlateau on
       val_loss); compare convergence with/without.
    2. Add nn.Dropout between hidden layers + weight_decay to Adam; eyeball
       whether it tightens the train/val gap.
    3. Plot the loss curves (matplotlib) and the early-stopping epoch as a
       vertical line.
    4. Swap dataset to load_diabetes (442 samples, smaller); same code should
       run with minor argparse defaults.
    5. Add Huber loss (nn.HuberLoss) and compare to MSE — Huber is robust to
       outliers in target (Calf Housing has a hard cap at 5.0 that's worth
       discussing).
------------------------------------------------------------------------------
"""
import argparse
import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error, r2_score

@dataclass
class DataSet:
    X_tr: np.ndarray
    y_tr_scaled: np.ndarray
    X_va: np.ndarray
    y_va_scaled: np.ndarray
    X_te: np.ndarray
    y_te_scaled: np.ndarray

def set_seed(seed: int) -> None:
    """Make the run reproducible across numpy and torch."""
    # TODO: seed numpy and torch (same as the previous capstone).
    np.random.seed(seed); torch.manual_seed(seed)


def load_data(seed: int):
    """Load California Housing, split into train/val/test, and scale X AND y.

    Returns: (X_tr, y_tr, X_va, y_va, X_te, y_te), x_scaler, y_scaler, n_features

    Requirements:
      - Use fetch_california_housing(return_X_y=True). Shapes: X (20640, 8), y (20640,).
      - Split into THREE sets with train_test_split (no stratify — y is continuous).
        Same 80/20 then 80/20 pattern as before.
      - Fit a StandardScaler on TRAINING X only → transform train, val, test.
      - **Also** fit a second StandardScaler on y_train (reshape to (-1, 1) first
        because StandardScaler wants 2D input). Use it to transform y_va and y_te.
        Keep this scaler around — main() will save it; evaluate() will use it to
        inverse_transform predictions back to $100k units.
    """
    # TODO: implement per the docstring.
    X, y = fetch_california_housing(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed)
    X_tr, X_va, y_tr, y_va = train_test_split(X_tr, y_tr, test_size=0.2, random_state=seed)
    
    sc_X = StandardScaler().fit(X_tr)
    X_tr, X_va, X_te = sc_X.transform(X_tr), sc_X.transform(X_va), sc_X.transform(X_te)
    
    sc_y = StandardScaler().fit(y_tr.reshape(-1, 1)) # standard scaler wants 2D input
    y_tr_scaled = sc_y.transform(y_tr.reshape(-1, 1))
    y_va_scaled = sc_y.transform(y_va.reshape(-1, 1))
    y_te_scaled = sc_y.transform(y_te.reshape(-1, 1))
    
    ds = DataSet(X_tr=X_tr, y_tr_scaled=y_tr_scaled, X_va=X_va, y_va_scaled=y_va_scaled, X_te=X_te, y_te_scaled=y_te_scaled)
    
    return ds, sc_X, sc_y, X.shape[1]

def make_loaders(X_tr, y_tr_scaled, X_va, y_va_scaled, batch_size: int):
    """Wrap arrays in tensors and DataLoaders.

    Reminders (DIFFERENT from the classification capstone!):
      - Features X → float32 tensors, shape (N, 8).
      - Targets y → **float32** tensors (not int64!) and shape **(N, 1)** so that
        the model's (N, 1) output and y line up element-wise — MSELoss expects
        matching shapes.
        Either pass y as (N, 1) numpy then to_tensor, or reshape after wrapping:
        torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
      - shuffle=True for the train loader, shuffle=False for validation.
    Returns: train_loader, val_loader
    """
    # TODO: implement per the docstring.
    ds_tr = TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr_scaled, dtype=torch.float32))
    ds_va = TensorDataset(torch.tensor(X_va, dtype=torch.float32), torch.tensor(y_va_scaled, dtype=torch.float32))

    train_loader = DataLoader(ds_tr, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(ds_va, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def build_model(n_features: int, hidden: list[int]) -> nn.Module:
    """MLP: [n_features] -> hidden... -> [1] (no activation on output).

    Same pattern as the classifier — only TWO things change:
      - Final layer: nn.Linear(prev, 1) (not n_classes).
      - NO activation on the output: regression should be able to predict any
        real number (positive or negative).
    """
    # TODO: build and return the model.
    layers = []
    prev = n_features
    for h in hidden:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.ReLU())
        prev = h
    layers.append(nn.Linear(prev, 1))
    
    model = nn.Sequential(*layers)
    
    return model


def train_with_early_stopping(
    model,
    train_loader,
    val_loader,
    loss_fn,
    optimizer,
    epochs: int,
    device,
    patience: int = 10,
):
    """Training loop with EARLY STOPPING and best-weight restoration.

    The train/val pass each epoch is identical to the classifier capstone.
    What's new is the bookkeeping AROUND the loop:

    Setup (before the epoch loop):
        best_val = float("inf")
        best_state = None
        epochs_without_improvement = 0
        history = {"train": [], "val": []}

    Each epoch (after both passes):
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())   # SNAPSHOT
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch + 1} (best val {best_val:.4f})")
            break

    After the loop:
        if best_state is not None:
            model.load_state_dict(best_state)                # RESTORE best weights

    Returns: history dict with 'train' and 'val' per-epoch lists.

    WHY copy.deepcopy: state_dict() returns a dict of references to the live
    parameter tensors. Without deepcopy, "best_state" would silently mutate every
    optimizer.step(). With deepcopy, you freeze a true snapshot in memory.
    """
    # TODO: implement the loop (re-use the inner train/val pass pattern from
    # capstone_tabular_nn.py and ADD the early-stopping bookkeeping above).
    best_val = float("inf")
    best_state = None
    epochs_without_improvement = 0
    history = {"train": [], "val": []}
    
    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()                   # reset gradient
            loss = loss_fn(model(xb), yb)           # loss
            loss.backward()                         # backpropagate to compute gradients
            optimizer.step()                        # update weights + biases
            running_train_loss += loss.item() * xb.size(0)
        train_loss = running_train_loss / len(train_loader.dataset)
        
        model.eval()
        with torch.no_grad():
            running_val_loss = 0.0
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                loss = loss_fn(model(xb), yb)
                running_val_loss += loss.item() * xb.size(0)
            val_loss = running_val_loss / len(val_loader.dataset)
            
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict()) # Model SNAPSHOT
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            
        history["train"].append(train_loss)
        history["val"].append(val_loss)
        
        if (epoch + 1) % max(1, epochs // 10) == 0:
            print(f"epoch {epoch+1:3d} | train {train_loss:.4f} | val {val_loss:.4f}")
            
        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch + 1} (best val {best_val:.4f})")
            break
    
        
        
    if best_state is not None:
        model.load_state_dict(best_state) # Model RESTORE best weights
    
    return history
                

def evaluate(model, X_te, y_te_scaled, y_scaler, device) -> tuple[float, float]:
    """Predict on the test set; inverse-transform; report RMSE and R² in $100k units.

    Returns: (rmse, r2)

    Reminders:
      - model.eval() + torch.no_grad() (or torch.inference_mode()).
      - Predictions come out in **scaled** space — apply
        y_scaler.inverse_transform(pred.reshape(-1, 1)) to get back to $100k.
        Then ravel() / flatten() for sklearn metrics.
      - Same trick for y_te_scaled — invert it before comparison.
      - root_mean_squared_error(y_true, y_pred)   # sklearn 1.4+
      - r2_score(y_true, y_pred)
      - Print them with reasonable precision; main() will save the model + scalers.
    """
    # TODO: implement per the docstring.
    model.eval()
    with torch.no_grad():
        X_te_t = torch.tensor(X_te, dtype=torch.float32).to(device)
        y_pred = y_scaler.inverse_transform(model(X_te_t).cpu().numpy().reshape(-1, 1)).ravel()
        y_true = y_scaler.inverse_transform(y_te_scaled).reshape(-1, 1).ravel()
        rmse = root_mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
    print(f"rmse: {rmse:.4f} | r2: {r2:.4f}")
    
    return rmse, r2


def main():
    p = argparse.ArgumentParser(description="MLP regression on California Housing with early stopping.")
    p.add_argument("--epochs", type=int, default=200,
                   help="upper bound on epochs; early stopping will usually stop sooner")
    p.add_argument("--patience", type=int, default=10,
                   help="how many epochs without val improvement before we stop")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--hidden", type=int, nargs="+", default=[64, 32],
                   help="hidden layer widths, e.g. --hidden 128 64")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="cal_housing_model.pt")
    args = p.parse_args()

    # TODO: wire the pieces together (mirrors the previous capstone but with two
    # extra artifacts to save at the end — there are now TWO scalers):
    #   1. set_seed(args.seed)
    #   2. device = cuda if available else cpu
    #   3. (X_tr, y_tr, X_va, y_va, X_te, y_te), x_scaler, y_scaler, n_features
    #          = load_data(args.seed)
    #   4. train/val loaders via make_loaders(...)
    #   5. model = build_model(n_features, args.hidden).to(device)
    #   6. loss_fn = nn.MSELoss()  # <-- NOT CrossEntropy
    #      optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    #   7. history = train_with_early_stopping(..., patience=args.patience)
    #   8. rmse, r2 = evaluate(model, X_te, y_te, y_scaler, device); print them
    #   9. Save three artifacts (you need ALL THREE to predict on new raw data
    #      later — x_scaler to normalise inputs; y_scaler to un-normalise outputs):
    #          torch.save(model.state_dict(), args.out)
    #          joblib.dump(x_scaler, args.out.replace(".pth", "_xscaler.joblib"))
    #          joblib.dump(y_scaler, args.out.replace(".pth", "_yscaler.joblib"))
    
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data
    ds, x_scaler, y_scaler, n_features = load_data(args.seed)
    
    # Build data loader
    train_loader, va_loader = make_loaders(ds.X_tr, ds.y_tr_scaled, ds.X_va, ds.y_va_scaled, args.batch_size)

    # Build model
    model = build_model(n_features, args.hidden).to(device)
    
    # Optimizer and loss fn
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Train model
    history = train_with_early_stopping(model, train_loader, va_loader, loss_fn, optimizer, args.epochs, device)

    # Evaluate on test set
    rmse, r2 = evaluate(model, ds.X_te, ds.y_te_scaled, y_scaler, device)
    
    # Save model + scalers
    torch.save(model.state_dict(), args.out)
    joblib.dump(x_scaler, args.out.replace(".pt", "_x_scaler.joblib"))
    joblib.dump(y_scaler, args.out.replace(".pt", "_y_scaler.joblib"))
    
    print(f"model saved here: {args.out}\nscalers saved here: {args.out.replace('.pt', '_x_scaler.joblib')} and here: {args.out.replace('.pt', '_y_scaler.joblib')}")

if __name__ == "__main__":
    main()
