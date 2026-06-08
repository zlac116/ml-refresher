# Neural Networks Cheatsheet (PyTorch)

The 9-step experiment recipe, condensed. See `neural_networks.ipynb` for the
explained version.

## Recipe

1. Setup → 2. Split → 3. Scale → 4. Tensors/Loader → 5. Model →
6. Loss+Optimizer → 7. Train loop → 8. Evaluate on test → 9. Diagnose.

## 1. Setup
```python
np.random.seed(42); torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

## 2. Split (3 sets)
```python
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42,
                                           stratify=y)   # stratify for classification
X_tr, X_va, y_tr, y_va = train_test_split(X_tr, y_tr, test_size=0.2, random_state=42)
```
Train learns params · val tunes hyperparams · test = touched once.

## 3. Scale — fit on TRAIN only (else leakage)
```python
sc = StandardScaler().fit(X_tr)
X_tr, X_va, X_te = sc.transform(X_tr), sc.transform(X_va), sc.transform(X_te)
# regression: scale y too, invert before reporting.
```

## 4. Tensors + DataLoader
```python
ds = TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                   torch.tensor(y_tr, dtype=torch.float32))  # long for CrossEntropy
loader = DataLoader(ds, batch_size=32, shuffle=True)         # shuffle train only
```

## 5. Model (MLP) — output set by task
```python
nn.Sequential(nn.Linear(in, 64), nn.ReLU(), nn.Linear(64, OUT))
```
ReLU between layers (without it, the net collapses to one linear map).

## 6. Loss + optimizer — what changes per task

| Task | Output layer | Loss | Label dtype | Metrics |
|------|-------------|------|-------------|---------|
| Regression | `1`, no activation | `MSELoss` | float `(N,1)` | RMSE, R² |
| Binary clf | `1` logit, no sigmoid | `BCEWithLogitsLoss` | float `(N,1)` | acc, ROC AUC |
| Multiclass | `n_classes` logits | `CrossEntropyLoss` | long `(N,)` | acc, macro-F1 |

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # safe default
```
Sigmoid/softmax live **inside** the loss — never put them in the model.

## 7. Training loop
```python
for epoch in range(epochs):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()        # gradients accumulate — must clear
        loss = loss_fn(model(xb), yb)
        loss.backward()              # backprop
        optimizer.step()             # update weights
    model.eval()
    with torch.no_grad():            # no grad tracking in validation
        ...                          # compute val loss
```

## 8. Evaluate (test, once)
```python
model.eval()
with torch.no_grad():
    out = model(torch.tensor(X_te, dtype=torch.float32).to(device)).cpu()
# regression: y_scaler.inverse_transform(out); RMSE/R2
# binary:     probs = torch.sigmoid(out); preds = (probs >= .5)
# multiclass: preds = out.argmax(1)
```

## 9. Diagnose
- **Loss curve:** both fall+flatten = ok · val rises while train falls = overfit ·
  both high = underfit.
- **Pred vs actual** (regression) / **ROC curve** (classification).

## Pitfalls
- Scaler fit on all data (leakage) · peeking at test during tuning ·
  forgetting `zero_grad()` · double sigmoid · reporting in scaled units ·
  judging imbalanced classifiers by accuracy (use ROC AUC / F1).

## Hyperparameter quick guide
- **lr** is the #1 knob: too high diverges, too low crawls. Start `1e-3`.
- Bigger/more layers → more capacity (and more overfit risk).
- Regularize: `nn.Dropout`, `weight_decay` in Adam, early stopping.
