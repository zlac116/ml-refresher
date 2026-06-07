# Regression NN (California Housing + early stopping) — Lessons (code-first)

Canonical patterns from this build. Each section: code → why → trap.
For phase-by-phase generic playbook: [`toolkit/ml_project_methodology.md`](../../../toolkit/ml_project_methodology.md).

---

## 1. Load + split (no stratify) + scale X **and** y

```python
X, y = fetch_california_housing(return_X_y=True)                      # (20640, 8), (20640,)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed)
X_tr, X_va, y_tr, y_va = train_test_split(X_tr, y_tr, test_size=0.2, random_state=seed)

sc_X = StandardScaler().fit(X_tr)
X_tr, X_va, X_te = sc_X.transform(X_tr), sc_X.transform(X_va), sc_X.transform(X_te)

sc_y = StandardScaler().fit(y_tr.reshape(-1, 1))                      # 2D required
y_tr_s = sc_y.transform(y_tr.reshape(-1, 1))                          # shape (N, 1)
y_va_s = sc_y.transform(y_va.reshape(-1, 1))
y_te_s = sc_y.transform(y_te.reshape(-1, 1))
```

**Why**: scaling y normalises gradient magnitudes (median house value is
~$200k — unscaled MSE on that scale needs absurdly small LRs). Standard
`lr=1e-3` works once both X and y are unit-variance.
**Trap**: `StandardScaler.fit(y_tr)` (1D) → ValueError. Always `.reshape(-1, 1)`.

---

## 2. Tensors + loaders — `float32` y of shape `(N, 1)`

```python
ds_tr = TensorDataset(
    torch.tensor(X_tr, dtype=torch.float32),
    torch.tensor(y_tr_s, dtype=torch.float32),         # float32, NOT int64
)
train_loader = DataLoader(ds_tr, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(ds_va, batch_size=batch_size, shuffle=False)
```

**Why**: regression target = float, not class index. Shape `(N, 1)` so it
matches the model's `(N, 1)` output — MSE compares like-shapes.
**Trap**: y as `(N,)` while model outputs `(N, 1)` → MSE broadcasts to
`(N, N)`, trains on garbage, no error.

---

## 3. MLP — 1 output, **no** activation

```python
def build_model(n_features, hidden):
    layers, prev = [], n_features
    for h in hidden:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.ReLU())
        prev = h
    layers.append(nn.Linear(prev, 1))                 # 1 output, no final activation
    return nn.Sequential(*layers)
```

**Why**: regression target can be any real number → no sigmoid/softmax
cap. Bare linear output.
**Trap**: `nn.Sigmoid()` on the output → predictions stuck in `(0, 1)`.

---

## 4. Early stopping — three pieces of state + `copy.deepcopy`

```python
best_val, best_state, patience_used = float("inf"), None, 0
history = {"train": [], "val": []}

for epoch in range(epochs):
    # ... train + val pass identical to classifier ...
    history["train"].append(train_loss); history["val"].append(val_loss)

    if val_loss < best_val:                                  # ← < not >
        best_val   = val_loss
        best_state = copy.deepcopy(model.state_dict())       # SNAPSHOT (deepcopy!)
        patience_used = 0
    else:
        patience_used += 1
    if patience_used >= patience:
        print(f"Early stop @ epoch {epoch+1} (best val {best_val:.4f})")
        break

if best_state is not None:
    model.load_state_dict(best_state)                        # RESTORE best
```

**Why**: `state_dict()` returns *references* to live tensors — without
`deepcopy`, "best_state" silently mutates with every `opt.step()` so
"restoring" does nothing. `deepcopy` freezes a true copy in memory.
**Trap**: writing `val_loss > best_val` (wrong direction) → snapshots the
WORST weights → mysterious bad results.

---

## 5. Evaluate — inverse-transform predictions AND targets

```python
model.eval()
with torch.no_grad():
    Xt = torch.tensor(X_te, dtype=torch.float32).to(device)
    y_pred_scaled = model(Xt).cpu().numpy()                          # (N, 1) scaled
y_pred = sc_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()  # $100k units
y_true = sc_y.inverse_transform(y_te_s).ravel()
rmse = root_mean_squared_error(y_true, y_pred)                       # sklearn >= 1.4
r2   = r2_score(y_true, y_pred)
print(f"RMSE {rmse:.4f} | R² {r2:.4f}")
```

**Why**: predictions come out in *scaled* space (zero-mean, unit-variance).
Inverse-transform BOTH `y_pred` and `y_true` before metrics so RMSE is in
real units ($100k), not "sigmas of house price".
**Trap**: reporting RMSE in scaled space → unitless number that says nothing.

---

## 6. Two metrics — RMSE and R²

| Metric | Question it answers |
|---|---|
| `root_mean_squared_error` | "typical error in target units" → ~$50k on average |
| `r2_score`                | "fraction of variance explained" → 0.79 = 79% |

**Why**: RMSE for absolute size, R² for relative informativeness. Report
both — neither alone tells the full story.
**Trap**: `root_mean_squared_error` exists only in sklearn >= 1.4. Older:
`np.sqrt(mean_squared_error(y_true, y_pred))`.

---

## 7. Save THREE artifacts (model + x_scaler + y_scaler)

```python
torch.save(model.state_dict(), "cal_housing_model.pt")
joblib.dump(sc_X, "cal_housing_model_x_scaler.joblib")
joblib.dump(sc_y, "cal_housing_model_y_scaler.joblib")            # ← the extra one
```

**Why**: to predict on a new raw row I need (1) x_scaler to standardise
inputs, (2) model to predict in scaled space, (3) y_scaler to
inverse-transform back to dollars. Forget any one → wrong predictions, no
error.
**Trap**: this regressor has TWO scalers — classifier-style "model +
scaler" is incomplete.

---

## 8. The wiring (`main()`)

```python
set_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ds, sc_X, sc_y, n_features = load_data(args.seed)
tr_loader, va_loader = make_loaders(ds.X_tr, ds.y_tr_scaled, ds.X_va, ds.y_va_scaled, args.batch_size)
model = build_model(n_features, args.hidden).to(device)
loss_fn = nn.MSELoss()                                            # NOT CrossEntropy
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
history = train_with_early_stopping(
    model, tr_loader, va_loader, loss_fn, optimizer,
    args.epochs, device, patience=args.patience,
)
rmse, r2 = evaluate(model, ds.X_te, ds.y_te_scaled, sc_y, device)
torch.save(model.state_dict(), args.out)
joblib.dump(sc_X, args.out.replace(".pt", "_x_scaler.joblib"))
joblib.dump(sc_y, args.out.replace(".pt", "_y_scaler.joblib"))
```

**Why**: same top-to-bottom skeleton as the classifier, with two
substitutions: `MSELoss` for `CrossEntropyLoss`, and **save three
artifacts** for two.
**Trap**: `.pth` vs `.pt` — PyTorch docs use `.pt`; stick with it.

---

## Bugs I hit (so I don't repeat them)

- `val_loss > best_val` (wrong direction) → snapshotted WORST weights
- `loss.backward()` missing → loss never updates, training "flatlines"
- `best_state = model.state_dict()` (no deepcopy) → snapshot mutates with
  every `opt.step()`
- `breakpoint()` left in committed code → CI freezes
- `.pth` instead of `.pt` → works but not canonical
- Model not `.to(device)` in `main` → CUDA error
- `StandardScaler().fit(y_tr)` without `.reshape(-1, 1)` → ValueError

---

Related: [`LESSONS_tabular_nn.md`](LESSONS_tabular_nn.md) ·
[`LESSONS.md` (LMM)](../../../quant_finance/capstones/lmm_nn_surrogate/LESSONS.md) ·
[`toolkit/ml_project_methodology.md`](../../../toolkit/ml_project_methodology.md)
