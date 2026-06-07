# Regression NN (California Housing + early stopping) — Lessons

Reflection on what *this specific capstone* taught me — regression on
California Housing (8 features → 1 continuous target) with early stopping
and best-weight restoration. For the generic phase-by-phase playbook see
[`toolkit/ml_project_methodology.md`](../../../toolkit/ml_project_methodology.md).

---

## 1. Early stopping is *bookkeeping around the loop*, not a new loop

```python
best_val, best_state, patience_used = float("inf"), None, 0

for epoch in range(epochs):
    # ... train + val pass exactly like the classifier ...
    if val_loss < best_val:
        best_val   = val_loss
        best_state = copy.deepcopy(model.state_dict())   # SNAPSHOT
        patience_used = 0
    else:
        patience_used += 1
    if patience_used >= patience:
        break                                            # STOP

if best_state is not None:
    model.load_state_dict(best_state)                    # RESTORE best
```

The train/val pass itself is identical to the classifier. What's new is
**three pieces of state**: `best_val`, `best_state`, `patience_used`.
Internalise this triple — it's the same shape for any early-stopping
scenario (LR scheduling, hyperparameter search, etc).

## 2. `copy.deepcopy(model.state_dict())` — the non-obvious bit

```python
best_state = model.state_dict()           # ⚠️  shares tensor references
best_state = copy.deepcopy(model.state_dict())   # ✅ true frozen snapshot
```

`state_dict()` returns a **dict of references** to the live parameter
tensors. Without `deepcopy`, `best_state` silently mutates on every
`optimizer.step()` — "best weights" becomes "latest weights" → restoring
does nothing. With `deepcopy`, you freeze a copy in memory.

> **General rule**: any time I want to keep a "snapshot" of mutable state,
> assume Python passes references unless I deepcopy. PyTorch tensors
> aren't special here — same rule as for lists, dicts, dataclasses.

## 3. The `<` not `>` direction matters

```python
if val_loss < best_val:    # ✅ lower is better
    best_val = val_loss
```

`>` here = "save when val_loss goes UP" → snapshot the worst, restore the
worst. Hours of mysterious bad results from a single inverted operator.
**Whenever I write a comparison against a "best", I now read it back as
English first**: "if this is better than current best — then save".

## 4. Scale y too — but keep the y-scaler around

```python
sc_X = StandardScaler().fit(X_tr)
sc_y = StandardScaler().fit(y_tr.reshape(-1, 1))   # 2D input required

y_tr_scaled = sc_y.transform(y_tr.reshape(-1, 1))
# ... train on scaled targets ...
# ... at eval time: ...
y_pred_real = sc_y.inverse_transform(model(X).cpu().numpy().reshape(-1, 1)).ravel()
y_true_real = sc_y.inverse_transform(y_te_scaled).ravel()
rmse = root_mean_squared_error(y_true_real, y_pred_real)
```

Why scale `y`: MSE on unscaled targets (median house value in $100k)
has huge magnitudes → optimiser needs tiny LRs to not blow up. Scaling
both X and y normalises gradient magnitudes → standard `lr=1e-3` works.

**Crucially: report metrics in real units**, not scaled space. RMSE in
"sigmas of house price" is meaningless. `inverse_transform` predictions
AND targets before computing the metric.

## 5. Target shape `(N, 1)` to match model output

```python
output = nn.Linear(prev, 1)          # model output: (N, 1)
y_t    = torch.tensor(y_scaled, dtype=torch.float32)   # already (N, 1) since sklearn produced it that way
```

If `y_t` is `(N,)` and model output is `(N, 1)`, MSE broadcasts to
`(N, N)` → trains on garbage, no error. Same `.squeeze(-1)` / `(-1, 1)`
shape vigilance as the LMM capstone. **Always sanity-check what the loss
compares.**

## 6. RMSE + R² are different questions

```python
rmse = root_mean_squared_error(y_true, y_pred)   # in $100k units
r2   = r2_score(y_true, y_pred)                  # unitless, [0, 1]
```

- **RMSE**: "what's my typical error in target units?" → "my model is off
  by ~$50k on average". Reports in absolute terms.
- **R²**: "what fraction of target variance does my model explain?"
  → "0.79 means I explain 79% of the variance". Comparable across
  datasets and models.

Report both. RMSE for "how bad", R² for "how informative".

## 7. Save THREE artifacts, not two

```python
torch.save(model.state_dict(), "cal_housing_model.pt")
joblib.dump(x_scaler, "cal_housing_model_x_scaler.joblib")
joblib.dump(y_scaler, "cal_housing_model_y_scaler.joblib")   # ← extra one
```

To predict on a new raw `(8,)` row I need:
1. `x_scaler` to standardise the inputs,
2. the model to predict in scaled space,
3. `y_scaler` to inverse-transform the prediction back to dollars.

Forget any one → the prediction is wrong, no error. **The rule from
the classifier (save what fits the input pipeline) extends to whatever
fits the OUTPUT pipeline too.**

## 8. `root_mean_squared_error` lives in sklearn 1.4+

```python
from sklearn.metrics import root_mean_squared_error    # sklearn >= 1.4
```

For older sklearn: `np.sqrt(mean_squared_error(...))`. Don't lose 10
minutes to a missing import — check `sklearn.__version__` first.

## 9. Bugs I hit (so I don't repeat them)

- `loss.backward()` missing from training loop → loss never updates,
  loss stays flat → I blame the model, the bug is upstream.
- Inverted `val_loss > best_val` → snapshotted the WORST weights.
- `breakpoint()` left in committed code → CI freezes.
- `.pth` vs `.pt`: **`.pt` is the canonical PyTorch extension.** `.pth`
  works but isn't the documented convention.
- Model not `.to(device)` in `main` — same gotcha as the classifier.

---

For the generic phase-by-phase playbook, see
[`toolkit/ml_project_methodology.md`](../../../toolkit/ml_project_methodology.md).
For the multiclass/CrossEntropy specifics, see
[`LESSONS_tabular_nn.md`](LESSONS_tabular_nn.md).
