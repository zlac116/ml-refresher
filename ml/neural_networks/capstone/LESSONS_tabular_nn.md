# Tabular NN (Wine multiclass) — Lessons

Reflection on what *this specific capstone* taught me — multiclass
classification on Wine (3 classes, 13 features). For the generic
phase-by-phase playbook see
[`toolkit/ml_project_methodology.md`](../../../toolkit/ml_project_methodology.md).

---

## 1. CrossEntropyLoss eats raw logits, not probabilities

```python
nn.Linear(prev, n_classes)        # output: 3 logits, NO activation
loss = nn.CrossEntropyLoss()(logits, labels)   # softmax is INSIDE
```

Softmax is *baked into* `CrossEntropyLoss`. If I add `nn.Softmax()` to the
output, I'm softmax-ing twice → gradients go weird, training is silently
slower. Lesson: **for multiclass, the model returns logits, the loss
handles the rest.**

> **Binary classification analogue**: use `BCEWithLogitsLoss` (raw logit
> in, sigmoid inside). Same rule — never add a final activation when the
> loss has one baked in.

## 2. Multiclass labels are int64 of shape `(N,)`, not one-hot

```python
yb = torch.tensor(y_tr, dtype=torch.int64)   # shape (N,), values in {0, 1, 2}
```

Common mistakes I almost made:
- one-hot encoded `(N, 3)` — CrossEntropy doesn't want this
- `float32` labels — runtime error, CrossEntropy requires `long`
- shape `(N, 1)` — CrossEntropy wants `(N,)`

> **Mental prompt**: *"What dtype + shape does this specific loss expect?"*
> Get it wrong, get a cryptic CUDA stack trace.

## 3. StandardScaler fit on TRAIN only

```python
sc = StandardScaler().fit(X_tr)              # fit ONLY on train
X_tr, X_va, X_te = sc.transform(X_tr), sc.transform(X_va), sc.transform(X_te)
```

**Leakage = the silent killer.** If I fit the scaler on the whole dataset
(or train+val), the val/test statistics leak into the training
distribution and metrics look better than they really are. Fit on train,
transform everything else. Always.

Same rule applies to any fitted preprocessor (PCA, KNN imputer, target
encoder, …).

## 4. Stratified splits for class-imbalanced data

```python
train_test_split(X, y, stratify=y, ...)
```

Without `stratify=y`, a small dataset (n=178 in Wine) can produce a val
set with zero examples of one class — accuracy becomes meaningless.
Stratify whenever classes are uneven OR the dataset is small.

## 5. `argmax(dim=1)` for predictions

```python
y_pred = model(Xt).argmax(dim=1).cpu().numpy()   # then to sklearn
```

Then `.cpu().numpy()` at the boundary so sklearn metrics can read it.
Forgetting either gives weird errors (`expected long`, `expected on CPU`).

## 6. Save the model AND the scaler

```python
torch.save(model.state_dict(), "wine_model.pt")
joblib.dump(scaler, "wine_model_scaler.joblib")
```

The model alone is **useless on raw inputs** — it expects standardised
features. Without the scaler, every prediction on new data is garbage.
The scaler is just as important as the weights.

> **General rule**: save every fitted thing that touches the input
> pipeline. Forget one, the model is unservable.

## 7. Mini-batches via DataLoader, not full-batch

Different from the LMM surrogate where I used full-batch tensors. Here,
Wine has labels that need shuffling (otherwise the model sees class 0,
then 1, then 2 in long runs → noisy gradients). DataLoader with
`shuffle=True` is the right choice for classifiers.

> **Decision rule**:
> - **Full-batch**: surrogates/regressors, small data fits in memory, want
>   max convergence speed.
> - **DataLoader + shuffle**: classifiers, large data, or any case where
>   ordering matters and shuffling helps.

## 8. Accuracy AND macro F1

```python
acc = accuracy_score(y_te, y_pred)
f1  = f1_score(y_te, y_pred, average="macro")
```

Macro F1 is the average of per-class F1s — exposes whether one class is
silently doing all the heavy lifting. If accuracy is high but macro F1 is
low → class imbalance / weak class. **Two numbers, two views.**

## 9. Bugs I hit (so I don't repeat them)

- `nn.Linear(h, 1)` instead of `nn.Linear(prev, n_classes)` — final layer
  width matters; pass the right `prev` from the loop.
- Missing `.cpu().numpy()` before sklearn metrics → tensor-on-GPU error.
- Model not `.to(device)` in `main` — fine on CPU, fails on CUDA.
- `y` as `float32` for CrossEntropy → "expected long" runtime error.

---

For the generic phase-by-phase playbook, see
[`toolkit/ml_project_methodology.md`](../../../toolkit/ml_project_methodology.md).
For early stopping (didn't use it here), see
[`LESSONS_regression_nn.md`](LESSONS_regression_nn.md).
