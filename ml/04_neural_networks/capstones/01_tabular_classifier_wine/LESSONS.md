# Tabular NN (Wine multiclass) — Lessons (code-first)

Canonical patterns from this build. Each section: code → why → trap.
For phase-by-phase generic playbook: [`toolkit/ml_project_methodology.md`](../../../../toolkit/ml_project_methodology.md).

---

## 1. Load + stratified split + scaler-fit-on-train

```python
X, y = load_wine(return_X_y=True)                                 # (178, 13)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=seed, stratify=y)            # stratify on labels
X_tr, X_va, y_tr, y_va = train_test_split(
    X_tr, y_tr, test_size=0.2, random_state=seed, stratify=y_tr)

sc = StandardScaler().fit(X_tr)                                    # fit ONLY on train
X_tr, X_va, X_te = sc.transform(X_tr), sc.transform(X_va), sc.transform(X_te)
```

**Why**: `stratify=y` keeps class ratios consistent across splits — vital
when n is small (178) or classes are uneven. Fit scaler on **train only**
to avoid leaking val/test statistics into training distribution.
**Trap**: `StandardScaler().fit(X)` on the whole array → silent
leakage, val accuracy looks better than reality.

---

## 2. Tensors + DataLoader (`int64` labels, `(N,)` shape)

```python
ds_tr = TensorDataset(
    torch.tensor(X_tr, dtype=torch.float32),
    torch.tensor(y_tr, dtype=torch.int64),        # CrossEntropy wants long, NOT one-hot
)
ds_va = TensorDataset(
    torch.tensor(X_va, dtype=torch.float32),
    torch.tensor(y_va, dtype=torch.int64),
)
train_loader = DataLoader(ds_tr, batch_size=batch_size, shuffle=True)   # shuffle on train
val_loader   = DataLoader(ds_va, batch_size=batch_size, shuffle=False)  # no shuffle on val
```

**Why**: multiclass labels are **class indices** of shape `(N,)`, not one-hot
`(N, n_classes)`. CrossEntropyLoss requires `int64` (a.k.a. `long`).
Shuffle train so the model doesn't see "all class 0, then all class 1".
**Trap**: `float32` labels → "expected long" runtime error.

---

## 3. MLP factory — `n_classes` logits, no output activation

```python
def build_model(n_features, n_classes, hidden):
    layers, prev = [], n_features
    for h in hidden:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.ReLU())
        prev = h
    layers.append(nn.Linear(prev, n_classes))     # raw logits, NO softmax here
    return nn.Sequential(*layers)
```

**Why**: `CrossEntropyLoss` bakes softmax in. Adding `nn.Softmax()` on the
output → softmax twice → gradients warp, training silently degrades.
**Trap**: `nn.Linear(h, 1)` instead of `nn.Linear(prev, n_classes)` — uses
last-iteration `h` value instead of width-tracker `prev`.

---

## 4. Train loop — running loss, eval-no_grad val pass

```python
opt     = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.CrossEntropyLoss()
history = {"train": [], "val": []}

for epoch in range(epochs):
    model.train()
    running = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        opt.step()
        running += loss.item() * xb.size(0)                # weight by batch size
    train_loss = running / len(train_loader.dataset)

    model.eval()
    running_val = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            running_val += loss_fn(model(xb), yb).item() * xb.size(0)
    val_loss = running_val / len(val_loader.dataset)

    history["train"].append(train_loss); history["val"].append(val_loss)
```

**Why**: `running += loss * batch_size` then divide by dataset size = true
epoch-average loss (last batch may be smaller). `model.train()/eval()`
toggle for dropout/BN even if you don't use them yet — habit.
**Trap**: `running += loss.item()` (no `* batch_size`) only works if every
batch is the same size.

---

## 5. Evaluate — `argmax(dim=1)` then `.cpu().numpy()` for sklearn

```python
model.eval()
with torch.no_grad():
    Xt = torch.tensor(X_te, dtype=torch.float32).to(device)
    y_pred = model(Xt).argmax(dim=1).cpu().numpy()        # logits → class indices → numpy
acc = accuracy_score(y_te, y_pred)
f1  = f1_score(y_te, y_pred, average="macro")
print(f"acc {acc:.4f} | macro F1 {f1:.4f}")
```

**Why**: `argmax(dim=1)` picks the class with the highest logit per row
(equivalent to argmax over softmax since softmax is monotonic). Macro F1
is the average of per-class F1s — exposes whether one class is silently
weak.
**Trap**: forget `.cpu().numpy()` → sklearn gets a CUDA tensor and errors.

---

## 6. Save — model state_dict + scaler (both required for inference)

```python
torch.save(model.state_dict(), "wine_model.pt")
joblib.dump(scaler, "wine_model_scaler.joblib")
```

**Why**: model alone is useless on raw inputs — it expects standardised
features. Without the scaler, every prediction on new data is garbage.
**Trap**: saving only the model. Next person tries `model(raw_X)` →
nonsense predictions, no error.

---

## 7. The wiring (`main()`)

```python
set_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ds, scaler, n_features, n_classes = load_data(args.seed)
tr_loader, va_loader = make_loaders(ds.X_tr, ds.y_tr, ds.X_va, ds.y_va, args.batch_size)
model = build_model(n_features, n_classes, args.hidden).to(device)         # .to(device)!
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
history = train(model, tr_loader, va_loader, loss_fn, optimizer, args.epochs, device)
acc, f1 = evaluate(model, ds.X_te, ds.y_te, device)
torch.save(model.state_dict(), args.out)
joblib.dump(scaler, args.out.replace(".pt", "_scaler.joblib"))
```

**Why**: one-screen top-to-bottom wiring. Same order every project:
seed → device → data → loaders → model → loss/opt → train → eval → save.
**Trap**: `model = build_model(...)` without `.to(device)` — fine on CPU,
"expected all tensors on same device" on GPU.

---

## Bugs I hit (so I don't repeat them)

- `nn.Linear(h, 1)` for output layer → wrong width AND wrong number of
  classes; should be `nn.Linear(prev, n_classes)`
- Missing `.cpu().numpy()` before sklearn metrics → CUDA tensor error
- Model not `.to(device)` in `main` → silent on CPU, error on CUDA
- `y` as `float32` for CrossEntropy → "expected long" runtime error
- `StandardScaler().fit(X)` on the whole dataset → val/test leakage

---

Related: [`LESSONS.md` (regression NN)](../02_regression_california_housing/LESSONS.md) ·
[`LESSONS.md` (LMM)](../../../../quant_finance/capstones/04_lmm_nn_surrogate/LESSONS.md) ·
[`toolkit/ml_project_methodology.md`](../../../../toolkit/ml_project_methodology.md)
