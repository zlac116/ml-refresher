# LMM NN Surrogate — Lessons (code-first)

Canonical patterns from this build. Each section: code → why → trap.
For phase-by-phase generic playbook: [`toolkit/ml_project_methodology.md`](../../../toolkit/ml_project_methodology.md).

---

## 1. Constants at top of file (single source of truth)

```python
LMM_PARAM_LO = np.array([0.10, 0.30, 0.005, 0.10])
LMM_PARAM_HI = np.array([0.25, 0.50, 0.025, 0.50])
T_LO, T_HI   = 0.5, 10.0
F_LO, F_HI   = 0.02, 0.05
LOG_M_LO, LOG_M_HI = -0.3, 0.3
N_FEATURES   = 7
```

**Why**: these bounds are used in three places — *training sampling*,
*calibration `least_squares` bounds*, and *API validators*. One definition.
**Trap**: if you inline-duplicate these in three places, divergence is silent.

---

## 2. Data generation — broadcast for sampling, loop for the function call

```python
def generate_data(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)                       # modern RNG, not legacy
    params = rng.uniform(LMM_PARAM_LO, LMM_PARAM_HI, size=(n, 4))  # column-broadcast
    T      = rng.uniform(T_LO, T_HI, size=n)
    F      = rng.uniform(F_LO, F_HI, size=n)
    K      = F * np.exp(rng.uniform(LOG_M_LO, LOG_M_HI, size=n))   # log-moneyness in fixed range
    ivs    = np.array([mock_lmm_iv(p, t, k, f)
                       for p, t, k, f in zip(params, T, K, F)], dtype=np.float32)
    X      = np.column_stack([params, T, np.log(K/F), F]).astype(np.float32)
    return X, ivs
```

**Why**: vectorise the broadcast (numpy idiom), loop the function calls
(sub-second at n=10k, no point vectorising). Cast to `float32` — matches
torch default + half the memory of `float64`.
**Trap**: `np.random.seed(...)` is the legacy global API. Use `default_rng`.

---

## 3. Train/val split — slice indices, not data

```python
def split_train_val(X, y, val_frac=0.2, seed=0):
    rng  = np.random.default_rng(seed)
    idx  = rng.permutation(len(X))
    n_va = max(1, int(val_frac * len(X)))           # defensive against tiny n
    return X[idx[:-n_va]], y[idx[:-n_va]], X[idx[-n_va:]], y[idx[-n_va:]]
```

**Why**: permuting indices is `O(n)` ints; permuting the data is `O(n × d)`
floats. Order of magnitude faster for the same answer.

---

## 4. Surrogate model — `prev/h` MLP factory, SiLU for smoothness

```python
class Surrogate(nn.Module):
    def __init__(self, d_in: int = N_FEATURES, hidden=(64, 64)):
        super().__init__()
        layers, prev = [], d_in
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.SiLU())              # smooth → clean Jacobians
            prev = h
        layers.append(nn.Linear(prev, 1))         # no output activation
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)            # (N,1) → (N,)
```

**Why**: `SiLU` is C¹ — `scipy.least_squares` uses finite-difference
Jacobians and chokes on `ReLU`'s kink at 0. `.squeeze(-1)` matches label
shape `(N,)`.
**Trap**: forget `.squeeze(-1)` → `(N,1) - (N,)` broadcasts to `(N,N)`
→ trains on garbage, no error.

---

## 5. Full-batch training loop (data fits in memory)

```python
Xt_tr = torch.tensor(X_tr, dtype=torch.float32).to(device)
yt_tr = torch.tensor(y_tr, dtype=torch.float32).to(device)
Xt_va = torch.tensor(X_va, dtype=torch.float32).to(device)
yt_va = torch.tensor(y_va, dtype=torch.float32).to(device)

opt     = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.MSELoss()
history = {"train": [], "val": []}

for epoch in range(epochs):
    model.train()
    loss = loss_fn(model(Xt_tr), yt_tr)
    opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        val_loss = loss_fn(model(Xt_va), yt_va)

    history["train"].append(loss.item())
    history["val"].append(val_loss.item())
```

**Why**: at n ≤ ~50k, materialising all data on device once is faster than
a DataLoader. One step per epoch. `loss.item()` (not `loss`) to avoid
keeping tensor refs in history.
**Trap**: `opt.zero_grad()` MUST come before `backward()` — gradients
accumulate by default.

---

## 6. Inference helper — single forward pass at the scipy boundary

```python
def nn_iv(model, params, instruments, device):
    feats = [[*params, T_, np.log(K_/F_), F_] for T_, K_, F_ in instruments]
    x = torch.tensor(feats, dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        return model(x).cpu().numpy()
```

**Why**: one feature matrix, one forward pass, return numpy for scipy.
Anti-pattern is looping `model(one_instrument)` — wastes thousands of
Python↔tensor boundary crossings inside the calibration loop.
**Trap**: forget `.cpu()` → `scipy.optimize` gets a CUDA tensor and dies.

---

## 7. Calibration — bounds + return the full `res`

```python
res = least_squares(
    fun    = lambda p: nn_iv(model, p, market_instruments, device) - market_ivs,
    x0     = (LMM_PARAM_LO + LMM_PARAM_HI) / 2,
    bounds = (LMM_PARAM_LO, LMM_PARAM_HI),
)
return res                          # NOT res.x — caller wants .cost/.success/.message/.nfev
```

**Why**: NNs extrapolate WILDLY outside training region — pass bounds or
the optimiser drifts into nonsense. `x0` = midpoint of box = neutral
starting point.
**Trap**: returning just `.x` strips the audit fields VC needs (`success`,
`message`, `nfev`).

---

## 8. Verification — three IVs, two orthogonal residuals

```python
iv_nn = nn_iv(model, theta_star, market_instruments, device)
iv_mc = np.array([
    black76_implied_vol(mock_lmm_price(theta_star, T, K, F), F, K, T)
    for (T, K, F) in market_instruments
])
df = pd.DataFrame(market_instruments, columns=["T", "K", "F"])
df["market"]       = market_ivs
df["nn"]           = iv_nn
df["mc"]           = iv_mc
df["calib_bp"]     = (df["market"] - df["mc"]) * 1e4          # "does the LMM fit market?"
df["surrogate_bp"] = (df["nn"]     - df["mc"]) * 1e4          # "did the NN tell the truth at θ*?"
rmse_calib     = float(np.sqrt(np.mean(df["calib_bp"]**2)))
rmse_surrogate = float(np.sqrt(np.mean(df["surrogate_bp"]**2)))
```

**Why**: `IV_NN(θ*) − IV_market` is ≈0 by construction (optimiser drove it
there) — useless as an audit. The truth must come from a different code
path (the MC pricer).
**Trap**: writing `iv_nn - market_ivs` and labelling it "calib" — zero
audit value.

---

## 9. Save — timestamped run dir, one format per consumer

```python
run_dir = Path(out_dir) / datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
run_dir.mkdir(parents=True, exist_ok=True)

torch.save(model.state_dict(), run_dir / "surrogate.pt")
df.to_csv(run_dir / "verify_report.csv", index=False)
pd.DataFrame(history).to_csv(run_dir / "training_history.csv", index_label="epoch")
(run_dir / "run.json").write_text(json.dumps({
    "theta_star": theta_star.tolist(),
    "cost": float(res.cost), "success": bool(res.success),
    "message": res.message,  "nfev": int(res.nfev),
    "hyperparams": {...},    "market_inputs": {...},
}, indent=2))
```

**Why**: `.pt` for framework, `.csv` for humans+pandas, `.json` for
machines+audit. Never `.pkl` (opaque, version-fragile).
**Trap**: `"%Y%m%d%T%H%M%S"` — `%T` is `HH:MM:SS`. Escape as a literal:
`"%Y%m%dT%H%M%S"`.

---

## 10. Display the verify table

```python
print(df.to_string(index=False, formatters={
    "market": "{:.4f}".format,  "nn": "{:.4f}".format,  "mc": "{:.4f}".format,
    "calib_bp": "{:+.1f}".format, "surrogate_bp": "{:+.1f}".format,
}))
print(f"RMSE calib: {rmse_calib:5.1f} bp | RMSE surrogate: {rmse_surrogate:5.1f} bp")
```

**Why**: DataFrame is the single source of truth — print AND CSV come
from it. `{:+.1f}` forces a sign on residuals so a reader sees direction
at a glance.
**Trap**: a parallel `print(f"...")` loop next to a DataFrame is the #1
"display drifted from saved data" bug.

---

## Bugs I hit (so I don't repeat them)

- `"%Y%m%d%T%H%M%S"` → `%T` is `HH:MM:SS`; use literal `T`
- `for (_T, _F, _K) in market_instruments` → tuples are `(T, K, F)`; K/F silently swap in display
- `(iv_nn - market_ivs)` labelled "calib" → that's the optimiser's residual, ≈0 by construction
- `prev = N_FEATURES` instead of `prev = d_in` → factory uses module global, ignores constructor arg
- `history.append(loss)` instead of `loss.item()` → keeps tensor refs alive, memory grows
- `raise NotImplementedError(...)` inside a Pydantic class body → fires at import time, breaks the module

---

Related: [`LESSONS.md` (tabular NN)](../../../ml/04_neural_networks/capstones/01_tabular_classifier_wine/LESSONS.md) ·
[`LESSONS.md` (regression NN)](../../../ml/04_neural_networks/capstones/02_regression_california_housing/LESSONS.md) ·
[`toolkit/ml_project_methodology.md`](../../../toolkit/ml_project_methodology.md)
