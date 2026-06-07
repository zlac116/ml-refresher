# ML / Quant Project Methodology — Low-Level Playbook

Personal cheatsheet for building an ML/quant project from a blank directory,
organised by phase. Each section is a pattern + one-line why + the mental
prompt that makes it second nature. Canonical example used throughout:
`quant_finance/capstones/lmm_nn_surrogate/lmm_nn_capstone.py`.

Two related docs: `eda_decisions.md` (what model/transform to pick),
`deployment.ipynb` (how to ship a model). This doc is the *code-writing*
layer between them.

---

## 0. Bootstrap

- `uv init --no-readme` for a new project. `uv sync` after every dep change.
- Pin CPU torch in `pyproject.toml` if no GPU needed:
  ```toml
  [tool.uv.sources]
  torch = { index = "pytorch-cpu" }
  [[tool.uv.index]]
  name = "pytorch-cpu"
  url = "https://download.pytorch.org/whl/cpu"
  explicit = true
  ```
- `argparse` at the bottom with `--seed --n-data --epochs --lr --hidden --out-dir`.
- **Constants at the top of the file** — bounds, dimensions, names. Never inline magic.
  ```python
  LMM_PARAM_LO = np.array([0.10, 0.30, 0.005, 0.10])
  T_LO, T_HI = 0.5, 10.0
  N_FEATURES = 7
  ```
  > **Mental prompt**: *"If I change this number, how many places have to change?"*
  > If `> 1`, it belongs at the top.

---

## 1. Data generation / loading

- **Modern numpy RNG**:
  ```python
  rng = np.random.default_rng(seed)
  ```
  Not `np.random.seed(...)` — that mutates a global and silently couples
  functions. `default_rng(seed)` is local and ownable.

- **Broadcast where natural, loop where cheap to reason about**:
  ```python
  params = rng.uniform(LO, HI, size=(n, 4))                          # broadcast
  ivs    = np.array([f(p, *xs) for p, xs in zip(...)])               # loop
  ```
  Don't blind-vectorise. Pure-Python loops at n=10k are sub-second.

- **Cast to float32 explicitly** (saves memory, matches torch defaults).
- **`np.column_stack` to assemble feature matrices** — keeps shape obvious.

> **Mental prompt**: *"Where does randomness enter this function? Make it a parameter."*

---

## 2. Train / val split

```python
rng    = np.random.default_rng(seed)
idx    = rng.permutation(len(X))
n_va   = max(1, int(val_frac * len(X)))      # defensive against tiny n
train_idx, val_idx = idx[:-n_va], idx[-n_va:]
return X[train_idx], y[train_idx], X[val_idx], y[val_idx]
```

- Slice the **index array**, not the data. Cheaper to reason about.
- `max(1, ...)` guards against `n_va = 0` for tiny datasets.

> **Mental prompt**: *"What's the smallest dataset this can handle without crashing?"*

---

## 3. Model architecture

```python
class Surrogate(nn.Module):
    def __init__(self, d_in: int, hidden: tuple[int, ...]):
        super().__init__()
        layers, prev = [], d_in
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.SiLU())            # smooth for downstream optimisers
            prev = h
        layers.append(nn.Linear(prev, 1))       # no output activation
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)          # (N,1) -> (N,)
```

| Choice | When |
|---|---|
| **SiLU / GELU** | Surrogate fed to an optimiser that uses Jacobians (least_squares, BO). |
| **ReLU** | Plain classification/regression with no downstream gradient needs. |
| **No output activation** | Regression with unbounded target. |
| **Sigmoid output** | Binary class probability *and* you need the probability (not just argmax). |
| **`.squeeze(-1)`** | Always, for regression — match label shape `(N,)` not `(N, 1)`. |

> **Mental prompts**:
> - *"What does the consumer of this function expect?"* (smooth? probabilistic? logits?)
> - *"What shape does the loss compare?"* — make output match exactly.

---

## 4. Training loop

```python
opt     = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.MSELoss()
history = {"train": [], "val": []}

for epoch in range(epochs):
    model.train()
    pred = model(Xt_tr)
    loss = loss_fn(pred, yt_tr)
    opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        val_loss = loss_fn(model(Xt_va), yt_va)

    history["train"].append(loss.item())
    history["val"].append(val_loss.item())
```

- **Full-batch** if data fits in memory (faster than mini-batching at n≤50k).
- **`model.train()` / `model.eval()`** consistently — toggles dropout/BN.
- **`opt.zero_grad()` BEFORE `backward`** — gradients accumulate by default.
- **`torch.no_grad()`** for val — ~2× faster, no autograd memory.
- **`.item()`** when appending to history — don't keep tensor references.
- **Print every `epochs // 10`** — cheap progress signal without spam.

For early stopping:
```python
best_state, best_val, patience_used = None, float("inf"), 0
if val_loss < best_val:
    best_val = val_loss
    best_state = copy.deepcopy(model.state_dict())
    patience_used = 0
else:
    patience_used += 1
    if patience_used >= patience:
        break
model.load_state_dict(best_state)
```

> **Mental prompt**: *"What's the cheapest signal that this is training?"*
> → log per-epoch train AND val loss.

---

## 5. Inference helper (the boundary)

```python
def nn_iv(model, params, instruments, device):
    feats = [[*params, T, np.log(K/F), F] for T, K, F in instruments]
    x = torch.tensor(feats, dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        return model(x).cpu().numpy()
```

- **Single forward pass** for batched inputs. Never loop the model call.
- **`.cpu().numpy()`** at the boundary — return numpy to non-torch consumers (scipy, pandas, sklearn).
- **`model.eval()` + `torch.no_grad()`** — both, always.

> **Mental prompt**: *"Am I crossing Python ↔ tensor in a loop?"*
> If yes, pull the boundary outside the loop.

---

## 6. Optimisation / calibration loop

```python
res = least_squares(
    fun = lambda p: predict(p) - target,
    x0  = (LO + HI) / 2,
    bounds = (LO, HI),
)
return res                              # NOT res.x — caller decides
```

- **Return the full result object** — `.x`, `.cost`, `.success`, `.message`,
  `.nfev` are all evidence the caller may need (audit, drift detection).
- **Pass bounds explicitly** — NN-backed optimisers extrapolate wildly
  outside their training region without bounds.
- **x0 from midpoint, warm-start, or domain prior** — never default to zero.
- **Lambda closures** for fixed args inside the residual (model, target,
  device) — cleaner than `partial` or globals.

> **Mental prompt**: *"What region must my approximation be valid in?"*
> → name those bounds as top-of-file constants, then reuse them as
> calibration bounds. **Single source of truth.**

---

## 7. Verification

The pattern that catches the most bugs:

1. Compute the truth-at-the-answer **with a different code path** than the
   one the optimiser used. (Here: real MC, not the NN.)
2. Form residuals that test **orthogonal failure modes**:
   - "model fits data": `target − truth(θ*)`
   - "approximation was honest at θ\*": `nn(θ*) − truth(θ*)`
3. Both must pass independently.

Anti-pattern: comparing the optimiser's output to its target. That's
≈ 0 by construction and provides no evidence.

For the report itself:
```python
df = pd.DataFrame(...)
df["resid_a"] = ...
df["resid_b"] = ...
print(df.to_string(index=False, formatters={
    "resid_a": "{:+.1f}".format,        # signed for residuals
    "iv":      "{:.4f}".format,
}))
rmse_a = float(np.sqrt(np.mean(df["resid_a"]**2)))
```

- **DataFrame is the single source of truth** — print AND CSV from it.
- **Per-column formatters** in `to_string` for mixed-precision tables.
- **`+` in format spec** for signed columns (residuals, bp).
- **RMSE footer** — one summary number per residual class, for the audit email.

> **Mental prompt**: *"How will I know it worked?"*
> Design verification BEFORE you start coding.

---

## 8. Saving artifacts

```
out/<timestamp>/
├── model.pt              ← framework reads this
├── report.csv            ← humans + pandas read this
├── history.csv           ← drift monitor reads this
└── run.json              ← machines / audit read this
```

- **Timestamped run dir** — never overwrite. `datetime.now().strftime("%Y%m%dT%H%M%S")`.
  Note the **literal `T`** — `%T` is HH:MM:SS, common typo.
- **One artifact per consumer**, format chosen for the reader:

  | Consumer | Format |
  |---|---|
  | Framework (PyTorch) | `.pt` / `.safetensors` |
  | Humans + pandas | `.csv` |
  | Machines + audit | `.json` |
  | Tabular at scale | `.parquet` |
  | **Avoid** | `.pkl` (opaque, version-fragile) |

- **`run.json` manifest** — include `theta_star`, `cost`, `success`,
  `message`, `nfev`, hyperparams, seed, market inputs, timestamp.

> **Mental prompt**: *"Who reads this file in 6 months, and what tool do they have?"*

---

## 9. The CLI entrypoint

```python
def main():
    args = parse_args()

    # 1. seed
    torch.manual_seed(args.seed)

    # 2. device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 3. data
    X, y = generate_data(args.n_data, args.seed)
    splits = split_train_val(X, y, args.val_frac, args.seed)

    # 4. model
    model = Surrogate(N_FEATURES, tuple(args.hidden)).to(device)

    # 5. train
    history = train(model, *splits, args.epochs, args.lr, device)

    # 6. use (calibrate / predict / etc)
    res = calibrate(model, market_data, x0, bounds, device)

    # 7. verify
    report = verify(model, res.x, market_data, device)

    # 8. save
    save_artifacts(model, report, history, res, args)
```

Top-to-bottom, no surprises. If `main` has to scroll, it's doing too much.

---

## 10. The mental checklist for a fresh project

Before writing line one, answer these in order:

1. **What slow expensive function am I replacing?** (MC pricer? Bayesian inner loop? Simulator?)
2. **What's its signature?** Design the NN to mirror it exactly.
3. **What region must the approximation be valid in?** Name those bounds at the top.
4. **What's the smallest pure function per workflow step?** One screenful each: generate → split → build → train → infer → use → verify.
5. **Where does the consumer meet the approximation?** (the inference boundary). Make sure it's ONE batched forward pass, not a loop.
6. **What do I save and for whom?** Pick the format for the reader.
7. **How will I know it worked?** Design two orthogonal verification residuals first.

If you can answer all seven cleanly, the code almost writes itself.

---

## 11. Anti-patterns to recognise instantly

| Smell | What to do |
|---|---|
| `np.random.seed(...)` | Use `np.random.default_rng(seed)` |
| Model call inside a Python loop over instruments | Build one feature matrix, one forward pass |
| Returning `res.x` instead of `res` | Return the whole `OptimizeResult` |
| Compare `nn(θ*)` to `target` to "verify" | That's the optimiser's residual; use truth from a different code path |
| Two parallel lists held together by `zip` | Build a DataFrame; columns can't drift |
| `.pkl` artifacts | Use `.csv` / `.json` / `.pt` for the right consumer |
| Magic numbers inline | Lift to top-of-file constants |
| Manual print loop alongside a DataFrame | `df.to_string(index=False, formatters=...)` |
| `request.app.state.model` directly in routes | `Depends(get_model)` via `ModelDep` alias |
| MLflow `Stage` (Staging/Production/Archived) | Use aliases (`@production`, `@candidate`) |
| `from __future__ import annotations` in Pydantic models | Drop it — pydantic v2 prefers concrete annotations |
| `mlflow.pytorch.log_model` without `signature` | Add `infer_signature(...)` — register the I/O contract |

---

## 12. When this methodology applies (and when it doesn't)

**Applies cleanly to**: surrogate models for calibration, regression NNs,
small tabular classifiers, any "train an approximation, deploy it" workflow.

**Modify for**:
- **Big data / streaming** → DataLoaders + mini-batches, not full-batch.
- **Sequence models** → swap MLP for transformer/LSTM, padding becomes the issue.
- **RL / online learning** → no static train/val split; need a replay buffer.
- **Generative models** → no single "verify residual"; need likelihood / sample quality / FID.

The phase structure (bootstrap → data → split → model → train → infer → use → verify → save) stays. The contents of each phase change.

---

## Cross-references

**Canonical implementations + project-specific lessons:**
- LMM NN surrogate (calibration / surrogate-replaces-slow-function):
  `quant_finance/capstones/lmm_nn_surrogate/lmm_nn_capstone.py` +
  `quant_finance/capstones/lmm_nn_surrogate/LESSONS.md`
- Tabular NN (multiclass classification, CrossEntropy):
  `ml/neural_networks/capstone/capstone_tabular_nn.py` +
  `ml/neural_networks/capstone/LESSONS_tabular_nn.md`
- Regression NN (continuous target, early stopping, target scaling):
  `ml/neural_networks/capstone/capstone_regression_nn.py` +
  `ml/neural_networks/capstone/LESSONS_regression_nn.md`

**Production wrapping (MLflow + FastAPI):**
- `quant_finance/capstones/lmm_nn_surrogate/api_extension/`

**Related toolkit docs:**
- EDA / model selection upstream: `toolkit/eda_decisions.md`
- Deployment downstream: `toolkit/deployment.ipynb`
