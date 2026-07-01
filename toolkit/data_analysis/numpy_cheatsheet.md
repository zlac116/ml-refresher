# NumPy + SciPy Cheatsheet (2026)

Companion to `numpy_card.pdf`. Narrative + worked I/O examples; the card is the
reference lookup.

Stack: **numpy ≥2.0 · scipy ≥1.14 · Python ≥3.10**. Modern conventions:
`np.random.default_rng()` (not `np.random.seed`), explicit `axis=`, `@` for
matmul, `solve()` not `inv()`.

---

## §0 — The general pattern

Every numpy script has the same 6 steps:

```
1. RNG        rng = np.random.default_rng(seed)      (canonical entry point)
2. CREATE     np.zeros / arange / linspace / list    (declare dtype)
3. RESHAPE    .reshape(-1, k)                         (-1 = auto)
4. INDEX      basic slice / boolean / fancy          (know view-vs-copy!)
5. VECTORISE  ufuncs; axis= on reductions            (never a Python loop)
6. LINALG     solve / lstsq / svd / cholesky          (not inv)
```

The **mental model**: think in **arrays and axes**, not in loops.
`axis=0` collapses the FIRST axis (result has one fewer dim);
`axis=1` collapses the SECOND axis.

---

## §1 — Array creation + dtype

```python
import numpy as np

np.zeros((3, 4), dtype=np.float32)     # shape (3, 4), all zeros
np.ones((2, 2))                         # all ones
np.arange(0, 10, 2)                     # array([0, 2, 4, 6, 8])
np.linspace(0, 1, 5)                    # array([0. , 0.25, 0.5, 0.75, 1. ])  ← inclusive
np.eye(3)                                # 3×3 identity
np.empty((5, 5))                         # uninitialised (fast; YOU must fill it)

# Meshgrid — turn two 1D vectors into 2D coordinate arrays
x = np.linspace(-2, 2, 5); y = np.linspace(-1, 1, 3)
X, Y = np.meshgrid(x, y, indexing="ij")  # X.shape = (5, 3)
```

**Dtype choice matters** for memory + speed:

| Dtype | Bytes/elem | When to use |
|---|---|---|
| `float64` | 8 | default; general |
| `float32` | 4 | ML feature tensors; halves memory |
| `int64` | 8 | default int; safe |
| `int32` / `int16` | 4 / 2 | known-bounded values |
| `uint8` | 1 | images (0-255) |
| `bool_` | 1 (byte, not bit) | masks |

```python
a = np.arange(1_000_000, dtype=np.float32)
a.nbytes / 1e6         # 4.0  MB (vs 8.0 for float64)
```

---

## §2 — Indexing: view vs copy is the whole game

```python
a = np.arange(24).reshape(4, 6)
# array([[ 0,  1,  2,  3,  4,  5],
#        [ 6,  7,  8,  9, 10, 11],
#        [12, 13, 14, 15, 16, 17],
#        [18, 19, 20, 21, 22, 23]])

a[1, 2]         # 8       (single element)
a[1]            # array([6,7,8,9,10,11])   (row 1 — 1D)
a[:, 2]         # array([2, 8, 14, 20])    (col 2 — 1D)
a[1:3, 2:5]     # 2×3 sub-block (VIEW — mutations propagate)
a[[0, 2]]       # rows 0 and 2 (fancy → COPY)
a[a > 10]       # mask (COPY, flattened to 1D)
```

**Check view vs copy**:

```python
b = a[1:3]
b.base is a          # True  → b is a view
b[0, 0] = -99
print(a[1, 0])        # -99   (view — mutation propagated to a!)

c = a[[0, 2]]
c.base is a          # False → c is a copy
c[0, 0] = 999
print(a[0, 0])        # 0     (copy — a unchanged)
```

**Rule of thumb**:
- Basic slicing (`:`) → view
- Fancy indexing (list of ints, array of ints, or boolean mask) → copy

If unsure and mutation might matter: `a[1:3].copy()`.

---

## §3 — Broadcasting

Broadcasting is why numpy is fast: it lets you write operations across arrays
of different-but-compatible shapes without explicit loops or reshapes.

**Rule**: align shapes from the **right**. Pad missing left dims with 1.
Compatible iff each dimension is equal OR one of them is 1.

```python
a = np.zeros((3, 5))     # shape (3, 5)
b = np.arange(5)          # shape (5,)  → conceptually (1, 5) → broadcasts to (3, 5)
c = a + b                 # OK — element-wise add across rows

col = np.arange(3).reshape(3, 1)   # (3, 1)
row = np.arange(5)                  # (5,) → (1, 5)
col + row                            # (3, 5) — outer sum
```

**Standardise columns of a feature matrix**:

```python
X = np.array([[1.0, 10.0],
               [2.0, 20.0],
               [3.0, 30.0]])   # shape (3, 2) — rows=samples, cols=features

mean = X.mean(axis=0, keepdims=True)   # (1, 2)   — keeping dim helps broadcast back
std  = X.std(axis=0, ddof=1, keepdims=True)
Xz = (X - mean) / std                    # (3, 2) — element-wise, broadcast (1, 2) → (3, 2)
```

**`keepdims=True`** on reductions is the single trick that eliminates most
`.reshape` boilerplate.

---

## §4 — Axis reductions (always specify `axis=`)

Every reduction takes an `axis=` argument. Missing it collapses ALL axes into a
scalar — rarely what you want.

```python
X = np.arange(12).reshape(3, 4)
# array([[ 0,  1,  2,  3],
#        [ 4,  5,  6,  7],
#        [ 8,  9, 10, 11]])

X.sum()                # 66      (scalar; ALL axes collapsed)
X.sum(axis=0)          # array([12, 15, 18, 21])   (collapse rows → col-sums)
X.sum(axis=1)          # array([ 6, 22, 38])       (collapse cols → row-sums)
X.mean(axis=0)         # array([4., 5., 6., 7.])
X.std(axis=0, ddof=1)  # sample std per column
np.quantile(X, q=[0.25, 0.5, 0.75], axis=0)   # per-column quartiles
```

**NaN-aware variants** (crucial with real data):

```python
X_with_nan = X.astype(float); X_with_nan[0, 0] = np.nan
X_with_nan.mean(axis=0)               # array([nan, 5., 6., 7.])   ← propagates
np.nanmean(X_with_nan, axis=0)        # array([6., 5., 6., 7.])    ← skips NaN
```

---

## §5 — RNG (`default_rng`, not `seed`)

```python
rng = np.random.default_rng(seed=42)     # canonical; reproducible

rng.standard_normal(size=(3, 2))
# array([[ 0.30471708, -1.03998411],
#        [ 0.7504512 ,  0.94056472],
#        [-1.95103519, -1.30217951]])

rng.integers(low=0, high=10, size=5)     # array([9, 4, 7, 6, 0])   ← high EXCLUSIVE
rng.integers(0, 10, size=5, endpoint=True)                          # ← high INCLUSIVE

rng.choice(np.array(["A","B","C"]), size=4, replace=True, p=[.5, .3, .2])
# array(['B', 'A', 'A', 'A'])

rng.multivariate_normal(mean=[0, 0], cov=[[1, 0.5], [0.5, 1]], size=1000).mean(0)
# array([0.001, -0.002])   ← ~ zero mean as expected
```

**Legacy**: `np.random.seed(42); np.random.rand(...)` uses a global RNG. Don't
use in new code — non-reproducible in parallel, deprecated in numpy 2.x roadmap.

**Parallel workers**: `rng.spawn(4)` creates 4 statistically independent child
RNGs — each worker gets one.

---

## §6 — Linear algebra

```python
# Matrix-matrix / matrix-vector
A = rng.standard_normal((5, 3))
x = rng.standard_normal(3)
b = A @ x                              # matmul (prefer @ over np.dot for 2D)

# Solving a linear system Ax = b  (NEVER use inv)
A = np.array([[3.0, 1.0], [1.0, 2.0]])
b = np.array([9.0, 8.0])
x = np.linalg.solve(A, b)              # array([2., 3.])  — cheaper + stabler than inv

# Least squares (overdetermined system)
A = rng.standard_normal((100, 3))
y = A @ np.array([1.0, 2.0, 3.0]) + 0.01 * rng.standard_normal(100)
beta, resid, rank, sv = np.linalg.lstsq(A, y, rcond=None)
beta                                    # ≈ array([1., 2., 3.])

# Decompositions
U, s, Vt = np.linalg.svd(A, full_matrices=False)  # economy SVD
L = np.linalg.cholesky(A.T @ A)                    # Cholesky of ATA (positive definite)
w, V = np.linalg.eigh(A.T @ A)                     # symmetric eig — MUCH faster than eig
```

**The `inv(A) @ b` anti-pattern**: numerically bad AND slower than `solve(A, b)`.
Only use `inv` when you genuinely need the inverse matrix (rare — usually you
need `A @ x = b` solved, which is `solve`).

---

## §7 — Where / select / clip

```python
x = np.array([-3, -1, 0, 2, 4])
np.where(x > 0, x, 0)          # array([0, 0, 0, 2, 4])    ← ternary
np.where(x > 0)                 # (array([3, 4]),)         ← indices only

# Multi-way
np.select(
    condlist=[x < 0, x == 0, x > 0],
    choicelist=["neg", "zero", "pos"],
    default="?")
# array(['neg', 'neg', 'zero', 'pos', 'pos'], dtype='<U4')

np.clip(x, a_min=-2, a_max=3)  # array([-2, -1,  0,  2,  3])
```

---

## §8 — scipy.stats

**Frozen distributions** are the canonical pattern (create once, reuse):

```python
from scipy import stats

# Frozen — bind parameters once
d = stats.norm(loc=0, scale=1)
d.pdf(0)       # 0.3989422804014327
d.cdf(1.96)    # 0.9750021048517795
d.ppf(0.975)   # 1.9599639845400545      ← inverse CDF
d.rvs(size=1000, random_state=rng)         # draw 1000 samples
```

**Standard distributions available**: `norm`, `t`, `chi2`, `f`, `gamma`,
`lognorm`, `beta`, `expon`, `uniform`, `poisson`, `binom`, `multivariate_normal`.

**Hypothesis tests** — 2-sample t-test on returns:

```python
a = rng.normal(0.001, 0.02, size=500)
b = rng.normal(0.002, 0.02, size=500)
res = stats.ttest_ind(a, b, equal_var=False)   # Welch's t-test
print(res.statistic, res.pvalue)                # -0.89, 0.37
```

**Correlation**:

```python
stats.pearsonr(x, y)       # linear; sensitive to outliers
stats.spearmanr(x, y)      # rank-based; robust
stats.kendalltau(x, y)     # rank-based; smaller sample sizes
```

---

## §9 — scipy.optimize

```python
from scipy import optimize

# 1D root find (needs a bracket)
optimize.brentq(lambda x: np.exp(x) - 5, 0, 5)        # 1.609...

# Multivariate minimisation with bounds (L-BFGS-B is a strong default)
def neg_log_lik(params):
    mu, log_sigma = params
    sigma = np.exp(log_sigma)
    return -stats.norm.logpdf(data, mu, sigma).sum()

data = rng.normal(2.0, 0.5, size=1000)
res = optimize.minimize(neg_log_lik, x0=[0.0, 0.0], method="L-BFGS-B",
                        bounds=[(-10, 10), (-3, 3)])
res.x           # array([1.97, -0.68])   ← ≈ [μ=2, log σ = log 0.5 = -0.69]

# Non-linear curve fitting
def model(x, a, b): return a * np.exp(-b * x)
x = np.linspace(0, 5, 50); y_true = model(x, 3, 0.5)
y = y_true + 0.05 * rng.standard_normal(50)
popt, pcov = optimize.curve_fit(model, x, y, p0=[1.0, 1.0])
popt                                    # array([2.99, 0.50])   ← recovered params
np.sqrt(np.diag(pcov))                  # parameter standard errors
```

**Method picker**:

| Method | When |
|---|---|
| `L-BFGS-B` | smooth + box bounds; strong default |
| `trust-constr` | smooth + general inequality/equality constraints |
| `SLSQP` | constraints; small-to-medium dim |
| `Nelder-Mead` | derivative-free; noisy objective |
| `differential_evolution` | global optimum on non-convex |

---

## §10 — scipy.sparse (for large mostly-zero matrices)

```python
from scipy import sparse

# Build a large sparse matrix (99.9% zeros)
rows = rng.integers(0, 10000, size=100_000)
cols = rng.integers(0, 5000,  size=100_000)
vals = rng.standard_normal(100_000)
A = sparse.coo_array((vals, (rows, cols)), shape=(10000, 5000)).tocsr()

A.nnz              # 100000       — non-zeros stored
A.shape            # (10000, 5000)
A.nbytes / 1e6     # ~1.2 MB      (vs 400 MB dense)

# Sparse arithmetic stays sparse
b = rng.standard_normal(5000)
y = A @ b          # sparse mat-vec — fast

# Sparse linear solve (never .toarray() first)
from scipy.sparse.linalg import spsolve
sq = A.T @ A + sparse.eye(5000) * 1e-6   # ridge-regularised normal equations
x = spsolve(sq, A.T @ y_target)
```

**Format picker**:

| Format | Best for |
|---|---|
| CSR | row slicing, matmul, general ML |
| CSC | column slicing / column ops |
| COO | construction only; convert to CSR/CSC |
| LIL / DOK | incremental building (slow arith) |

---

## §11 — Top traps

See the card for the full 12-item list. The most-common ones:

1. **Missing `axis=`** — reductions collapse ALL axes → scalar.
2. **`np.random.seed()`** legacy — use `np.random.default_rng(seed)`.
3. **`inv(A) @ b`** — numerically bad. Use `np.linalg.solve(A, b)`.
4. **`a == b` for floats** — unsafe. Use `np.isclose(a, b, rtol=1e-5)`.
5. **NaN propagation** — `x.mean()` → NaN if any NaN. Use `np.nanmean`.
6. **`std` ddof default** — `ddof=0` (population); use `ddof=1` for sample std.
7. **View trap** — `a[1:3] = 0` mutates `a`. Use `.copy()` when unsure.
8. **`np.empty`** returns garbage — always fill before reading.
9. **Broadcasting error** — shapes align from RIGHT. Use `keepdims=True`.
10. **`curve_fit` without `p0`** — silent bad local min. Always give a starting guess.

---

## §12 — Cross-references

- Card (dense reference): `numpy_card.pdf` in this folder.
- Drills: `../data_analysis/numpy_scipy_drill.py` and `numpy_scipy_project.py`.
- Companion: `pandas_cheatsheet.md` + `pandas_card.pdf` — the tabular DSL that
  sits on top of numpy.
- Downstream: every ML capstone uses numpy + scipy — see
  `ml/*/capstones/*/train.py` and `quant_finance/capstones/*/src/*`.
