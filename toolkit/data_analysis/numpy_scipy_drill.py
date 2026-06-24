"""
TOOLKIT — numpy + scipy task-indexed drill
============================================

OBJECTIVE
    Practise the 12 canonical numpy + scipy idioms from the cheatsheet:
    array creation, indexing, axis reductions, np.where conditionals,
    broadcasting, linear algebra, RNG, scipy.stats moments + tests +
    distributions, cumulative ops, and view-vs-copy semantics.

ESTIMATED TIME
    60–90 min

TOPICS
    np.zeros / ones / full / linspace / arange
    fancy indexing, boolean masks
    axis-aware reductions (.sum / .mean / .argmax / np.percentile)
    np.where (single + nested) for elementwise conditionals
    broadcasting 1-D → 2-D (rows AND columns)
    np.linalg.solve, np.linalg.lstsq
    np.random.default_rng (reproducible)
    scipy.stats.skew / kurtosis / jarque_bera / ks_2samp / ttest_ind
    scipy.stats.norm.cdf / .ppf
    np.cumsum, np.log1p, np.maximum.accumulate (equity / drawdown patterns)

EXPECTED OUTPUT
    zeros sum:          0.0
    full sum:          -4.0
    linspace[2]:        0.5
    fancy row 2:        [-0.7037, -1.2654, -0.6233, 0.0413]
    mask positive #:    9
    col means:          [-0.7966, -0.3142, 0.0974, 0.2807]
    np.where clipped:   [0, 0, 0, 1, 2]
    broadcasted sums:   row=366 col=2466
    solve Ax=b:         [0.8, 1.3]
    lstsq beta:         [2.001, -0.992, 0.499]
    sample stats:       mean=-0.0289 std=0.9887
    skew / kurt:        -0.0437 / 0.0854
    JB p-value:         0.7326       (looks normal)
    KS p / Welch p:     0.0089 / 0.0109  (distributions differ)
    norm cdf(1.96):     0.975
    norm ppf(0.975):    1.96
    equity last:        1.0300
    a after view mut:   99

GRADING
    All asserts must pass.
"""
import numpy as np
from scipy import stats


# ── TASK 1 — Array creation ────────────────────────────────────────────────
def make_arrays() -> dict[str, np.ndarray]:
    """Return a dict with the following arrays:
        'zeros':    1-D, length 5, all zeros
        'ones':     2-D shape (2, 3), all ones
        'full':     2-D shape (2, 2), all -1.0
        'linspace': 5 evenly-spaced floats from 0 to 1 inclusive
        'arange':   integers 0, 2, 4, 6, 8
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 2 — Indexing + boolean masks ─────────────────────────────────────
def index_and_mask(x: np.ndarray) -> tuple[np.ndarray, int]:
    """Return (row_2, n_positive).
        row_2       : the third row of `x` (shape (4,))
        n_positive  : count of elements in x that are strictly > 0
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 3 — Axis-aware reductions ────────────────────────────────────────
def axis_reductions(x: np.ndarray) -> dict:
    """Return dict with keys:
        col_means : x.mean(axis=0), shape (4,)
        argmax    : index of x.max() in the FLATTENED array (int)
        p90       : 90th percentile of x (scalar)
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 4 — np.where conditional ─────────────────────────────────────────
def clip_negatives_to_zero(arr: np.ndarray) -> np.ndarray:
    """Return an array where positive elements are kept, negatives become 0.
    Use np.where (not a list comprehension).
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 5 — Broadcasting 1-D → 2-D ───────────────────────────────────────
def add_row_and_col(M: np.ndarray, v_row: np.ndarray, v_col: np.ndarray) -> tuple[int, int]:
    """Given a 2-D matrix `M` shape (3, 4):
        v_row shape (4,): add to EVERY ROW of M
        v_col shape (3,): add to EVERY COLUMN of M (use v_col[:, None])

    Return (sum_after_row, sum_after_col) as Python ints.
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 6 — Linear algebra ───────────────────────────────────────────────
def solve_linear_system(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve A x = b. Use np.linalg.solve (NOT np.linalg.inv).
    Returns 1-D array x.
    """
    # TODO: implement
    raise NotImplementedError


def linear_regression_via_lstsq(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fit y = X @ beta + noise via np.linalg.lstsq. Return beta of shape (n_features,)."""
    # TODO: implement (hint: np.linalg.lstsq(X, y, rcond=None)[0])
    raise NotImplementedError


# ── TASK 7 — Reproducible RNG ─────────────────────────────────────────────
def reproducible_normals(n: int, mean: float, std: float, seed: int = 42) -> np.ndarray:
    """Draw n samples from N(mean, std^2) reproducibly.
    Use np.random.default_rng(seed) — NOT the legacy np.random.normal.
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 8 — scipy.stats moments + normality test ────────────────────────
def moments_and_normality(sample: np.ndarray) -> dict:
    """Return dict with:
        skew         : scipy.stats.skew(sample)
        kurt         : scipy.stats.kurtosis(sample)   (excess, default)
        jarque_bera_p: p-value from scipy.stats.jarque_bera
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 9 — Two-sample tests ─────────────────────────────────────────────
def compare_two_samples(a: np.ndarray, b: np.ndarray) -> dict:
    """Compare distributions and means:
        ks_p     : scipy.stats.ks_2samp(a, b).pvalue
        welch_p  : scipy.stats.ttest_ind(a, b, equal_var=False).pvalue
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 10 — Distributions: cdf + ppf ────────────────────────────────────
def normal_cdf_and_ppf() -> tuple[float, float]:
    """Return (cdf_at_1_96, ppf_at_0_975) for a standard normal.
    Use scipy.stats.norm.cdf and .ppf.
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 11 — Cumulative ops (equity + running max) ──────────────────────
def equity_curve_from_simple_returns(rets: np.ndarray) -> np.ndarray:
    """equity_t = product over k=1..t of (1 + r_k).

    Use the log-space recipe (more numerically stable):
        equity = np.exp(np.cumsum(np.log1p(rets)))
    """
    # TODO: implement
    raise NotImplementedError


def running_max(x: np.ndarray) -> np.ndarray:
    """Element-wise running maximum. Use np.maximum.accumulate (NOT a for loop)."""
    # TODO: implement
    raise NotImplementedError


# ── TASK 12 — View vs copy semantics ─────────────────────────────────────
def mutating_slice_propagates() -> int:
    """Build a = np.arange(10), take view = a[2:5], set view[0] = 99.
    Return a[2] AFTER the mutation (should be 99 — basic slices are VIEWS).
    """
    # TODO: implement
    raise NotImplementedError


# ── GRADING ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Task 1
    arrs = make_arrays()
    assert set(arrs) == {"zeros", "ones", "full", "linspace", "arange"}
    assert arrs["zeros"].sum() == 0.0
    assert arrs["full"].sum() == -4.0
    assert abs(arrs["linspace"][2] - 0.5) < 1e-9
    assert arrs["arange"].tolist() == [0, 2, 4, 6, 8]

    # Task 2
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (5, 4))
    row_2, n_pos = index_and_mask(X)
    assert row_2.shape == (4,)
    assert np.allclose(row_2, [-0.7037, -1.2654, -0.6233, 0.0413], atol=1e-3)
    assert n_pos == 9

    # Task 3
    red = axis_reductions(X)
    assert np.allclose(red["col_means"], [-0.7966, -0.3142, 0.0974, 0.2807], atol=1e-3)
    assert int(red["argmax"]) == 6
    assert abs(red["p90"] - 0.9566) < 1e-3

    # Task 4
    out = clip_negatives_to_zero(np.array([-2, -1, 0, 1, 2]))
    assert out.tolist() == [0, 0, 0, 1, 2]

    # Task 5
    M = np.arange(12).reshape(3, 4)
    v_row = np.array([10, 20, 30, 40])
    v_col = np.array([100, 200, 300])
    s_row, s_col = add_row_and_col(M, v_row, v_col)
    assert int(s_row) == 366
    assert int(s_col) == 2466

    # Task 6
    A = np.array([[3.0, 2.0], [1.0, 4.0]])
    b = np.array([5.0, 6.0])
    x_sol = solve_linear_system(A, b)
    assert np.allclose(x_sol, [0.8, 1.3])

    rng_reg = np.random.default_rng(1)
    Xr = rng_reg.normal(0, 1, (100, 3))
    beta_true = np.array([2.0, -1.0, 0.5])
    y = Xr @ beta_true + rng_reg.normal(0, 0.1, 100)
    beta_hat = linear_regression_via_lstsq(Xr, y)
    assert np.allclose(beta_hat, beta_true, atol=0.05)

    # Task 7
    s = reproducible_normals(1000, 0.0, 1.0, seed=42)
    assert s.shape == (1000,)
    assert abs(s.mean() - -0.0289) < 1e-3
    assert abs(s.std()  -  0.9887) < 1e-3

    # Task 8
    m = moments_and_normality(s)
    assert abs(m["skew"] - -0.0437) < 1e-3
    assert abs(m["kurt"] -  0.0854) < 1e-3
    assert m["jarque_bera_p"] > 0.05  # looks normal

    # Task 9
    rng3 = np.random.default_rng(0)
    a = rng3.normal(0, 1, 500)
    b2 = rng3.normal(0.2, 1, 500)
    cmp = compare_two_samples(a, b2)
    assert cmp["ks_p"] < 0.05
    assert cmp["welch_p"] < 0.05

    # Task 10
    cdf_v, ppf_v = normal_cdf_and_ppf()
    assert abs(cdf_v - 0.975) < 1e-3
    assert abs(ppf_v - 1.96)  < 1e-3

    # Task 11
    rets = np.array([0.01, -0.005, 0.02, -0.01, 0.015])
    eq = equity_curve_from_simple_returns(rets)
    assert abs(eq[-1] - 1.030) < 1e-3
    assert running_max(np.array([1, 3, 2, 4, 3])).tolist() == [1, 3, 3, 4, 4]

    # Task 12
    assert mutating_slice_propagates() == 99

    print(f"zeros sum:          {arrs['zeros'].sum()}")
    print(f"full sum:          {arrs['full'].sum()}")
    print(f"linspace[2]:        {arrs['linspace'][2]}")
    print(f"fancy row 2:        {row_2.round(4).tolist()}")
    print(f"mask positive #:    {n_pos}")
    print(f"col means:          {red['col_means'].round(4).tolist()}")
    print(f"np.where clipped:   {out.tolist()}")
    print(f"broadcasted sums:   row={s_row} col={s_col}")
    print(f"solve Ax=b:         {x_sol.tolist()}")
    print(f"lstsq beta:         {beta_hat.round(3).tolist()}")
    print(f"sample stats:       mean={s.mean():.4f} std={s.std():.4f}")
    print(f"skew / kurt:        {m['skew']:.4f} / {m['kurt']:.4f}")
    print(f"JB p-value:         {m['jarque_bera_p']:.4f}       (looks normal)")
    print(f"KS p / Welch p:     {cmp['ks_p']:.4f} / {cmp['welch_p']:.4f}")
    print(f"norm cdf(1.96):     {cdf_v:.4f}")
    print(f"norm ppf(0.975):    {ppf_v:.4f}")
    print(f"equity last:        {eq[-1]:.4f}")
    print(f"a after view mut:   {mutating_slice_propagates()}")
    print("\n✓ All checks passed.")
