"""
DRILL 7 — Liability Cashflow Panel: PV & Portfolio DV01  (pandas)
================================================================

OBJECTIVE
    Convert a long-format (scheme / tenor / cashflow) DataFrame of annuity
    liability cashflows into a wide panel indexed by tenor (one column per
    scheme), interpolate a discount curve onto the tenors, then compute the
    PV per scheme, the total portfolio PV, and the portfolio DV01.

RUN
    uv run python just_pandas_drills.py
    (pandas/numpy come from this project's uv env; plain uv run python won't have them)

ESTIMATED TIME
    20 min

TOPICS
    pandas.DataFrame.pivot · .fillna · column ordering · sort_index
    DataFrame.mul(series, axis=0) then .sum()  (vector PV across a panel)
    numpy.interp for curve interpolation · curve bump for DV01

FORMULAS
    DF(t) = exp(-z(t)·t)            z(t) via linear interp of the zero curve
    PV(scheme) = Σ_tenor CF(tenor) · DF(tenor)
    Portfolio DV01 = PV(curve) − PV(curve + 1bp)

EXPECTED OUTPUT
    long rows:          50
    wide shape:         (25, 3)
    columns:            ['SchemeA', 'SchemeB', 'SchemeC']
    PV SchemeA:         10,966,364.84
    PV SchemeB:          9,265,844.79
    PV SchemeC:         16,007,532.61
    portfolio PV:       36,239,742.24
    portfolio DV01:         26,421.83

GRADING
    All asserts must pass.
"""
import numpy as np
import pandas as pd

# ── GIVEN: zero curve + deterministic liability cashflows (do not change) ────
CURVE_T = np.array([1, 2, 3, 5, 7, 10, 15, 20, 30.0])
CURVE_Z = np.array([4.5, 4.4, 4.3, 4.2, 4.15, 4.1, 4.0, 3.95, 3.9]) / 100

_SCHEMES = {"SchemeA": (1_000_000, 15), "SchemeB": (600_000, 25), "SchemeC": (2_000_000, 10)}
long_df = pd.DataFrame(
    [{"scheme": s, "tenor": t, "cashflow": float(amt)}
     for s, (amt, n) in _SCHEMES.items() for t in range(1, n + 1)]
)


# ── TASK 1 ──────────────────────────────────────────────────────────────────
def to_wide(long_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot to index=tenor, columns=scheme, values=cashflow.
    Fill missing with 0.0, sort the index ascending, and order the columns
    ['SchemeA', 'SchemeB', 'SchemeC']."""
    raise NotImplementedError


# ── TASK 2 ──────────────────────────────────────────────────────────────────
def discount_factors(tenors, bump_bp: float = 0.0) -> pd.Series:
    """DF(t) = exp(-z(t)·t), where z(t) = numpy.interp(t, CURVE_T, CURVE_Z) + bump_bp/1e4.
    Return a pd.Series indexed by `tenors`."""
    raise NotImplementedError


# ── TASK 3 ──────────────────────────────────────────────────────────────────
def pv_per_scheme(wide: pd.DataFrame, dfs: pd.Series) -> pd.Series:
    """PV per column = Σ_tenor cashflow · DF(tenor).
    Hint: wide.mul(dfs, axis=0).sum()  (dfs is indexed by tenor to match wide.index)."""
    raise NotImplementedError


# ── TASK 4 ──────────────────────────────────────────────────────────────────
def portfolio_dv01(wide: pd.DataFrame) -> float:
    """Total portfolio value lost per +1bp parallel shift:
    PV(curve) − PV(curve + 1bp), summed over all schemes."""
    raise NotImplementedError


# ── GRADING ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    assert len(long_df) == 50

    wide = to_wide(long_df)
    assert wide.shape == (25, 3), wide.shape
    assert list(wide.columns) == ["SchemeA", "SchemeB", "SchemeC"]
    assert wide.index.is_monotonic_increasing
    assert wide.loc[1, "SchemeC"] == 2_000_000 and wide.loc[20, "SchemeC"] == 0.0  # C stops at 10y

    dfs = discount_factors(wide.index)
    assert abs(float(dfs.loc[10]) - np.exp(-0.041 * 10)) < 1e-12

    pvs = pv_per_scheme(wide, dfs)
    assert abs(pvs["SchemeA"] - 10_966_364.84) < 1e-1, pvs["SchemeA"]
    assert abs(pvs["SchemeB"] -  9_265_844.79) < 1e-1, pvs["SchemeB"]
    assert abs(pvs["SchemeC"] - 16_007_532.61) < 1e-1, pvs["SchemeC"]
    assert abs(pvs.sum()      - 36_239_742.24) < 1e-1

    dv01 = portfolio_dv01(wide)
    assert abs(dv01 - 26_421.83) < 1e-1, dv01

    print(f"long rows:        {len(long_df)}")
    print(f"wide shape:       {wide.shape}")
    print(f"PV SchemeA:       {pvs['SchemeA']:,.2f}")
    print(f"PV SchemeB:       {pvs['SchemeB']:,.2f}")
    print(f"PV SchemeC:       {pvs['SchemeC']:,.2f}")
    print(f"portfolio PV:     {pvs.sum():,.2f}")
    print(f"portfolio DV01:   {dv01:,.2f}")
    print("\n✓ All checks passed.")
