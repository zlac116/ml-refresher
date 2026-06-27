"""
DRILL 7 — Liability Cashflow Panel (pandas)  — SOLUTION KEY
Run: uv run python just_pandas_drills_SOLUTIONS.py
"""
import numpy as np
import pandas as pd

CURVE_T = np.array([1, 2, 3, 5, 7, 10, 15, 20, 30.0])
CURVE_Z = np.array([4.5, 4.4, 4.3, 4.2, 4.15, 4.1, 4.0, 3.95, 3.9]) / 100

_SCHEMES = {"SchemeA": (1_000_000, 15), "SchemeB": (600_000, 25), "SchemeC": (2_000_000, 10)}
long_df = pd.DataFrame(
    [{"scheme": s, "tenor": t, "cashflow": float(amt)}
     for s, (amt, n) in _SCHEMES.items() for t in range(1, n + 1)]
)

def to_wide(long_df: pd.DataFrame) -> pd.DataFrame:
    w = long_df.pivot(index="tenor", columns="scheme", values="cashflow").fillna(0.0).sort_index()
    return w[["SchemeA", "SchemeB", "SchemeC"]]

def discount_factors(tenors, bump_bp: float = 0.0) -> pd.Series:
    t = np.asarray(tenors, dtype=float)
    z = np.interp(t, CURVE_T, CURVE_Z) + bump_bp / 1e4
    return pd.Series(np.exp(-z * t), index=tenors)

def pv_per_scheme(wide: pd.DataFrame, dfs: pd.Series) -> pd.Series:
    return wide.mul(dfs, axis=0).sum()

def portfolio_dv01(wide: pd.DataFrame) -> float:
    pv0 = pv_per_scheme(wide, discount_factors(wide.index, 0.0)).sum()
    pv1 = pv_per_scheme(wide, discount_factors(wide.index, 1.0)).sum()
    return float(pv0 - pv1)


if __name__ == "__main__":
    wide = to_wide(long_df)
    assert wide.shape == (25, 3) and list(wide.columns) == ["SchemeA", "SchemeB", "SchemeC"]
    dfs = discount_factors(wide.index)
    pvs = pv_per_scheme(wide, dfs)
    assert abs(pvs["SchemeA"] - 10_966_364.84) < 1e-1
    assert abs(pvs["SchemeB"] -  9_265_844.79) < 1e-1
    assert abs(pvs["SchemeC"] - 16_007_532.61) < 1e-1
    assert abs(pvs.sum()      - 36_239_742.24) < 1e-1
    assert abs(portfolio_dv01(wide) - 26_421.83) < 1e-1
    print("✓ Solution key — all checks passed.")
