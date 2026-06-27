"""
DRILL 8 — Historical-Simulation VaR Engine  — SOLUTION KEY
Run: uv run python hist_var_drill_SOLUTIONS.py
"""

import numpy as np
CURVE_T = [1, 2, 3, 5, 7, 10, 15, 20, 30]
CURVE_Z = [0.045, 0.044, 0.043, 0.042, 0.0415, 0.041, 0.040, 0.0395, 0.039]

def interp(x, xs, ys):
    if x <= xs[0]:  return ys[0]
    if x >= xs[-1]: return ys[-1]
    for i in range(1, len(xs)):
        if x <= xs[i]:
            w = (x - xs[i-1]) / (xs[i] - xs[i-1]); return ys[i-1] + w*(ys[i]-ys[i-1])

def zero_rate(t, rate_bump_bp=0.0):
    return interp(t, CURVE_T, CURVE_Z) + rate_bump_bp/1e4

N = 250
d_rate_bp   = [8*np.sin(i*0.3) + ((i % 11) - 5) for i in range(N)]
d_spread_bp = [5*np.cos(i*0.2) + ((i % 7) - 3) for i in range(N)]

TIMES = [1, 2, 3, 4, 5]
CFS   = [4, 4, 4, 4, 104]
BASE_SPREAD_BP = 150.0

# TASK 1
def bond_price(times, cfs, rate_bump_bp=0.0, spread_bp=BASE_SPREAD_BP):
    s = spread_bp / 1e4
    return sum(cf * np.exp(-(zero_rate(t, rate_bump_bp) + s) * t) for t, cf in zip(times, cfs))

# TASK 2
def dv01(times, cfs, spread_bp=BASE_SPREAD_BP):
    return bond_price(times, cfs, 0.0, spread_bp) - bond_price(times, cfs, 1.0, spread_bp)

def cs01(times, cfs, spread_bp=BASE_SPREAD_BP):
    return bond_price(times, cfs, 0.0, spread_bp) - bond_price(times, cfs, 0.0, spread_bp + 1.0)

# TASK 3
def pnl_vector(times, cfs, base_spread_bp, d_rate_bp, d_spread_bp):
    base = bond_price(times, cfs, 0.0, base_spread_bp)
    return [bond_price(times, cfs, dr, base_spread_bp + ds) - base
            for dr, ds in zip(d_rate_bp, d_spread_bp)]

# TASK 4
def historical_var(pnls, conf=0.995):
    s = sorted(pnls)
    idx = int((1 - conf) * len(s))
    return -s[idx]


if __name__ == "__main__":
    base = bond_price(TIMES, CFS)
    assert abs(base - 92.0706468) < 1e-5
    assert abs(dv01(TIMES, CFS) - 0.0424617) < 1e-6
    assert abs(cs01(TIMES, CFS) - 0.0424617) < 1e-6
    pnls = pnl_vector(TIMES, CFS, BASE_SPREAD_BP, d_rate_bp, d_spread_bp)
    assert abs(min(pnls) - (-0.8011423)) < 1e-5 and abs(max(pnls) - 0.8504003) < 1e-5
    assert abs(historical_var(pnls, 0.995) - 0.7362917) < 1e-5
    assert abs(historical_var(pnls, 0.99)  - 0.7032813) < 1e-5
    print("✓ Solution key — all checks passed.")
