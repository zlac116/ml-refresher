"""
JUST GROUP — VALUATION QUANT DRILLS  (SOLUTION KEY)
Only peek if stuck. Try the blanks in just_valuation_drills.py first.
"""

import numpy as np
from scipy.special import erf
CURVE_T = [1, 2, 3, 5, 7, 10, 15, 20, 30]
CURVE_Z = [0.045, 0.044, 0.043, 0.042, 0.0415, 0.041, 0.040, 0.0395, 0.039]

def interp(x, xs, ys):
    if x <= xs[0]:  return ys[0]
    if x >= xs[-1]: return ys[-1]
    for i in range(1, len(xs)):
        if x <= xs[i]:
            w = (x - xs[i-1]) / (xs[i] - xs[i-1]); return ys[i-1] + w*(ys[i]-ys[i-1])

def zero_rate(t, bump_bp=0.0):
    return interp(t, CURVE_T, CURVE_Z) + bump_bp/1e4

T  = list(range(1, 21))
CF = [1_000_000.0] * 20

# DRILL 1
def df(t, bump_bp=0.0):
    return np.exp(-zero_rate(t, bump_bp) * t)

def pv_cashflows(times, cfs, bump_bp=0.0):
    return sum(cf * df(t, bump_bp) for t, cf in zip(times, cfs))

# DRILL 2
def dv01(times, cfs):
    return pv_cashflows(times, cfs, 0.0) - pv_cashflows(times, cfs, 1.0)

# DRILL 3
def par_swap_rate(maturity, freq=1):
    n = maturity * freq
    pay = [(i + 1) / freq for i in range(n)]
    tau = 1.0 / freq
    dfs = [df(t) for t in pay]
    return (1.0 - dfs[-1]) / sum(tau * d for d in dfs)

# DRILL 4
def price_flat(y, times, cfs):
    return sum(cf * np.exp(-y * t) for t, cf in zip(times, cfs))

def mod_duration(times, cfs, y=0.041, dy=1e-4):
    P0 = price_flat(y, times, cfs)
    return -(price_flat(y+dy, times, cfs) - price_flat(y-dy, times, cfs)) / (2*dy*P0)

def convexity(times, cfs, y=0.041, dy=1e-4):
    P0 = price_flat(y, times, cfs)
    return (price_flat(y+dy, times, cfs) - 2*P0 + price_flat(y-dy, times, cfs)) / (P0*dy*dy)

# DRILL 5
def variation_margin_call(times, cfs, shock_bp):
    return dv01(times, cfs) * shock_bp

# DRILL 6
def norm_cdf(x):
    return 0.5 * (1 + erf(x / np.sqrt(2)))

def nneg_put(S0, loan0, roll, T, q, vol, r):
    K = loan0 * np.exp(roll * T)
    F = S0 * np.exp((r - q) * T)
    d1 = (np.log(F/K) + 0.5*vol*vol*T) / (vol*np.sqrt(T))
    d2 = d1 - vol*np.sqrt(T)
    return np.exp(-r*T) * (K*norm_cdf(-d2) - F*norm_cdf(-d1))


if __name__ == "__main__":
    pv = pv_cashflows(T, CF)
    assert abs(df(0) - 1.0) < 1e-12
    assert abs(pv - 13_417_310.884) < 1e-1
    d = dv01(T, CF);                 assert abs(d - 12_349.792) < 1e-2
    s = par_swap_rate(10);           assert abs(s - 0.0420239344) < 1e-8
    md, cx = mod_duration(T, CF), convexity(T, CF)
    assert abs(md - 9.1518274) < 1e-4 and abs(cx - 115.914469) < 1e-3
    assert abs(pv_cashflows(T, CF, 100) - 12_256_453.617) < 1e-1
    assert abs(pv_cashflows(T, CF, 200) - 11_231_686.770) < 1e-1
    assert abs(pv_cashflows(T, CF, -100) - 14_735_710.729) < 1e-1
    vm = variation_margin_call(T, CF, 100); assert abs(vm - 1_234_979.203) < 1e-1
    assert abs(norm_cdf(0) - 0.5) < 1e-12
    nneg = nneg_put(400_000, 150_000, 0.06, 15, 0.02, 0.13, 0.041)
    assert abs(nneg - 14_798.951) < 1e-2
    print("✓ Solution key — all checks passed.")
