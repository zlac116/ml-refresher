"""
JUST GROUP — VALUATION QUANT DRILLS (fill-in-the-blank)
=======================================================

HOW TO USE
    Each DRILL gives you the FORMULA and the EXPECTED OUTPUT. You implement the
    function body (replace `raise NotImplementedError`). Run the file; the grading
    block asserts your answers. Goal: map maths -> Python from memory, not reading.

    $ uv run python just_valuation_drills.py
    (stuck? peek at just_valuation_drills_SOLUTIONS.py)

GIVEN TO YOU (do not change)
    A continuously-compounded zero curve + a linear interpolator + zero_rate().
    Pure standard library (math only) so it runs anywhere.

TOPICS
    discounting · annuity PV · PV01/DV01 · par swap rate · duration & convexity
    rate stress + variation-margin sizing · NNEG (Black-76) put for a lifetime mortgage
"""

from scipy.special import erf
import numpy as np

# ── GIVEN: market data + helpers (do not change) ────────────────────────────
CURVE_T = [1, 2, 3, 5, 7, 10, 15, 20, 30]
CURVE_Z = [0.045, 0.044, 0.043, 0.042, 0.0415, 0.041, 0.040, 0.0395, 0.039]

def interp(x, xs, ys):
    if x <= xs[0]:  return ys[0]
    if x >= xs[-1]: return ys[-1]
    for i in range(1, len(xs)):
        if x <= xs[i]:
            w = (x - xs[i-1]) / (xs[i] - xs[i-1]); return ys[i-1] + w*(ys[i]-ys[i-1])

def zero_rate(t, bump_bp=0.0):
    """Interpolated zero rate at t, plus an optional parallel bump in bp."""
    return interp(t, CURVE_T, CURVE_Z) + bump_bp/1e4

# Liability used throughout: a 20-year level £1,000,000 p.a. annuity
T  = list(range(1, 21))
CF = [1_000_000.0] * 20


# ═══════════════════════════════════════════════════════════════════════════
# DRILL 1 — Discount factor & annuity PV          (~6 min)
#   FORMULA:  DF(t) = exp(-z(t)·t)        PV = Σ CFₜ · DF(t)
#   EXPECTED: PV(20y @ £1m) = £13,417,310.88
# ═══════════════════════════════════════════════════════════════════════════
def df(t, bump_bp=0.0):
    """Continuously-compounded discount factor: exp(-zero_rate(t)·t)."""
    r = zero_rate(t, bump_bp)
    return np.exp(-r*t)

def pv_cashflows(times, cfs, bump_bp=0.0):
    """PV = Σ CFₜ · DF(t).  Pass bump_bp through to df()."""
    dfs = np.array([df(t, bump_bp) for t in times])
    return np.sum(cfs * dfs)


# ═══════════════════════════════════════════════════════════════════════════
# DRILL 2 — PV01 / DV01                            (~4 min)
#   FORMULA:  DV01 = PV(curve) − PV(curve + 1bp)   (value lost per +1bp)
#   EXPECTED: DV01 = £12,349.79   (≈ 9.2y effective duration = DV01/PV·1e4)
# ═══════════════════════════════════════════════════════════════════════════
def dv01(times, cfs):
    """Reuse pv_cashflows with bump_bp = 0 and 1."""
    return pv_cashflows(times, cfs) - pv_cashflows(times, cfs, 1.0)


# ═══════════════════════════════════════════════════════════════════════════
# DRILL 3 — Par swap rate from discount factors    (~7 min)
#   FORMULA:  S = (DF₀ − DF_N) / Σ τᵢ·DFᵢ     (annual fixed leg, τ = 1/freq, DF₀ = 1... here use 1 - DF_N)
#             i.e.  S = (1 − DF_N) / Σ τᵢ·DFᵢ
#   EXPECTED: 10y par swap rate = 4.2024%
# ═══════════════════════════════════════════════════════════════════════════
def par_swap_rate(maturity, freq=1):
    """Build payment times (1..maturity·freq)/freq, τ = 1/freq, DFs via df();
       return (1 − DF_last) / Σ τ·DF."""
    df_last = df(maturity)
    float_pv = 1 - df_last
    delta = 1 / freq
    times = np.arange(1, maturity + 1) / freq
    dfs = np.array([df(t) for t in times])
    fix_pv = np.sum(delta * dfs)
    return float_pv / fix_pv


# ═══════════════════════════════════════════════════════════════════════════
# DRILL 4 — Modified duration & convexity          (~8 min)
#   FORMULA (flat yield y, central difference, step dy):
#     P(y)      = Σ CFₜ·exp(−y·t)
#     ModDur    = −(P(y+dy) − P(y−dy)) / (2·dy·P(y))
#     Convexity =  (P(y+dy) − 2P(y) + P(y−dy)) / (P(y)·dy²)
#   EXPECTED: ModDur = 9.1518 yrs ,  Convexity = 115.91
# ═══════════════════════════════════════════════════════════════════════════
def price_flat(y, times, cfs):
    """Σ CFₜ · exp(−y·t)."""
    raise NotImplementedError

def mod_duration(times, cfs, y=0.041, dy=1e-4):
    raise NotImplementedError

def convexity(times, cfs, y=0.041, dy=1e-4):
    raise NotImplementedError


# ═══════════════════════════════════════════════════════════════════════════
# DRILL 5 — Rate stress + variation-margin sizing  (~5 min)  ← the 2022 LDI story
#   IDEA: liability falls when rates rise (good for solvency) BUT a fully-hedged
#         receive-fixed book LOSES MtM and gets a CASH variation-margin call.
#   FORMULA:  VM call on +Δbp  ≈  DV01 · Δbp
#   EXPECTED: PV(+100bp)=£12,256,453.62 ; PV(+200bp)=£11,231,686.77 ;
#             PV(−100bp)=£14,735,710.73 ; VM call(+100bp)=£1,234,979.20
# ═══════════════════════════════════════════════════════════════════════════
def variation_margin_call(times, cfs, shock_bp):
    """Cash margin call ≈ DV01 · shock_bp  (use your dv01())."""
    raise NotImplementedError


# ═══════════════════════════════════════════════════════════════════════════
# DRILL 6 — NNEG put for a lifetime mortgage (Black-76)   (~9 min)  ← Just core
#   The lender is SHORT a put on the property.
#   FORMULA:  F  = S₀·exp((r − q)·T)        (forward at the deferment rate q)
#             K  = loan₀·exp(roll·T)        (rolled-up loan = strike)
#             d₁ = [ln(F/K) + ½σ²T] / (σ√T) ,  d₂ = d₁ − σ√T
#             Put = exp(−rT)·[K·N(−d₂) − F·N(−d₁)]
#   EXPECTED: NNEG put = £14,798.95   (S₀=400k, loan=150k, roll=6%, T=15, q=2%, σ=13%, r=4.1%)
# ═══════════════════════════════════════════════════════════════════════════
def norm_cdf(x):
    """Standard normal CDF.  Hint: 0.5·(1 + erf(x/√2))."""
    raise NotImplementedError

def nneg_put(S0, loan0, roll, T, q, vol, r):
    raise NotImplementedError


# ═══════════════════════════════════════════════════════════════════════════
# GRADING — run me; all asserts must pass
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # DRILL 1
    assert abs(df(0) - 1.0) < 1e-12
    pv = pv_cashflows(T, CF)
    assert abs(pv - 13_417_310.884) < 1e-1, f"PV off: {pv}"

    # DRILL 2
    d = dv01(T, CF)
    assert abs(d - 12_349.792) < 1e-2, f"DV01 off: {d}"

    # DRILL 3
    s = par_swap_rate(10)
    assert abs(s - 0.0420239344) < 1e-8, f"par rate off: {s}"

    # DRILL 4
    md, cx = mod_duration(T, CF), convexity(T, CF)
    assert abs(md - 9.1518274) < 1e-4, f"ModDur off: {md}"
    assert abs(cx - 115.914469) < 1e-3, f"Convexity off: {cx}"

    # DRILL 5
    assert abs(pv_cashflows(T, CF, 100) - 12_256_453.617) < 1e-1
    assert abs(pv_cashflows(T, CF, 200) - 11_231_686.770) < 1e-1
    assert abs(pv_cashflows(T, CF, -100) - 14_735_710.729) < 1e-1
    vm = variation_margin_call(T, CF, 100)
    assert abs(vm - 1_234_979.203) < 1e-1, f"VM off: {vm}"

    # DRILL 6
    assert abs(norm_cdf(0) - 0.5) < 1e-12
    nneg = nneg_put(400_000, 150_000, 0.06, 15, 0.02, 0.13, 0.041)
    assert abs(nneg - 14_798.951) < 1e-2, f"NNEG off: {nneg}"

    print(f"D1  Liability PV           = £{pv:,.2f}")
    print(f"D2  DV01                   = £{d:,.2f}   (~{d/pv*1e4:.1f}y duration)")
    print(f"D3  10y par swap rate      = {s*100:.4f}%")
    print(f"D4  ModDur {md:.4f}y  Convexity {cx:.2f}")
    print(f"D5  VM call (+100bp)       = £{vm:,.2f}")
    print(f"D6  NNEG put               = £{nneg:,.2f}")
    print("\n✓ All checks passed.")
