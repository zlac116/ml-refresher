"""
DRILL 13 — Three ways to compute VaR  (historical · parametric · Monte Carlo)
============================================================================

Same portfolio, three engines — the "what other ways to simulate losses?" answer.
  GIVEN   historical_var  (full-reval over the actual scenarios — your DRILL-11 engine)
  TASK 1  parametric_var  (delta-normal:  z·√(sᵀΣs))
  TASK 2  monte_carlo_var (simulate factors ~ N(0,Σ) via Cholesky, FULL-revalue, percentile)

THE TRADE-OFFS (be able to say these)
  • Historical — no distributional assumption; the data IS the distribution. Limited
    by the window; misses anything not in it.
  • Parametric/variance-covariance — fast, analytic, but assumes factors are NORMAL
    and the book LINEAR (sensitivities only) → misses fat tails AND convexity.
  • Monte Carlo — simulate from an assumed Σ and full-revalue → captures non-linearity
    and any distribution, but heavy (paths × full reval) and model-dependent.

WHAT YOU'LL SEE
  Parametric ≈ Monte Carlo here (book is near-linear, so delta-normal ≈ full-reval MC),
  but BOTH > historical — because the historical scenarios are bounded/non-normal, so
  their real tail is THINNER than a normal with the same covariance. Methods disagree,
  and the assumption is why.

RUN
    uv run python var_methods_drill.py        (stuck? var_methods_drill_SOLUTIONS.py)

EXPECTED (99.5%)
    historical 4.6781 | parametric 6.5238 | monte-carlo 6.3411   (P&L std 2.5327)

GRADING: all asserts must pass.
"""
import numpy as np
from dataclasses import replace
from scipy.stats import norm
import extension_drill2_SOLUTIONS as e

# ── GIVEN: the capstone portfolio, base market, and the historical factor moves
PF = e.Portfolio([
    e.Bond((1,2,3,4,5),(4,4,4,4,104),150.0), e.InterestRateSwap((1,2,3,4,5),1.0,0.042,100.0),
    e.CreditDefaultSwap((1,2,3,4,5),1.0,100.0,120.0,100.0),
    e.TotalReturnSwap(e.Bond((1,2,3,4,5),(4,4,4,4,104),150.0),90.0),
    e.ForeignBond((1,2,3,4,5),(3,3,3,3,103),100.0,1.25), e.Swaption(2.0,(3,4,5),1.0,0.042,0.20,100.0),
    e.Repo(100.0,0.043,0.5), e.SecLending(100.0,20.0,(1,2,3),1.0), e.Loan(100.0,0.05,5,200.0),
    e.FXSpot(80.0,1.25), e.FXOption(1.0,1.30,0.12,1.25,100.0)])
BASE = e.Market()
N = 250
# historical factor-change matrix (N x 4): [rate_bp, spread_bp, d_vol, fx_ret]
FACTORS = np.array([[8*np.sin(i*0.3)+((i%11)-5), 5*np.cos(i*0.2)+((i%7)-3),
                     0.005*np.sin(i*0.4)+0.001*((i%3)-1), 0.01*np.sin(i*0.5)+0.002*((i%5)-2)]
                    for i in range(N)])


class MultiMethodVaR:
    def __init__(self, portfolio, base, factors):
        self.pf, self.base, self.F = portfolio, base, factors
        self.V0 = portfolio.value(base)

    # GIVEN: covariance of the factor moves, and the first-order sensitivity vector
    def covariance(self):
        return np.cov(self.F, rowvar=False)                 # 4x4
    def sensitivities(self):
        b, v0 = self.base, self.V0
        return np.array([self.pf.value(replace(b, rate_bump_bp=1)) - v0,
                         self.pf.value(replace(b, spread_bump_bp=1)) - v0,
                         (self.pf.value(replace(b, vol_shock=0.01)) - v0)/0.01,
                         (self.pf.value(replace(b, fx_shock=0.01)) - v0)/0.01])

    # GIVEN: historical simulation (full-reval over the actual scenarios)
    def historical_var(self, conf=0.995):
        pnl = sorted(self.pf.value(e.Scenario(*f).apply(self.base)) - self.V0 for f in self.F)
        return -pnl[int((1 - conf) * len(pnl))]

    # ═══════════════════════════════════════════════════════════════════════
    # TASK 1 — parametric / variance-covariance VaR (delta-normal)
    #   P&L std = sqrt(sᵀ · Σ · s)        (s = sensitivities, Σ = covariance)
    #   VaR(conf) = z · std,  z = norm.ppf(conf)
    #   EXPECTED: 99.5% -> 6.5238 ,  99% -> 5.8919
    # ═══════════════════════════════════════════════════════════════════════
    def parametric_var(self, conf=0.995):
        raise NotImplementedError

    # ═══════════════════════════════════════════════════════════════════════
    # TASK 2 — Monte Carlo VaR (Cholesky -> simulate -> FULL revalue -> percentile)
    #   L = cholesky(Σ)
    #   rng = np.random.default_rng(seed)
    #   draws = rng.standard_normal((n_paths, 4)) @ L.T        # ~ N(0, Σ)
    #   pnl_k = pf.value(Scenario(*draws[k]).apply(base)) − V0  # FULL reval
    #   VaR(conf) = −percentile(pnl, (1−conf)·100)
    #   EXPECTED (n_paths=10000, seed=42): 99.5% -> 6.3411 , 99% -> 5.8760
    # ═══════════════════════════════════════════════════════════════════════
    def monte_carlo_var(self, conf=0.995, n_paths=10000, seed=42):
        raise NotImplementedError

# ═══════════════════════════════════════════════════════════════════════════
# GRADING
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    v = MultiMethodVaR(PF, BASE, FACTORS)
    std = float(np.sqrt(v.sensitivities() @ v.covariance() @ v.sensitivities()))
    assert abs(std - 2.5326844) < 1e-5

    hist = v.historical_var(0.995)
    par995, par99 = v.parametric_var(0.995), v.parametric_var(0.99)
    mc995, mc99 = v.monte_carlo_var(0.995), v.monte_carlo_var(0.99)

    assert abs(hist   - 4.6781031) < 1e-5, hist
    assert abs(par995 - 6.5237627) < 1e-5, par995
    assert abs(par99  - 5.8919050) < 1e-5, par99
    assert abs(mc995  - 6.3411445) < 1e-4, mc995
    assert abs(mc99   - 5.8759691) < 1e-4, mc99
    assert par995 > hist and mc995 > hist          # normal assumption > bounded-historical tail

    print(f"P&L std:       {std:.4f}")
    print(f"99.5% VaR  —  historical {hist:.4f} | parametric {par995:.4f} | monte-carlo {mc995:.4f}")
    print(f"99.0% VaR  —                          parametric {par99:.4f} | monte-carlo {mc99:.4f}")
    print("\n✓ All checks passed.")
