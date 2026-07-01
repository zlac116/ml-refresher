"""
DRILL 13 — Three ways to compute VaR — SOLUTION KEY
Run: uv run python var_methods_drill_SOLUTIONS.py
"""
import numpy as np
from dataclasses import replace
from scipy.stats import norm
import extension_drill2_SOLUTIONS as e

PF = e.Portfolio([
    e.Bond((1,2,3,4,5),(4,4,4,4,104),150.0), e.InterestRateSwap((1,2,3,4,5),1.0,0.042,100.0),
    e.CreditDefaultSwap((1,2,3,4,5),1.0,100.0,120.0,100.0),
    e.TotalReturnSwap(e.Bond((1,2,3,4,5),(4,4,4,4,104),150.0),90.0),
    e.ForeignBond((1,2,3,4,5),(3,3,3,3,103),100.0,1.25), e.Swaption(2.0,(3,4,5),1.0,0.042,0.20,100.0),
    e.Repo(100.0,0.043,0.5), e.SecLending(100.0,20.0,(1,2,3),1.0), e.Loan(100.0,0.05,5,200.0),
    e.FXSpot(80.0,1.25), e.FXOption(1.0,1.30,0.12,1.25,100.0)])
BASE = e.Market()
N = 250
FACTORS = np.array([[8*np.sin(i*0.3)+((i%11)-5), 5*np.cos(i*0.2)+((i%7)-3),
                     0.005*np.sin(i*0.4)+0.001*((i%3)-1), 0.01*np.sin(i*0.5)+0.002*((i%5)-2)]
                    for i in range(N)])


class MultiMethodVaR:
    def __init__(self, portfolio, base, factors):
        self.pf, self.base, self.F = portfolio, base, factors
        self.V0 = portfolio.value(base)
    def covariance(self):
        return np.cov(self.F, rowvar=False)
    def sensitivities(self):
        b, v0 = self.base, self.V0
        return np.array([self.pf.value(replace(b, rate_bump_bp=1)) - v0,
                         self.pf.value(replace(b, spread_bump_bp=1)) - v0,
                         (self.pf.value(replace(b, vol_shock=0.01)) - v0)/0.01,
                         (self.pf.value(replace(b, fx_shock=0.01)) - v0)/0.01])
    def historical_var(self, conf=0.995):
        pnl = sorted(self.pf.value(e.Scenario(*f).apply(self.base)) - self.V0 for f in self.F)
        return -pnl[int((1 - conf) * len(pnl))]

    def parametric_var(self, conf=0.995):
        s, S = self.sensitivities(), self.covariance()
        std = np.sqrt(s @ S @ s)
        return float(norm.ppf(conf) * std)

    def monte_carlo_var(self, conf=0.995, n_paths=10000, seed=42):
        L = np.linalg.cholesky(self.covariance())
        rng = np.random.default_rng(seed)
        draws = rng.standard_normal((n_paths, 4)) @ L.T          # ~ N(0, Σ)
        pnl = np.array([self.pf.value(e.Scenario(*d).apply(self.base)) - self.V0 for d in draws])
        return float(-np.percentile(pnl, (1 - conf) * 100))


if __name__ == "__main__":
    v = MultiMethodVaR(PF, BASE, FACTORS)
    std = float(np.sqrt(v.sensitivities() @ v.covariance() @ v.sensitivities()))
    assert abs(std - 2.5326844) < 1e-5
    assert abs(v.historical_var(0.995) - 4.6781031) < 1e-5
    assert abs(v.parametric_var(0.995) - 6.5237627) < 1e-5
    assert abs(v.parametric_var(0.99)  - 5.8919050) < 1e-5
    assert abs(v.monte_carlo_var(0.995) - 6.3411445) < 1e-4
    assert abs(v.monte_carlo_var(0.99)  - 5.8759691) < 1e-4
    print("✓ Solution key — all checks passed.")
