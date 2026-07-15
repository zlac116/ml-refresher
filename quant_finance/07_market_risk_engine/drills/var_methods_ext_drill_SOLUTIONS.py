"""
DRILL 13b — Four MORE ways to compute VaR — SOLUTION KEY
Run: uv run python var_methods_ext_drill_SOLUTIONS.py

Extends var_methods_drill_SOLUTIONS.MultiMethodVaR (historical/parametric/MC come
by inheritance). Adds: Expected Shortfall, EWMA, Cornish-Fisher, Delta-Gamma.
"""
import numpy as np
from scipy.stats import norm, skew, kurtosis
import extension_drill2_SOLUTIONS as e
from var_methods_drill_SOLUTIONS import MultiMethodVaR, PF, BASE, FACTORS


class ExtendedVaR(MultiMethodVaR):

    # TASK 1 — Expected Shortfall (CVaR): average loss beyond VaR.
    def expected_shortfall(self, conf=0.975, method="parametric"):
        if method == "parametric":
            # normal, zero-mean closed form:  σ · φ(z) / (1−conf)
            std = np.sqrt(self.sensitivities() @ self.covariance() @ self.sensitivities())
            z = norm.ppf(conf)
            return float(std * norm.pdf(z) / (1 - conf))
        if method == "historical":
            pnl = np.array([self.pf.value(e.Scenario(*f).apply(self.base)) - self.V0
                            for f in self.F])
        elif method == "monte_carlo":
            L = np.linalg.cholesky(self.covariance())
            rng = np.random.default_rng(42)
            draws = rng.standard_normal((10000, 4)) @ L.T
            pnl = np.array([self.pf.value(e.Scenario(*d).apply(self.base)) - self.V0
                            for d in draws])
        else:
            raise ValueError(method)
        q = np.percentile(pnl, (1 - conf) * 100)          # the VaR threshold (neg)
        return float(-pnl[pnl <= q].mean())               # mean of the tail, flipped

    # TASK 2 — EWMA / RiskMetrics: recent days weighted more, same delta-normal formula.
    def ewma_covariance(self, lam=0.94):
        F = self.F
        T = len(F)
        age = np.arange(T)[::-1]                # newest row -> age 0 -> biggest weight
        w = (1 - lam) * lam ** age
        w /= w.sum()
        Xc = F - F.mean(axis=0)
        return (w[:, None] * Xc).T @ Xc         # 4x4 weighted covariance

    def ewma_var(self, conf=0.995, lam=0.94):
        s = self.sensitivities()
        return float(norm.ppf(conf) * np.sqrt(s @ self.ewma_covariance(lam) @ s))

    # TASK 3 — Cornish-Fisher / modified VaR: bend z by the P&L's skew & excess kurtosis.
    @staticmethod
    def cf_quantile(z, S, K):                   # reduces to z when S=K=0 (normal)
        return (z
                + (z**2 - 1) / 6 * S
                + (z**3 - 3*z) / 24 * K
                - (2*z**3 - 5*z) / 36 * S**2)

    def cornish_fisher_var(self, conf=0.995):
        p = self.F @ self.sensitivities()
        S, K = skew(p), kurtosis(p)             # kurtosis() is EXCESS (Fisher)
        z = norm.ppf(1 - conf)                  # lower-tail quantile (negative)
        return float(-(p.mean() + p.std(ddof=1) * self.cf_quantile(z, S, K)))

    # TASK 4 — Delta-Gamma: add the 2nd-order term  ½ Δfᵀ Γ Δf  (captures convexity).
    def gamma_matrix(self, h=(1.0, 1.0, 0.01, 0.01)):
        n = 4
        G = np.zeros((n, n))
        V0 = self.V0
        val = lambda vec: self.pf.value(e.Scenario(*vec).apply(self.base))
        for i in range(n):                      # diagonal curvature
            ei = np.zeros(n); ei[i] = h[i]
            G[i, i] = (val(ei) - 2 * V0 + val(-ei)) / h[i] ** 2
        for i in range(n):                      # cross-gammas
            for j in range(i + 1, n):
                ei = np.zeros(n); ei[i] = h[i]
                ej = np.zeros(n); ej[j] = h[j]
                gij = (val(ei + ej) - val(ei) - val(ej) + V0) / (h[i] * h[j])
                G[i, j] = G[j, i] = gij
        return G

    def delta_gamma_var(self, conf=0.995, n_paths=10000, seed=42):
        L = np.linalg.cholesky(self.covariance())
        rng = np.random.default_rng(seed)
        draws = rng.standard_normal((n_paths, 4)) @ L.T
        G = self.gamma_matrix()
        quad = 0.5 * np.einsum('ki,ij,kj->k', draws, G, draws)   # ½ Δfᵀ Γ Δf per path
        pnl = draws @ self.sensitivities() + quad
        return float(-np.percentile(pnl, (1 - conf) * 100))


if __name__ == "__main__":
    v = ExtendedVaR(PF, BASE, FACTORS)
    s, Sig = v.sensitivities(), v.covariance()
    std = float(np.sqrt(s @ Sig @ s))
    par995, hist995, mc995 = v.parametric_var(0.995), v.historical_var(0.995), v.monte_carlo_var(0.995)

    # ── TASK 1 · Expected Shortfall ──────────────────────────────────────────
    es_p975 = v.expected_shortfall(0.975, "parametric")
    es_p995 = v.expected_shortfall(0.995, "parametric")
    es_h995 = v.expected_shortfall(0.995, "historical")
    es_m995 = v.expected_shortfall(0.995, "monte_carlo")
    # (i) coherence: ES >= VaR at the SAME confidence, for EVERY engine
    assert es_p995 >= par995,  "parametric ES must dominate parametric VaR"
    assert es_h995 >= hist995, "historical ES must dominate historical VaR"
    assert es_m995 >= mc995,   "MC ES must dominate MC VaR"
    # (ii) monotone in confidence: a deeper tail can only raise the average loss
    assert es_p995 > es_p975,  "ES must increase with confidence"
    # (iii) closed-form obeys the normal ES/VaR identity  φ(z)/((1-α)·z)
    ratio = norm.pdf(norm.ppf(0.995)) / (0.005 * norm.ppf(0.995))
    assert abs(es_p995 / par995 - ratio) < 1e-9, "normal ES/VaR ratio identity broken"
    # (iv) closed-form agrees with an INDEPENDENT simulation of the same normal
    zsim = np.random.default_rng(0).standard_normal(2_000_000) * std
    es_sim = -zsim[zsim <= np.percentile(zsim, 0.5)].mean()
    assert abs(es_p995 - es_sim) < 0.02, "closed-form ES disagrees with simulation"
    # (v) frozen exact values
    assert abs(es_p975 - 5.9209167) < 1e-4, es_p975
    assert abs(es_p995 - 7.3243931) < 1e-4, es_p995

    # ── TASK 2 · EWMA / RiskMetrics ──────────────────────────────────────────
    ewma995 = v.ewma_var(0.995)
    Sig_ew = v.ewma_covariance(0.94)
    # (i) it is a valid covariance: symmetric and positive semi-definite
    assert np.allclose(Sig_ew, Sig_ew.T),            "EWMA covariance not symmetric"
    assert np.linalg.eigvalsh(Sig_ew).min() > -1e-12, "EWMA covariance not PSD"
    # (ii) LIMIT: as λ→1 weights become uniform → EWMA VaR → ML(÷N)-covariance VaR
    ml = (v.F - v.F.mean(0)).T @ (v.F - v.F.mean(0)) / len(v.F)
    ml_var = norm.ppf(0.995) * np.sqrt(s @ ml @ s)
    assert abs(v.ewma_var(0.995, lam=1 - 1e-9) - ml_var) < 1e-3, "λ→1 limit does not hold"
    # (iii) with real (non-uniform) weighting it must differ from equal-weighted VaR
    assert abs(ewma995 - par995) > 1e-6, "EWMA should differ from equal-weighted parametric"
    # (iv) frozen exact value
    assert abs(ewma995 - 6.3219789) < 1e-4, ewma995

    # ── TASK 3 · Cornish-Fisher ──────────────────────────────────────────────
    cf995 = v.cornish_fisher_var(0.995)
    p = v.F @ s
    S_, K_ = float(skew(p)), float(kurtosis(p))
    zc = norm.ppf(0.005)
    # (i) DEGENERATE: with zero skew & kurtosis, CF collapses to the normal quantile
    assert abs(v.cf_quantile(zc, 0.0, 0.0) - zc) < 1e-12, "CF must reduce to normal at S=K=0"
    # (ii) data is thin-tailed (excess kurtosis < 0) → CF must sit BELOW parametric
    assert K_ < 0 and cf995 < par995, "thin tail should pull CF below parametric"
    # (iii) CF should be a BETTER estimate of the real tail than the normal, i.e.
    #       closer to the historical VaR than plain parametric is
    assert abs(cf995 - hist995) < abs(par995 - hist995), "CF should beat normal vs historical"
    # (iv) frozen exact value
    assert abs(cf995 - 5.2915787) < 1e-4, cf995

    # ── TASK 4 · Delta-Gamma ─────────────────────────────────────────────────
    dg995 = v.delta_gamma_var(0.995)
    G = v.gamma_matrix()
    # (i) the gamma matrix is symmetric (mixed partials commute)
    assert np.allclose(G, G.T), "gamma matrix not symmetric"
    # (ii) adding gamma must move CLOSER to full revaluation than delta-only does
    L = np.linalg.cholesky(Sig)
    draws = np.random.default_rng(42).standard_normal((10000, 4)) @ L.T
    lin995 = float(-np.percentile(draws @ s, 0.5))          # delta-only linear VaR
    assert abs(dg995 - mc995) < abs(lin995 - mc995), "gamma term should improve on delta-only"
    # (iii) and it should land within a few % of the full-reval MC number
    assert abs(dg995 - mc995) / mc995 < 0.02, "delta-gamma too far from full-reval MC"
    # (iv) frozen exact value
    assert abs(dg995 - 6.3753069) < 1e-4, dg995

    # ── report ───────────────────────────────────────────────────────────────
    print("        method                 99.5%     check")
    print(f"  parametric VaR (base) ...... {par995:7.4f}")
    print(f"  historical VaR (base) ...... {hist995:7.4f}")
    print(f"  Expected Shortfall (par) ... {es_p995:7.4f}   ES/VaR = {es_p995/par995:.4f}  (identity {ratio:.4f})")
    print(f"  Expected Shortfall (hist) .. {es_h995:7.4f}   >= hist VaR {hist995:.4f}")
    print(f"  EWMA VaR ................... {ewma995:7.4f}   λ→1 limit ✓ = {ml_var:.4f}")
    print(f"  Cornish-Fisher VaR ......... {cf995:7.4f}   skew {S_:+.3f}, exc-kurt {K_:+.3f} → pulls toward hist")
    print(f"  Delta-Gamma VaR ............ {dg995:7.4f}   delta-only {lin995:.4f} → full MC {mc995:.4f}")
    print("\n✓ Solution key — all invariants, identities, limits and frozen values passed.")
