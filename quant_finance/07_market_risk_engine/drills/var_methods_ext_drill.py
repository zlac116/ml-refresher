"""
DRILL 13b — Four MORE ways to compute VaR  (the popular extensions)
====================================================================
Extends DRILL 13 (var_methods_drill.py). You already have historical, parametric
(delta-normal) and Monte-Carlo. This drill adds the four extensions every risk
desk actually quotes. It SUBCLASSES MultiMethodVaR, so those three come for free
via inheritance — you only implement the four new methods below.

MENTAL MODEL — each new method fixes ONE weakness you saw in DRILL 13
  • Expected Shortfall (ES/CVaR) — VaR tells you the THRESHOLD; ES tells you the
    AVERAGE loss once you're past it. Coherent, and the Basel FRTB standard.
  • Delta-Gamma — parametric used sensitivities only (linear). Add the GAMMA
    (2nd-order) term and you capture the convexity delta-normal throws away. This
    is the exact "convexity gap" from our discussion, made numeric.
  • Cornish-Fisher (modified VaR) — plain parametric assumed NORMAL and so
    overstated the bounded, thin-tailed data. CF bends the z-quantile using the
    P&L's own SKEW and KURTOSIS, pulling it back toward the real tail.
  • EWMA / RiskMetrics — np.cov weights all 250 days equally. EWMA weights RECENT
    days more (lambda=0.94), so the covariance tracks the current regime.

BUILD ORDER (each is ~10 min; do them top to bottom)
  1. expected_shortfall  — easiest; just average the tail you already produce.
  2. ewma_var            — same delta-normal formula, only Sigma changes.
  3. cornish_fisher_var  — one quantile formula, fed skew & kurtosis.
  4. delta_gamma_var     — build the Gamma matrix, then quadratic P&L on MC draws.

CITATIONS (say these)
  ES / coherence ......... Artzner-Delbaen-Eber-Heath 1999; Basel FRTB (ES 97.5%)
  Delta-Gamma / mod VaR .. Zangari 1996 (RiskMetrics Monitor)
  Cornish-Fisher ......... Cornish & Fisher 1938; Zangari 1996
  EWMA ................... RiskMetrics Technical Document, J.P. Morgan 1996 (lambda=0.94)

RUN
    uv run python var_methods_ext_drill.py

GRADING
  The asserts below check INVARIANTS that must hold by theory (e.g. ES >= VaR),
  not magic numbers — so they don't leak the answer. Once your prints look right,
  freeze your own exact values into asserts the way var_methods_drill.py does.

SCOPE: ~45 min. If you finish early, the one-line STRETCH notes point further.
"""
import numpy as np
from scipy.stats import norm, skew, kurtosis
import extension_drill2_SOLUTIONS as e
from var_methods_drill import MultiMethodVaR, PF, BASE, FACTORS


class ExtendedVaR(MultiMethodVaR):
    # inherits: covariance(), sensitivities(), historical_var(),
    #           parametric_var(), monte_carlo_var(), V0, pf, base, F

    # ═══════════════════════════════════════════════════════════════════════
    # TASK 1 — Expected Shortfall  (a.k.a. CVaR / Conditional VaR / TVaR)
    #   Definition:  ES_conf = average loss GIVEN loss is worse than VaR_conf
    #                        = mean of the worst (1-conf) fraction of P&Ls
    #
    #   HISTORICAL / MC route (from a P&L array `pnl`, losses are negative):
    #       q   = np.percentile(pnl, (1-conf)*100)     # the VaR threshold (neg)
    #       ES  = -pnl[pnl <= q].mean()                # mean of the tail, flipped
    #
    #   PARAMETRIC (normal, zero-mean) closed form:
    #       z   = norm.ppf(conf)
    #       ES  = pnl_std * norm.pdf(z) / (1 - conf)   # note: pdf, not ppf
    #
    #   INVARIANT to remember: ES >= VaR at the same conf, ALWAYS.
    #   STRETCH: also return ES from the historical engine and compare the three.
    # ═══════════════════════════════════════════════════════════════════════
    def expected_shortfall(self, conf=0.975, method="parametric"):
        if method == "parametric":
            sensi, cov = self.sensitivities(), self.covariance()
            pnl_std = np.sqrt(sensi @ cov @ sensi)
            z = norm.ppf(conf)
            breakpoint()
            return pnl_std * norm.pdf(z) / (1 - conf)
        elif method == "historical":
            pnl = np.array([self.pf.value(e.Scenario(*f).apply(self.base)) - self.V0 for f in self.F])
        elif method == "monte_carlo":
            L = np.linalg.cholesky(self.covariance)
            rng = np.random.default_rng(42)
            draws = rng.standard_normal((10_000, 4)) @ L.T
            pnl = np.array([self.pf.value(e.Scenario(*d).apply(self.base)) - self.V0 for d in draws])
        
        q = np.percentile(pnl, (1 - conf) * 100)
        return -pnl[pnl <= q].mean()


    # ═══════════════════════════════════════════════════════════════════════
    # TASK 2 — EWMA / RiskMetrics parametric VaR
    #   Same delta-normal machine (z * sqrt(sᵀ Σ s)) — only Σ changes: recent
    #   days get exponentially more weight.
    #
    #   Weights (most recent row = last row of F gets the largest weight):
    #       T   = len(F)
    #       age = np.arange(T)[::-1]          # T-1 for oldest ... 0 for newest
    #       w   = (1-lam) * lam**age
    #       w  /= w.sum()                     # normalise to sum 1
    #
    #   Weighted covariance (RiskMetrics assumes zero mean; keep it simple):
    #       Xc  = F - F.mean(axis=0)          # or drop the mean per RiskMetrics
    #       Sig = (w[:,None]*Xc).T @ Xc       # 4x4 weighted covariance
    #
    #   Then reuse the delta-normal formula with this Sig and self.sensitivities().
    #   STRETCH: sweep lam in {0.90, 0.94, 0.97} and watch the VaR move.
    # ═══════════════════════════════════════════════════════════════════════
    def ewma_covariance(self, lam=0.94):
        # TODO: return the 4x4 exponentially-weighted covariance (see weights above).
        raise NotImplementedError

    def ewma_var(self, conf=0.995, lam=0.94):
        # TODO: z * sqrt(s @ self.ewma_covariance(lam) @ s).
        raise NotImplementedError

    # ═══════════════════════════════════════════════════════════════════════
    # TASK 3 — Cornish-Fisher / Modified VaR
    #   Correct the normal quantile for the P&L's own skew (S) and excess
    #   kurtosis (K), so a non-normal tail is priced without simulation.
    #
    #   p = F @ self.sensitivities()          # the historical P&L series
    #   S = skew(p)                           # scipy.stats.skew
    #   K = kurtosis(p)                       # scipy.stats.kurtosis -> EXCESS (Fisher)
    #   z = norm.ppf(1 - conf)                # lower-tail quantile (negative)
    #   z_cf = ( z
    #            + (z**2 - 1)/6      * S
    #            + (z**3 - 3*z)/24   * K
    #            - (2*z**3 - 5*z)/36 * S**2 )
    #   mVaR = -(p.mean() + p.std(ddof=1) * z_cf)
    #
    #   READ-BACK: with thin-tailed (negative-K) data, z_cf is closer to 0 than z,
    #   so CF VaR sits BELOW plain parametric — nearer the historical number.
    # ═══════════════════════════════════════════════════════════════════════
    @staticmethod
    def cf_quantile(z, S, K):
        # TODO: return the Cornish-Fisher adjusted quantile (formula above).
        #       MUST satisfy cf_quantile(z, 0, 0) == z  (reduces to normal).
        raise NotImplementedError

    def cornish_fisher_var(self, conf=0.995):
        # TODO: p = F @ s;  S, K = skew(p), kurtosis(p);  z = norm.ppf(1-conf)
        #       return -(p.mean() + p.std(ddof=1) * self.cf_quantile(z, S, K))
        raise NotImplementedError

    # ═══════════════════════════════════════════════════════════════════════
    # TASK 4 — Delta-Gamma VaR  (the convexity term, via MC draws)
    #   Taylor to 2nd order:  P&L(Δf) ≈ s·Δf + ½ Δfᵀ Γ Δf
    #
    #   (a) Build the Gamma matrix by finite differences on Scenario bumps
    #       (Scenario units == draw units == sensitivity units, so it's consistent).
    #       For the diagonal (curvature per factor i), with a bump h and unit vec e_i:
    #           V0     = self.V0
    #           Vp     = pf.value(Scenario(+h in slot i, 0 elsewhere).apply(base))
    #           Vm     = pf.value(Scenario(-h in slot i, 0 elsewhere).apply(base))
    #           Γ_ii   = (Vp - 2*V0 + Vm) / h**2
    #       Pick h per factor near the sensitivity bump sizes (rate/spread ~1 bp,
    #       vol/fx ~0.01). Diagonal-only Γ is fine for CORE.
    #       STRETCH: add cross terms  Γ_ij = (V(+h e_i +h e_j) - V(+h e_i)
    #                                          - V(+h e_j) + V0) / h**2.
    #
    #   (b) Regenerate the SAME MC draws as monte_carlo_var (Cholesky, seed), then:
    #           quad = 0.5 * np.einsum('ki,ij,kj->k', draws, Gamma, draws)
    #           pnl  = draws @ self.sensitivities() + quad
    #           VaR  = -np.percentile(pnl, (1-conf)*100)
    #       Compare against `draws @ s` alone (delta-only) — the gap IS the gamma P&L.
    # ═══════════════════════════════════════════════════════════════════════
    def gamma_matrix(self, h=(1.0, 1.0, 0.01, 0.01)):
        # TODO: return the 4x4 Gamma via finite differences (diagonal ok for CORE).
        raise NotImplementedError

    def delta_gamma_var(self, conf=0.995, n_paths=10000, seed=42):
        # TODO: draws (as in monte_carlo_var) -> quadratic P&L -> percentile.
        raise NotImplementedError


# ═══════════════════════════════════════════════════════════════════════════
# GRADING — theory checks (invariants / identities / limits), NOT magic numbers.
# Each must hold for ANY correct implementation. When they pass, your method is
# almost certainly right — then freeze your own exact values like the base drill.
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import numpy as np
    from scipy.stats import norm, kurtosis
    v = ExtendedVaR(PF, BASE, FACTORS)
    s, Sig = v.sensitivities(), v.covariance()
    std = float(np.sqrt(s @ Sig @ s))
    par995, hist995, mc995 = v.parametric_var(0.995), v.historical_var(0.995), v.monte_carlo_var(0.995)

    # ── TASK 1 · Expected Shortfall ──────────────────────────────────────────
    es_p975, es_p995 = v.expected_shortfall(0.975), v.expected_shortfall(0.995)
    es_h995 = v.expected_shortfall(0.995, "historical")
    assert es_p995 >= par995,  "ES must be >= VaR at the same confidence"
    assert es_h995 >= hist995, "historical ES must be >= historical VaR"
    assert es_p995 > es_p975,  "ES must increase with confidence"
    # closed-form must match the normal ES/VaR identity  φ(z)/((1-α)·z)
    ratio = norm.pdf(norm.ppf(0.995)) / (0.005 * norm.ppf(0.995))
    assert abs(es_p995 / par995 - ratio) < 1e-6, "normal ES/VaR ratio identity broken"
    # ...and agree with an independent simulation of the same normal
    zsim = np.random.default_rng(0).standard_normal(2_000_000) * std
    es_sim = -zsim[zsim <= np.percentile(zsim, 0.5)].mean()
    assert abs(es_p995 - es_sim) < 0.02, "closed-form ES disagrees with simulation"

    # ── TASK 2 · EWMA / RiskMetrics ──────────────────────────────────────────
    ewma995 = v.ewma_var(0.995)
    Sig_ew = v.ewma_covariance(0.94)
    assert np.allclose(Sig_ew, Sig_ew.T),             "EWMA covariance not symmetric"
    assert np.linalg.eigvalsh(Sig_ew).min() > -1e-12, "EWMA covariance not PSD"
    # LIMIT: as λ→1 the weights go uniform → EWMA VaR → ML(÷N)-covariance VaR
    ml = (v.F - v.F.mean(0)).T @ (v.F - v.F.mean(0)) / len(v.F)
    ml_var = norm.ppf(0.995) * np.sqrt(s @ ml @ s)
    assert abs(v.ewma_var(0.995, lam=1 - 1e-9) - ml_var) < 1e-3, "λ→1 limit does not hold"
    assert abs(ewma995 - par995) > 1e-6, "EWMA should differ from equal-weighted parametric"

    # ── TASK 3 · Cornish-Fisher ──────────────────────────────────────────────
    cf995 = v.cornish_fisher_var(0.995)
    K_ = float(kurtosis(v.F @ s))
    assert abs(v.cf_quantile(norm.ppf(0.005), 0.0, 0.0) - norm.ppf(0.005)) < 1e-12, \
        "CF must reduce to the normal quantile at S=K=0"
    assert K_ < 0 and cf995 < par995, "thin tail (excess-kurt<0) should pull CF below parametric"
    assert abs(cf995 - hist995) < abs(par995 - hist995), "CF should beat normal vs historical"

    # ── TASK 4 · Delta-Gamma ─────────────────────────────────────────────────
    dg995 = v.delta_gamma_var(0.995)
    G = v.gamma_matrix()
    assert np.allclose(G, G.T), "gamma matrix not symmetric"
    L = np.linalg.cholesky(Sig)
    draws = np.random.default_rng(42).standard_normal((10000, 4)) @ L.T
    lin995 = float(-np.percentile(draws @ s, 0.5))     # delta-only linear VaR
    assert abs(dg995 - mc995) < abs(lin995 - mc995), "gamma term should improve on delta-only"
    assert abs(dg995 - mc995) / mc995 < 0.02,         "delta-gamma too far from full-reval MC"

    print(f"parametric VaR 99.5% .... {par995:.4f}   (inherited baseline)")
    print(f"ES 97.5% / 99.5% ........ {es_p975:.4f} / {es_p995:.4f}   (ES/VaR identity ✓)")
    print(f"EWMA VaR 99.5% .......... {ewma995:.4f}   (λ→1 limit ✓)")
    print(f"Cornish-Fisher VaR 99.5%  {cf995:.4f}   (excess-kurt {K_:+.3f} → pulls toward hist)")
    print(f"Delta-Gamma VaR 99.5% ... {dg995:.4f}   (delta-only {lin995:.4f} → full MC {mc995:.4f})")
    print("\n✓ All theory checks passed. Now freeze your exact numbers into asserts.")
