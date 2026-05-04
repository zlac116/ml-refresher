"""Smoke tests — run from project root with:
    /path/to/python -m pytest tests/

Or to run directly without pytest:
    /path/to/python tests/test_smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.curves import Curve, par_swap_rate
from src.trades import IRS
from src.pricers.irs import price_irs, dv01_irs
from src.pricers.lmm import lmm_caplet_price
from src.vol.sabr import sabr_atm_alpha, sabr_lognormal_vol
from src.vol.lmm_calibration import bootstrap_lmm_caplet_vols, rebonato_correlation


def _flat_curve(rate: float = 0.04) -> Curve:
    return Curve(
        tenors_y=np.array([0.083, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30]),
        zero_rates=np.full(11, rate),
        name="OIS_flat",
    )


def test_par_swap_pv_is_zero():
    """At-par swap should price to ~0."""
    c = _flat_curve(0.04)
    par_K, _ = par_swap_rate(c, c, 0.0, 5.0, 1.0)
    trade = IRS(trade_id="T", notional=1e8, currency="USD", pay_receive="payer",
                fixed_K=par_K, T_start=0.0, T_end=5.0, pay_freq=1.0)
    pv = price_irs(trade, c)
    assert abs(pv) < 1.0, f"expected |PV| < $1; got {pv}"


def test_dv01_sign_payer_is_negative():
    """Payer swap is short-duration → DV01 should be negative under our convention."""
    c = _flat_curve(0.04)
    trade = IRS(trade_id="T", notional=1e8, currency="USD", pay_receive="payer",
                fixed_K=0.04, T_start=0.0, T_end=5.0, pay_freq=1.0)
    dv01 = dv01_irs(trade, c, c)
    assert dv01 < 0, f"expected payer DV01 < 0 under (pv_dn - pv_up)/2 convention; got {dv01}"


def test_lmm_caplet_repricing_within_3_sigma():
    """LMM MC caplet vs Black-76 closed form should match within ±3σ MC."""
    c = _flat_curve(0.04)
    N = 6
    tenor_dates = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    deltas = np.diff(tenor_dates)
    L0 = np.full(N, 0.04)
    sigma = np.full(N, 0.20)
    rho = rebonato_correlation(tenor_dates, beta=0.05, rho_inf=0.5)
    P_0_TN = float(c.DF(tenor_dates[-1]))

    # Caplet on L_2 ATM
    K = 0.04
    mc_price, mc_se = lmm_caplet_price(L0, sigma, rho, deltas, tenor_dates,
                                        K=K, i=2, P_0_TN=P_0_TN,
                                        n_steps=120, n_paths=10_000, seed=1)
    # Black-76 closed form
    T_i, T_iplus1 = tenor_dates[2], tenor_dates[3]
    F = 0.04
    sig = 0.20
    d1 = (np.log(F/K) + 0.5*sig**2*T_i) / (sig*np.sqrt(T_i))
    d2 = d1 - sig*np.sqrt(T_i)
    b76 = deltas[2] * float(c.DF(T_iplus1)) * (F*norm.cdf(d1) - K*norm.cdf(d2))

    diff_sigma = (mc_price - b76) / mc_se
    assert abs(diff_sigma) < 3.0, (
        f"LMM caplet vs Black-76 disagree by {diff_sigma:+.2f}σ "
        f"(MC={mc_price:.6f} ± {mc_se:.6f}, B76={b76:.6f})"
    )


def test_sabr_atm_round_trip():
    """sabr_atm_alpha(σ_ATM) → α; sabr_lognormal_vol(F, F, T, α) should recover σ_ATM."""
    F, T, beta, rho, nu = 0.04, 1.0, 0.5, -0.3, 0.4
    sigma_atm = 0.30
    alpha = sabr_atm_alpha(sigma_atm, F, T, beta, rho, nu)
    vol_recovered = sabr_lognormal_vol(F, F, T, alpha, beta, rho, nu)
    assert abs(vol_recovered - sigma_atm) < 1e-6, (
        f"SABR ATM round-trip failed: sigma_atm={sigma_atm}, recovered={vol_recovered}"
    )


def test_lmm_bootstrap_round_trip_flat_caps():
    """Flat cap-vol surface with T0=0 should bootstrap to flat per-period vols.

    With T0 > 0 the first period has different length than its cap-vol window so
    σ_1 ≠ σ_cap. Setting T0 = 0 with cap maturities at integer years aligns the
    windows and the bootstrap should recover σ_i = σ_cap exactly.
    """
    cap_maturities = np.array([1.0, 2.0, 3.0, 4.0])
    cap_vols = np.array([0.20, 0.20, 0.20, 0.20])
    sigma_inst = bootstrap_lmm_caplet_vols(cap_maturities, cap_vols, T0=0.0)
    assert np.allclose(sigma_inst, 0.20, atol=1e-10), (
        f"flat caps with T0=0 should give flat per-period σ; got {sigma_inst}"
    )

    # Also verify the cumulative-variance identity holds for the realistic T0=1.0 case
    sigma_inst_2 = bootstrap_lmm_caplet_vols(np.array([2.0, 3.0, 4.0]),
                                              np.array([0.20, 0.20, 0.20]), T0=1.0)
    cum_var = np.cumsum(sigma_inst_2**2 * np.diff(np.array([1.0, 2.0, 3.0, 4.0])))
    target_var = 0.04 * np.array([2.0, 3.0, 4.0])
    assert np.allclose(cum_var, target_var, atol=1e-10), (
        f"cumulative variance identity violated: cum_var={cum_var}, target={target_var}"
    )


def test_html_report_no_crash():
    """Ensure html_report doesn't crash on the CSS braces (audit Bug 3)."""
    import pandas as pd
    from src.reports import html_report

    df = pd.DataFrame({"col": [1.0, 2.0]})
    html = html_report("title", [("section", df)])
    assert "<title>title</title>" in html
    assert "<h2>section</h2>" in html


if __name__ == "__main__":
    # Run all tests directly when invoked as a script
    tests = [
        test_par_swap_pv_is_zero,
        test_dv01_sign_payer_is_negative,
        test_lmm_caplet_repricing_within_3_sigma,
        test_sabr_atm_round_trip,
        test_lmm_bootstrap_round_trip_flat_caps,
        test_html_report_no_crash,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    sys.exit(failed)
