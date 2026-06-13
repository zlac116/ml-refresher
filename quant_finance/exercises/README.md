# Quant Finance Exercises — 26 timed drills

Short, self-contained Python scripts (~15–20 min each) covering the everyday
quant toolkit: options, fixed income, futures, volatility, P&L attribution,
risk, portfolio construction, and XVA.

## How to use them

Each script has:

1. A **docstring** with objective, topics, time estimate, and the
   **expected output** when implemented correctly.
2. **Synthetic but realistic** data generated in-script (deterministic seeds).
3. **Function stubs** marked `# TODO:` for you to fill in.
4. **Assertion block** that grades your implementation.

```bash
cd quant_finance/exercises/options
python 01_black_scholes_pricer.py        # raises NotImplementedError until you fill it in
```

## Index

### `options/` — 4 exercises

| #  | File                                  | Focus                                              |
|----|---------------------------------------|----------------------------------------------------|
| 01 | `01_black_scholes_pricer.py`          | BSM call/put + Greeks (delta, gamma, vega, theta)  |
| 02 | `02_put_call_parity.py`               | Parity check + arbitrage detection                 |
| 03 | `03_implied_vol_newton.py`            | Implied vol via Newton-Raphson                     |
| 04 | `04_delta_gamma_hedge_pnl.py`         | Delta-gamma-theta P&L explain                      |

### `fixed_income/` — 6 exercises

| #  | File                                  | Focus                                              |
|----|---------------------------------------|----------------------------------------------------|
| 01 | `01_yield_curve_bootstrap.py`         | Bootstrap discount curve from par swap rates       |
| 02 | `02_duration_convexity.py`            | Macaulay/modified duration + convexity             |
| 03 | `03_forward_rates.py`                 | Forward rates from discount factors                |
| 04 | `04_bond_pricing_ytm.py`              | Clean/dirty/accrued + YTM back-out via brentq      |
| 05 | `05_loan_amortisation.py`             | Fixed-rate mortgage schedule + effective rate      |
| 06 | `06_deposit_fra.py`                   | ACT/360 money-market deposit + 3x6 FRA pricing     |

### `futures/` — 3 exercises

| #  | File                                  | Focus                                              |
|----|---------------------------------------|----------------------------------------------------|
| 01 | `01_equity_index_fair_value.py`       | ES/MES futures fair value + cash-and-carry arb    |
| 02 | `02_oil_contango_roll_yield.py`       | WTI contango/backwardation, roll yield, cal spread |
| 03 | `03_bond_futures_ctd.py`              | Conversion factor, invoice price, CTD selection    |

### `volatility/` — 3 exercises

| #  | File                                  | Focus                                              |
|----|---------------------------------------|----------------------------------------------------|
| 01 | `01_sabr_smile_calibration.py`        | Fit SABR (alpha, rho, nu) to a vol smile           |
| 02 | `02_hist_ewma_garch_vol.py`           | Compare hist vs EWMA(0.94) vs GARCH(1,1) vol       |
| 03 | `03_vol_surface_interp.py`            | Bilinear interp via RegularGridInterpolator        |

### `pnl_attribution/` — 4 exercises

| #  | File                                  | Focus                                              |
|----|---------------------------------------|----------------------------------------------------|
| 01 | `01_greek_taylor_explain.py`          | delta + gamma + vega + theta Taylor explain        |
| 02 | `02_portfolio_factor_attrib.py`       | Aggregate Greeks across a small derivatives book   |
| 03 | `03_new_trade_attrib.py`              | Old-book market move + new-trade execution P&L     |
| 04 | `04_finite_diff_attribution.py`       | FD Greeks (bump-and-revalue) + full reval attrib   |

### `risk/` — 3 exercises

| #  | File                                  | Focus                                              |
|----|---------------------------------------|----------------------------------------------------|
| 01 | `01_historical_var_es.py`             | Historical VaR + Expected Shortfall                |
| 02 | `02_parametric_var.py`                | Parametric VaR + component (Euler) decomposition   |
| 03 | `03_sensitivities_dv01_vega.py`       | DV01 buckets, vega buckets, cross-gamma matrix     |

### `portfolio/` — 3 exercises

| #  | File                                  | Focus                                              |
|----|---------------------------------------|----------------------------------------------------|
| 01 | `01_mean_variance_opt.py`             | Markowitz QP via SLSQP (long-only)                 |
| 02 | `02_risk_parity.py`                   | ERC portfolio (Maillard-Roncalli-Teiletche)        |
| 03 | `03_backtest_rebalance.py`            | Buy-and-hold vs monthly-rebalanced equal-weight    |

### `xva/` — 2 exercises

| #  | File                                  | Focus                                              |
|----|---------------------------------------|----------------------------------------------------|
| 01 | `01_cva_from_exposure.py`             | CVA from EPE + survival probs; hazard from CDS     |
| 02 | `02_fva_basics.py`                    | FVA on a one-way uncollateralised trade            |

## Conventions used

- ACT/365 day-count unless explicitly ACT/360 (money market drills).
- All `np.random` calls seeded for reproducibility.
- Implied vol / strike grids in absolute decimals (e.g. `0.20` = 20%).
- All asserts are tolerance-based; your impl just has to be canonical, not bit-exact.

## References (latest docs)

- numpy: <https://numpy.org/doc/stable/>
- scipy.optimize: <https://docs.scipy.org/doc/scipy/reference/optimize.html>
- scipy.stats: <https://docs.scipy.org/doc/scipy/reference/stats.html>
- scipy.interpolate: <https://docs.scipy.org/doc/scipy/reference/interpolate.html>
- pandas: <https://pandas.pydata.org/docs/user_guide/>

## Reading order if new to the topic

1. `options/01` → `options/02` (parity is the cheapest sanity check on BSM).
2. `volatility/02` before `volatility/01` (intuition before calibration).
3. `risk/01` → `risk/02` (empirical before parametric).
4. `pnl_attribution/01` → `02` → `03` → `04` (analytical → portfolio → execution → numerical).
5. `fixed_income/04` → `01` → `03` → `02` (price first, then curve, then forwards, then risk).
