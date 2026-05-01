# quant_finance — Interview-prep notebooks for IB & AM quant roles

Self-contained notebooks for the **key** financial risk, valuation, and modelling
techniques. Each notebook follows the same shape:

1. **Why this matters** — interview context + real-world use
2. **30-second concept**
3. **Mathematical derivation** — concise, just the bits an interviewer expects
4. **Python implementation from scratch** — no library black-boxes on first pass
5. **Validation against a closed-form / QuantLib** — sanity check
6. **Worked example on real data** — yfinance / FRED / your existing crypto panel
7. **Exercises** with hidden solutions
8. **Interview Q&A** — typical questions with crisp answers
9. **Pitfalls reference card**

Every notebook is **expert-reviewed** by a quant-finance subagent before being
marked complete. See `EXPERT_REVIEWS.md` for the audit log.

## Curriculum

### Tier 1 — Core (always asked at IB quant screens)

| Notebook | Topic |
|---|---|
| `01_options/01_black_scholes.ipynb` | Black-Scholes from first principles + closed-form |
| `01_options/02_greeks.ipynb` | Delta, gamma, vega, theta, rho — analytic + finite-diff |
| `01_options/03_binomial_trees.ipynb` | CRR + Jarrow-Rudd; American-style early exercise |
| `01_options/04_monte_carlo_pricing.ipynb` | MC + variance reduction (antithetic, control variates) |
| `01_options/05_implied_vol_surface.ipynb` | Inversion via Brent; smile, skew, term structure |
| `02_risk/01_var_methods.ipynb` | Parametric, historical, MC VaR — when each fails |
| `02_risk/02_expected_shortfall.ipynb` | ES vs VaR; coherence; FRTB context |
| `03_fixed_income/01_bond_pricing.ipynb` | YTM, dirty/clean price, accrued interest |
| `03_fixed_income/02_duration_convexity_krd.ipynb` | Modified duration, convexity, key rate durations |
| `04_portfolio/01_markowitz.ipynb` | Mean-variance optimisation with constraints |

### Tier 2 — Breadth (frequent at AM, structurer, risk roles)

| Notebook | Topic |
|---|---|
| `01_options/06_heston.ipynb` | Stochastic vol, calibration to surface |
| `02_risk/03_credit_merton.ipynb` | Structural credit model — PD from equity vol |
| `03_fixed_income/03_term_structure.ipynb` | Vasicek, CIR, Hull-White |
| `03_fixed_income/04_swaps_swaptions.ipynb` | IRS pricing, Black-76 swaptions |
| `04_portfolio/02_black_litterman.ipynb` | Bayesian portfolio construction |
| `04_portfolio/04_factor_models.ipynb` | Fama-French + factor exposures |
| `04_portfolio/05_performance_attribution.ipynb` | Brinson-Fachler attribution |
| `05_volatility/01_garch.ipynb` | GARCH(1,1), EGARCH, GJR-GARCH |
| `05_volatility/02_realized_vol.ipynb` | High-frequency vol estimation |
| `06_stoch_calc/01_brownian_motion.ipynb` | BM, GBM, OU — simulation + properties |
| `06_stoch_calc/02_ito_and_gbm.ipynb` | Itô's lemma applied to common models |

### Tier 3 — Specialist (xVA desks, exotic structuring)

| Notebook | Topic |
|---|---|
| `02_risk/04_cva_intro.ipynb` | CVA / DVA / FVA — exposure simulation |
| `04_portfolio/03_risk_parity.ipynb` | Equal risk contribution + HRP |
| `05_volatility/03_local_vol_dupire.ipynb` | Dupire's formula, local vol surface |
| `06_stoch_calc/03_lsmc_american.ipynb` | Longstaff-Schwartz for early exercise |

## Environment

Adds to the existing `requirements.txt`:

```
yfinance              # equity / option chain data
pandas-datareader     # FRED rates, Fama-French factors
fredapi               # direct FRED API (needs free API key)
arch                  # GARCH family
QuantLib-Python       # validation oracle ONLY
```

Set up a FRED key (free) at `https://fred.stlouisfed.org/docs/api/api_key.html`
and export `FRED_API_KEY=...` for the rate-pull cells. Notebooks fall back to
recent hardcoded values if the API is unavailable.

## How to use for interview prep

1. Read **Why this matters** + **Concept** + **Math** for vocabulary.
2. Implement from scratch — *don't* read Section 4 first; try it yourself.
3. Run the validation cell to check your work matches the closed-form / QuantLib.
4. Do all exercises with solutions hidden. Reveal one at a time.
5. Read **Interview Q&A** out loud, in your own words. If you can't, you don't know it.
6. Drop into `quant_finance/exercises/` for the cross-topic problems.

## Suggested order

Tier 1 in numerical order is the right path. Don't jump ahead — Greeks build on
BS, MC builds on GBM, implied vol builds on BS-inverse. Tier 2 can be picked
based on role focus (AM → portfolio + factor; IB → swaps + credit + Heston).
