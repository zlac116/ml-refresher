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

### Tier 1 — Core (always asked at IB quant screens) ✅ COMPLETE

| Notebook | Topic | Status |
|---|---|---|
| `01_options/01_black_scholes.ipynb` | BS from first principles, PDE + risk-neutral derivations, IV, smile, Black-76 | ✅ expert-reviewed |
| `01_options/02_bs_family_and_asset_classes.ipynb` | Bachelier, SABR, FX (Garman-Kohlhagen), Fixed Income (caplets, swaptions) | ✅ expert-reviewed |
| `01_options/03_greeks.ipynb` | δ, γ, ν, θ, ρ + vanna/volga/charm; analytic + FD; gamma-theta P&L | ✅ |
| `01_options/04_binomial_trees.ipynb` | CRR + trinomial; American puts; convergence to BS | ✅ |
| `01_options/05_monte_carlo_pricing.ipynb` | MC + variance reduction (antithetic, control variates); Asian; barriers; pathwise Greeks | ✅ |
| `01_options/06_implied_vol_surface.ipynb` | SVI parameterisation; arbitrage constraints; Breeden-Litzenberger density | ✅ |
| `02_risk/01_var_methods.ipynb` | Parametric, historical, MC VaR; Kupiec backtest; non-subadditivity | ✅ |
| `02_risk/02_expected_shortfall.ipynb` | ES coherence; FRTB 97.5%; Acerbi-Szekely backtest | ✅ |
| `03_fixed_income/01_bond_pricing.ipynb` | YTM via Brent; dirty/clean; day-counts | ✅ |
| `03_fixed_income/02_duration_convexity_krd.ipynb` | Macaulay/modified duration; convexity; KRDs; DV01 hedging | ✅ |
| `04_portfolio/01_markowitz.ipynb` | Efficient frontier; tangency; MV instability; 1/N benchmark | ✅ |

### T1 supplement — added based on user feedback

| Notebook | Topic | Status |
|---|---|---|
| `03_fixed_income/03_curve_building.ipynb` | Bootstrap deposit + swap curve; forward rates; multi-curve world | ✅ |

### Tier 2 — Breadth (frequent at AM, structurer, risk roles)

**Interview-critical T2 (DONE):**

| Notebook | Topic | Status |
|---|---|---|
| `01_options/07_heston.ipynb` | Stochastic vol; Heston 1993 char-fn pricing; calibration; vs SABR / local vol | ✅ |
| `02_risk/03_credit_merton.ipynb` | Structural credit; equity = call on assets; DD; equity-implied asset vol | ✅ |
| `04_portfolio/02_black_litterman.ipynb` | Bayesian portfolio; implied returns; views; stability vs MV | ✅ |
| `04_portfolio/04_factor_models.ipynb` | CAPM, FF3, Carhart 4F on real Ken-French + AAPL data; attribution; IR | ✅ |

**T2 remaining:**

| Notebook | Topic |
|---|---|
| `03_fixed_income/04_swaps_swaptions.ipynb` | IRS pricing in detail, Black-76 swaptions, normal-vol calibration |
| `04_portfolio/05_performance_attribution.ipynb` | Brinson-Fachler attribution (allocation vs selection) |
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
