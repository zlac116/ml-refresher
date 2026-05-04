# Project 2 — FX Portfolio VaR (2-Day Fluency Capstone)

## Project goal

You're the new joiner on the **FX market-risk middle office** at a tier-1 bank. You inherit a ~200-position FX desk book (spot, vanilla options, barriers, digitals, forwards across 8-10 pairs) and have **2 days (~16h focused)** to compute, validate, backtest and stress-test the desk's daily VaR. You'll deliver four VaR cuts side-by-side (parametric, parametric+Cornish-Fisher, historical sim, MC with calibrated t-copula), reconcile against a synthetic FO VaR, and present stress losses against three named episodes plus one hypothetical. Plumbing is pre-built; you do the calibration, root-finding, MLE, and explain.

This is the **FX desk book** — distinct from Project 1's rates portfolio. Project 3 (XVA) combines both at counterparty level.

## Why this matters

Daily VaR is the single number a CRO sees at 8am. Backtesting determines whether the regulator (PRA / Fed / ECB) keeps your IMA approval — losing it can 2-3x your capital charge. The 2-day compression is honest: this is a **fluency capstone**, not a production system. You get the interpretation reps without months of plumbing — but the calibration / inversion / MLE work is preserved so you make actual quant decisions, not just call black boxes.

You'll exercise: risk-factor decomposition for FX derivatives, three (four with CF) VaR methodologies and where each breaks (`02_risk/01_var_methods.ipynb`), Kupiec POF + Christoffersen independence backtests, Cornish-Fisher tail correction, ES under FRTB (`02_risk/02_expected_shortfall.ipynb`), GARCH-driven stress data (`05_volatility/01_garch.ipynb`), t-copula MLE for joint risk-factor moves, PCA on risk factors (`04_portfolio/04_factor_models.ipynb`), Garman-Kohlhagen (`01_options/02_bs_family_and_asset_classes.ipynb` Part 4), and **IV inversion via Brent root-finding** on raw option marks.

## The realistic IB context (compressed)

Market risk is **independent** of the trading desk by regulatory mandate (post-Barings, post-LTCM). FO has its own intraday VaR; you produce the overnight reportable number. Daily 6-8am batch: data ingest → risk-factor returns → all VaR engines → reconciliation against FO → sign-off → CRO meeting. Quarterly: backtest report to the regulator under **traffic light** (Basel III IMA, 250 days @ 99%, expected ≈ 2.5):

- **Green** (≤4 exceedances): no surcharge
- **Amber** (5-9): multiplier rises 3.0 → 3.4-4.0, scrutiny
- **Red** (≥10): IMA approval may be revoked

**FO/MO gap drivers** (typical 10-30%): smile-vol vs ATM-only; 250d vs 1y/2y EWMA lookback; full-reval vs delta-gamma-vanna for exotics; 5pm London vs 11pm post-Tokyo cut.

## What's pre-built vs what you build

The scaffolding below is **provided** — import and use. The user-built column preserves the genuine quant work: calibration, root-finding, MLE, P&L explain, model-validation thinking.

| Pre-built (import & use) | You build (the quant work) |
|---|---|
| `data/portfolio.parquet` (~200 positions) | **IV inversion via Brent**: market option mids → implied vols (root-finding with discounted-intrinsic lower bound + Brenner-Subrahmanyam initial guess) |
| `data/fx_history.parquet` (2y, GARCH+crises) | **t-copula df MLE**: calibrate degrees of freedom on rank-transformed risk-factor returns; MC sampler consumes your df |
| `data/vol_surfaces.parquet` (SABR-fit ATM/RR/BF) | **Cornish-Fisher tail correction**: skew + excess kurtosis on residuals, apply CF expansion, measure how much of the parametric-vs-HS gap closes |
| `data/option_marks.parquet` (raw bid/ask quotes) | **4-method VaR comparison narrative** — when each fails, why |
| `data/stress_scenarios.json` (SNB / COVID / gilt) | **Backtest interpretation** (Kupiec/Christoffersen → traffic light → bank operational meaning) |
| `src/pricers/` (GK vanilla, barrier MC, digital, forward) | **Stress test interpretation** (named scenarios + 1 hypothetical, P&L attribution per pair, sane-magnitude vs published industry numbers) |
| `src/risk_factors.py` (positions → spot/ATMvol/RR/BF sensitivities) | **FO/MO reconciliation waterfall** (gap → vol-surface + lookback + correlation + residual; bump-revert each FO methodology choice) |
| `src/var/` (parametric harness, HS, MC sampler — you supply df) | **PCA on risk-factor returns** (top-3 > 80% variance; name them; portfolio loadings) |
| `src/backtest.py` (Kupiec POF, Christoffersen, traffic light) | **1-page `ANALYSIS.md`** (primary method, regulator pushback, production gaps) |
| `src/stress.py`, `src/fo_var.py`, `src/reports.py` | |

Pre-built code is black-box on Day 1. On Day 2, open `risk_factors.py` and `fo_var.py` so the reconciliation isn't mystical.

## Synthetic data + scaffolding plan

**Scope**: 2 years daily, 8-10 pairs, ~200 positions.

1. **Spot history**: GBM with GARCH(1,1) per pair. G10 majors (EUR/USD, USD/JPY, GBP/USD, USD/CHF) ~8-12% ann, persistence ~0.97; EM (USD/MXN, USD/ZAR) ~15-25%, persistence ~0.95; pegged (USD/CNH) low daily vol with fat-tailed jumps.
2. **3 named crises** spliced in: 2015-01-15 SNB unpeg (EUR/CHF -30% intraday); 2020-03-09 COVID USD spike; 2022-09-26 UK gilt (GBP/USD trough).
3. **Vol surfaces** per pair (ATM term structure + 25Δ RR + 25Δ BF) SABR-fit. Raw bid/ask **option marks** also provided so you invert IVs yourself.
4. **Portfolio**: ~70% G10 majors, 25% liquid EM, 5% pegs/niche; ~40% spot, 35% vanillas, 15% forwards, 10% exotics.

All seeded (`np.random.default_rng(42)`).

## Project tree

```
quant_finance/capstones/02_fx_var/
├── README.md
├── data/                           (PRE-BUILT)
│   ├── fx_history.parquet
│   ├── vol_surfaces.parquet        SABR-fit, for valuation
│   ├── option_marks.parquet        raw bid/ask, for YOUR IV inversion
│   ├── portfolio.parquet
│   └── stress_scenarios.json
├── src/                            (PRE-BUILT)
│   ├── pricers/{gk_vanilla,barrier_mc,digital,forward}.py
│   ├── risk_factors.py
│   ├── var/{parametric,historical,mc}.py    (mc.py samples t-copula given your df)
│   ├── backtest.py                 (Kupiec POF, Christoffersen, traffic light)
│   ├── stress.py
│   ├── fo_var.py                   (synthetic FO with deliberately different methodology)
│   └── reports.py
├── notebooks/                      (YOU BUILD)
│   ├── 01_iv_inversion_parametric_cf.ipynb         (Day 1)
│   └── 02_hs_mc_stress_reconciliation.ipynb        (Day 2)
├── reports/                        (YOU GENERATE)
│   ├── implied_vols.csv
│   ├── daily_var.html
│   ├── backtest_traffic_light.html
│   ├── stress_report.html
│   └── fo_mo_reconciliation.html
└── ANALYSIS.md                     (YOU WRITE — 1 page)
```

## 2-day milestones

### Day 1 (8h) — IV inversion + parametric VaR + Cornish-Fisher + first backtest

- **H0-1**: load all parquets. Sanity-check position counts per pair, plot SNB and gilt episodes in EUR/CHF and GBP/USD, confirm vol clustering visually.
- **H1-3**: **IV inversion (root-finding)**. For each row in `option_marks.parquet`, invert via `scipy.optimize.brentq` against `gk_vanilla` to recover IV. Implement the discounted-intrinsic lower-bound check (skip stale quotes below it) and Brenner-Subrahmanyam initial guess. Output `reports/implied_vols.csv` with market mid, your IV, library SABR-fit IV, residual. Match within 5bp on >90% of ATM/25Δ; flag outliers with a reason.
- **H3-4**: `risk_factors.decompose(portfolio)` → exposure matrix (positions × {spot, ATMvol, RR, BF} per pair). Which pairs dominate gross/net? Outsized vega vs delta?
- **H4-5**: `var.parametric(exposures, cov)` for 1d 99% and 10d 99%. Top-10 contributors. Decomp spot vs vol risk.
- **H5-6**: **Cornish-Fisher**. Compute skew $S$ and excess kurtosis $K$ on residuals (1y or 2y window for stability — 250d is too noisy). Apply CF expansion to the parametric quantile. Compare CF-VaR vs vanilla parametric on the last 60 days.
- **H6-8**: 250-day rolling backtest of parametric and parametric+CF. Kupiec POF + Christoffersen independence; traffic-light read. Plot exceedance clustering — independence violation is as bad as too-many.

**Deliverables**: `01_iv_inversion_parametric_cf.ipynb` + `reports/implied_vols.csv` + `reports/daily_var.html` (parametric + CF) + `reports/backtest_traffic_light.html` (parametric + CF only, will extend Day 2).

### Day 2 (8h) — HS + t-copula MLE + MC + PCA + stress + FO/MO reconciliation + ANALYSIS

- **H0-1.5**: `var.historical(exposures, returns_window=250)` with full revaluation. Compare HS vs parametric/CF on last 60 days. Where do they diverge most? (High-vol regimes — HS catches what parametric assumes away; CF closes some marginal non-normality but not tail-dependence.)
- **H1.5-3**: **t-copula df MLE (calibration)**. Rank-transform risk-factor returns, fit copula df via MLE (`scipy.optimize.minimize_scalar`, df ∈ (2, 30)). Plot the LL surface, pick df, document choice. Feed into `var.mc(exposures, copula='t', df=YOUR_DF, n_paths=20_000)` for full-reval MC VaR.
- **H3-4**: backtest HS + MC via `backtest.kupiec` + `backtest.christoffersen`; update traffic-light HTML to all four methods. Cross-check at 95% (~12-13 expected) for clearer signal.
- **H4-4.5**: PCA on risk-factor returns. Confirm top-3 capture >80% variance. Project portfolio onto PCs — name them ("USD strength", "EUR/CHF skew", "EM carry").
- **H4.5-6**: `stress.apply(portfolio, scenario)` for SNB / COVID / gilt + 1 hypothetical you design (EUR/USD parity break, USD/JPY 200 print). P&L per pair, worst single position, sanity-check vs published industry (SNB cost industry ~$400M-$1B; your loss should be a sane fraction for your book size).
- **H6-7.5**: FO/MO reconciliation. Run `fo_var.compute(...)` (smile-vol + 250d lookback + delta-gamma-vanna for exotics). Build the **waterfall**: total gap = vol-surface + lookback + correlation + residual. Bump-revert each FO methodology choice in turn (swap MO's flat-vol input for FO's smile, recompute, see how much of the gap shrinks). Target: explain >90%. The residual is where the interesting interview answer lives.
- **H7.5-8**: write `ANALYSIS.md` (≤1 page). Three questions: (a) **which method as primary?** — defensible, not a hedge; (b) **where does the regulator push back?** — name two specific places; (c) **what's missing for production?** — name three (intraday, NMRF treatment for thin-EM pairs, multi-day scaling beyond √t).

**Deliverables**: `02_hs_mc_stress_reconciliation.ipynb` + updated `backtest_traffic_light.html` (all 4 methods) + `reports/stress_report.html` + `reports/fo_mo_reconciliation.html` + `ANALYSIS.md`.

## Final deliverables

1. `reports/implied_vols.csv` — Brent-inverted IVs vs market mids and library SABR values
2. `reports/daily_var.html` — four methods side-by-side, top-10 contributors, spot-vs-vol attribution
3. `reports/backtest_traffic_light.html` — exceedances, Kupiec p, Christoffersen p, traffic-light per method
4. `reports/stress_report.html` — 3 named + 1 hypothetical, P&L per pair, worst single position
5. `reports/fo_mo_reconciliation.html` — waterfall, named contributions, attribution methodology documented
6. PCA output (in notebook 02) — first 3 PCs, loadings, portfolio exposure
7. `ANALYSIS.md` (≤1 page) — primary-method recommendation, regulator pushback, production gaps

## Hints

- **IV inversion**: lower bound is **discounted intrinsic** ($S e^{-q\tau} - K e^{-r\tau}$ for calls), NOT raw intrinsic. Brent fails on stale quotes below it — pre-check and skip. Cap iterations near flat-vega (deep OTM far-dated).
- **Why four methods?** Each fails differently. Parametric assumes Gaussian (catastrophic for FX 6σ). CF fixes marginal non-normality but not tail-dependence. HS underestimates tails outside the lookback window. MC+t-copula handles tail dependence but is heaviest. Banks triangulate.
- **1d vs 10d**: Basel IMA wants **10d 99%**. Internal management uses 1d. √t scaling holds only under IID — overnight gaps and weekends break it.
- **When parametric fails**: bimodal regimes (peg breaks, CB intervention). EUR/CHF Jan 2015: Gaussian VaR ~$5M, realised ~$400M industry-wide.
- **t-copula df**: typically 4-6 for FX. **The MLE landscape is flat between df = 4 and 8** — pick the value that minimises tail-error in backtest, not just the LL maximum.
- **Cornish-Fisher**: sample skew/kurtosis are noisy on 250 days; use 1y or 2y window. CF doesn't fix tail-dependence — it'll close some of the gap to HS, not all.
- **Exceedance count of 0 is also a flag** — over-conservative model eats capital you don't need.
- **Don't tune to backtest pass**. Regulators look at methodology stability, not point-in-time pass rates.
- **Reconciliation residual**: if >30%, you're missing a risk factor (often skew or term-structure of vol).

## Cross-references

| Concept | Where |
|---|---|
| Parametric / HS / MC VaR | `02_risk/01_var_methods.ipynb` |
| Cornish-Fisher tail correction | `02_risk/01_var_methods.ipynb` (`cornish_fisher_var`) |
| Kupiec POF | `02_risk/01_var_methods.ipynb` (`kupiec_pof` — note `max(lr, 0)` fix) |
| Expected Shortfall + Acerbi-Szekely | `02_risk/02_expected_shortfall.ipynb` |
| GARCH(1,1) | `05_volatility/01_garch.ipynb` |
| Garman-Kohlhagen FX | `01_options/02_bs_family_and_asset_classes.ipynb` Part 4 |
| Brent IV inversion + Brenner-Subrahmanyam seed | `01_options/01_black_scholes.ipynb` |
| SABR smile | `01_options/02_bs_family_and_asset_classes.ipynb` Part 2 + `sabr_cheatsheet.md` |
| BS / Bachelier reference | `01_options/bs_cheatsheet.md`, `01_options/bachelier_cheatsheet.md` |
| GBM / log-return scaling | `06_stoch_calc/02_ito_and_gbm.ipynb` |
| PCA on factor returns | `04_portfolio/04_factor_models.ipynb` |

## Success criteria

The project is "done" when **all** hold:

1. **IV inversion** matches library SABR-fit IVs within 5bp on >90% of ATM/25Δ quotes; outliers flagged with reason
2. **t-copula df MLE** produces a defensible df (typically 4-7) with LL surface plotted; choice documented
3. **Cornish-Fisher** reduces parametric-vs-HS gap on tail days by ≥30% (won't close fully — that's the point)
4. **Four VaR methods** run end-to-end on the ~200-position book in <30s (parametric, CF, HS) and <3min (MC, 20k paths)
5. **Backtest passes Kupiec at 95%** over 250 days for **at least 2 of 4** methods
6. **Stress test** produces sane per-pair losses for SNB/COVID/gilt — directionally correct, magnitudes within an order of magnitude of published industry estimates scaled to book size
7. **FO/MO reconciliation waterfall** explains **>90% of the gap** with named contributors, attribution methodology documented
8. **PCA** shows top-3 factors > 80% variance; portfolio loadings interpretable
9. **`ANALYSIS.md`** exists, ≤1 page, answers all three writeup questions concretely

## What this prepares you for

- **FRTB / Basel III interview questions**: IMA vs SA, model-eligibility, NMRF, liquidity horizons, P&L attribution test
- **Regulatory audit conversations**: how do you defend the model? Backtest exception protocols?
- **Market-risk MO / risk-control roles**: VaR sign-off, capital optimisation, model validation
- **Quant dev at a hedge fund / fintech**: many shops outsource to MSCI / Numerix / Bloomberg PORT but need someone who knows what those numbers mean

The single biggest interview signal: you can speak fluently about **why VaR estimates differ across methods, what those differences mean operationally, and what you would do about them** — the actual day-job of market-risk MO. Two days won't make you that person; it'll make you sound like you've thought about being that person, which is enough for a first-round screen.
