# Project 1 — Rates Portfolio Calibration & FV P&L Reconciliation (2-Day Sprint)

## Project goal

You're the new joiner on the **rates desk product control / FO quant** at a tier-1 bank. You inherit a ~200-trade rates book — vanilla swaps, European and Bermudan swaptions, callable bonds, govvies, plus a couple of exotics. In 16 hours: bootstrap curves, calibrate SABR + LMM, price the book (including Bermudans via LMM+LSMC with a Hull-White cross-check), compute greeks, attribute 4 days of P&L, and reconcile against B&R. Plumbing is pre-built — your time goes into calibration choices, attribution convention, and break investigation.

## Why this matters

P&L attribution is the single most-scrutinised report a rates desk produces. B&R reconciliation certifies the desk's daily P&L for the firm's books. SABR and LMM aren't textbook curiosities — they're the calibration layer that makes thousands of swaption marks consistent across the cube and that lets you put a defensible price on a Bermudan.

Cross-references to your module work:

- Multi-curve discounting → `03_fixed_income/03_curve_building.ipynb`
- Bonds, KRDs → `03_fixed_income/01_bond_pricing.ipynb`, `02_duration_convexity_krd.ipynb`
- Swap PV = -A·(K - K_par), DV01 → `03_fixed_income/04_swaps_swaptions.ipynb`
- Black-76 / Bachelier swaptions → `01_options/02_bs_family_and_asset_classes.ipynb` + `bs_cheatsheet.md`, `bachelier_cheatsheet.md`
- SABR per-cell → `01_options/02_bs_family_and_asset_classes.ipynb` + `sabr_cheatsheet.md`
- LMM with predictor-corrector → `03_fixed_income/05_libor_market_model.ipynb` + `lmm_cheatsheet.md`
- LSMC backward induction → `06_stoch_calc/03_lsmc_american.ipynb`
- Greeks → `01_options/03_greeks.ipynb`
- Attribution methodology → `04_portfolio/05_performance_attribution.ipynb`

## The realistic IB context (compressed)

Two functions to keep clear:

- **FO quant / desk strat**: pricers, calibration, hedging greeks. Reports to the desk head.
- **MO product control**: independent reprice, P&L certification, B&R reconciliation, break escalation. Reports to CFO. Independent by design.

Same tooling, different sign-off chain. This sprint covers both.

**7am-9am workflow** (what your code reproduces, compressed):

| Time | Activity |
|---|---|
| 6:30 | Market data ingest, trade book frozen, curves bootstrapped |
| 7:00 | SABR per-cell + cap-vol bootstrap; fit residuals checked |
| 7:15 | Reprice: vanilla closed-form; Bermudans via LMM + LSMC |
| 7:30 | Greeks: KRDs (1y/2y/5y/10y/30y), vega per cell, theta |
| 7:45 | P&L attribution: delta×Δcurve + vega×Δvol + theta×Δt + cross + unexplained |
| 8:00 | B&R reconciliation; top breaks investigated |
| 8:30 | Sign-off; unexplained < 5% of |total P&L| on benign days |

**Tolerances** (per million notional): vanilla swap < $100; vanilla swaption < $500; Bermudan/callable < $2,000 (MC noise dominant); govvies < $50; exotics < $5,000.

Common break causes: stale B&R mark, smile residual at off-grid strike, day-count mismatch (silent killer), Bermudan MC noise, model choice (LMM vs HW).

## What's pre-built vs what you build

The plumbing is provided so the 16 hours go into calibration, attribution and reconciliation.

**Pre-built (`pip install -e .` and import):**

- `data/trades.parquet` — ~200 trades: 70 vanilla IRS (USD/EUR mix, 2y-30y), 50 European swaptions, 20 Bermudans, 25 callable bonds, 25 vanilla govvies/corps, 10 misc, 2 exotics
- `data/market_data/` — 5 trading days: D1 normal, D2 bull-steepener (front -10bp, long -2bp), D3 vol blowout (+3 vol pts), D4 parallel +25bp crisis, D5 partial reversion. Day moves engineered so attribution buckets carry signal
- `data/br_marks/` — synthetic B&R per-trade marks with ~8-12 deliberate breaks per day
- `src/curves.py` — OIS + projection bootstrap, log-DF interpolation, KRD bucket bumps
- `src/pricers/` — Black-76, Bachelier, IRS via -A·(K - K_par), vanilla bond, callable bond, LMM simulator with predictor-corrector, LSMC backward induction, Hull-White single-factor Bermudan helper for cross-check
- `src/vol/sabr.py` — SABR optimisation harness (you supply parameter choices; β fix, α-from-ATM, (ρ,ν) least-squares)
- `src/vol/lmm_calibration.py` — caplet-vol bootstrap harness (you supply cap quotes)
- `src/greeks.py` — finite-difference KRD, vega, theta engine
- `src/reports.py` — HTML/CSV table + waterfall chart helpers

**You build (the actual quant work):**

- **SABR per-cell calibration**: fix β = 0.5; solve α from the ATM cubic; least-squares (ρ, ν) on the smile. Build the residuals heatmap
- **LMM caplet-vol bootstrap**: root-find cumulative variance forward-by-forward against the cap-strip
- **LMM caplet repricing test**: simulate, compare to Black-76, verify ±2σ MC for every caplet — this is your model-validation check
- **Bermudan pricing**: LMM + LSMC with your basis choice (polynomial in S(T_e), ITM only). Cross-check 3 against Hull-White; bracket within ~10% is the win
- **Greeks dispatch** for all 5 days
- **P&L attribution waterfall**: pick the bucketing convention (delta first / vega next / theta / cross / unexplained), defend it, run it across the 4-day window
- **B&R reconciliation**: per-trade FV vs B&R, threshold by trade type, top-breaks table with cause hypothesis
- **Written analysis** (~1 page)

## Synthetic data + scaffolding plan

~200 trades, 5 trading days, deliberate breaks. Notional concentration mirrors a real desk (~80% vanilla swaps, ~15% vanilla swaptions, ~5% tail). Reproducible — `np.random.default_rng(42)`. Day moves chosen so each attribution bucket carries non-trivial signal across the 4-day window.

## Project tree structure

```
quant_finance/capstones/01_rates_portfolio/
├── README.md
├── data/
│   ├── trades.parquet
│   ├── market_data/{curves,swaption_cube,cap_vols,credit_spreads}_D{1..5}.parquet
│   └── br_marks/br_D{1..5}.parquet
├── src/                            (PRE-BUILT — pip install -e .)
│   ├── trades.py                   trade dataclasses
│   ├── curves.py                   bootstrap + KRD bumps
│   ├── pricers/                    irs, swaption, bond, callable, lmm, lsmc, hw
│   ├── vol/                        sabr.py, lmm_calibration.py
│   ├── greeks.py
│   └── reports.py
├── notebooks/                      (YOU BUILD)
│   ├── day1_calibrate_price_greeks.ipynb
│   └── day2_attribution_recon.ipynb
├── reports/
│   ├── pv_D1.csv
│   ├── greeks_D{1..5}.csv
│   ├── calibration_quality.html
│   ├── pnl_attribution.html
│   └── br_reconciliation.html
└── ANALYSIS.md
```

## 2-day milestones

**Day 1 — Calibration + Pricing + Bermudans + Greeks (8 hours)**

| Hour | Task |
|---|---|
| 0:00-0:30 | Load `trades.parquet` + D1 market data. Skim dataclasses |
| 0:30-1:00 | Bootstrap OIS + projection curves (D1). Round-trip par swap rates from the curve — sanity check, not optional |
| 1:00-2:30 | **SABR per (expiry × tail) cell** on D1 swaption cube. β = 0.5 fixed; α from the ATM cubic; (ρ, ν) least-squares to smile. Residuals heatmap |
| 2:30-3:30 | **LMM caplet-vol bootstrap** from cap strip. Forward-by-forward cumulative-variance root-find |
| 3:30-4:30 | **LMM caplet repricing test**: simulate every caplet, compare to Black-76, t-stat heatmap, pass/fail at ±2σ MC. If a caplet fails, fix calibration before moving on |
| 4:30-5:30 | Dispatch all ~200 trades: vanilla closed-form, callable = bond − American call on bond. Write `pv_D1.csv` |
| 5:30-7:00 | **Bermudans via LMM + LSMC**. Basis = polynomial in S(T_e) at the exercise date, degree 2-3, ITM paths only. Cross-check 3 against `src.pricers.hw_bermudan_price`; document the bracket |
| 7:00-8:00 | Greeks for all 5 days via `src.greeks` (KRDs per bucket, vega per cell, theta-by-Δt). Write `greeks_D{1..5}.csv`, `calibration_quality.html` |

**Day 1 deliverables**: `pv_D1.csv`, `greeks_D{1..5}.csv`, `calibration_quality.html`.

**Day 2 — Attribution + Reconciliation + Writeup (8 hours)**

| Hour | Task |
|---|---|
| 0:00-0:30 | Re-price the book on D2-D5 using the Day-1 calibration pipeline. Cache the PVs per day |
| 0:30-3:00 | **P&L attribution waterfall** across (D2-D1, D3-D2, D4-D3, D5-D4): delta × Δcurve (sum over KRD buckets), vega × Δvol (swaption cells + caplet vega for callables), theta × Δt, cross-term (full reprice − named buckets — captures curve-reshape, smile-shift, higher-order), unexplained residual. Document the order convention. Target: unexplained < 5% on D2 (curve-shape) and D5 (reversion); document why crisis day breaches |
| 3:00-4:00 | Generate `pnl_attribution.html` via `src.reports` — per-bucket line items + top-10 contributors per bucket |
| 4:00-6:00 | **B&R reconciliation** per day. Per-trade FV vs B&R diff, threshold check by type. Top-20 breaks per day with **cause hypothesis** (stale mark / off-grid smile / day-count / Bermudan MC noise / model). The deliberate breaks should all flag — if some don't, your tolerance is too loose |
| 6:00-6:30 | Generate `br_reconciliation.html` |
| 6:30-8:00 | Write `ANALYSIS.md` (~1 page): (a) where the model breaks (Bermudans with thin exercise schedule, deep-OTM swaption wings, the 2 exotics); (b) production additions needed (SABR-LMM for smile-aware Bermudans, shifted-LMM for negative rates, smile-aware delta, intraday repricing); (c) the SABR vs LMM choice and when each is the right tool |

**Day 2 deliverables**: `pnl_attribution.html`, `br_reconciliation.html`, `ANALYSIS.md`.

## Final deliverables

1. `reports/pv_D1.csv` — per-trade PV on D1
2. `reports/greeks_D{1..5}.csv` — per-trade KRDs, vega, theta across the 5-day window
3. `reports/calibration_quality.html` — SABR residual heatmap + LMM caplet repricing test (Black-76 vs MC, t-stats, pass/fail)
4. `reports/pnl_attribution.html` — 4-day waterfall + per-bucket line items + top-10 contributors per bucket
5. `reports/br_reconciliation.html` — per-trade FV vs B&R, breaks vs tolerance, top-20 breaks per day with cause hypothesis
6. `ANALYSIS.md` — ~1 page covering the three writeup questions

## Hints

- **Day-count is a silent killer**. USD swap fixed leg act/360, EUR swap fixed leg 30/360. USD govvie act/act, USD corp 30/360. The dataclasses encode this — don't override unless you mean it
- **Multi-curve**: discount on OIS for collateralised trades, project floats on the index curve. Single-curve is dead post-2008
- **SABR α-from-ATM**: don't joint-fit (α, ρ, ν). Fix β = 0.5, solve α from the ATM cubic, then fit (ρ, ν) to the smile. Joint fit gives parameter degeneracy — see `sabr_cheatsheet.md`
- **LSMC basis**: polynomial in S(T_e) only, degree 2-3, ITM paths only. Don't use the full forward strip — collinearity wrecks the regression
- **Caplet timing**: caplet on L(T, T+δ) pays at T+δ, discounts at D(0, T+δ). The pricer gets it right; verify you don't bypass it
- **Black-76 vs Bachelier**: if any forward < 50bp and ATM vol > 50% of forward, lognormal breaks. Switch to Bachelier or shifted Black-76. `bachelier_cheatsheet.md` has the Hagan-Kennedy conversion
- **Attribution is order-dependent** (the Σ-attribution problem). Pick a convention (typical: delta first, then vega, then theta, residual to cross). Document it in `ANALYSIS.md` and don't change it mid-window
- **LMM vs HW Bermudan**: they won't agree exactly. LMM captures forward-rate decorrelation; HW is single-factor and faster. A 5-10% gap is normal; > 15% means your LSMC basis is under-specified or your LMM calibration is off

## Cross-references

| Concept | Where it's covered |
|---|---|
| Bond PV, duration, convexity | `03_fixed_income/01_bond_pricing.ipynb`, `02_duration_convexity_krd.ipynb` |
| KRD construction | `03_fixed_income/02_duration_convexity_krd.ipynb` |
| OIS multi-curve bootstrap | `03_fixed_income/03_curve_building.ipynb` |
| Par swap rate, swap PV, DV01 | `03_fixed_income/04_swaps_swaptions.ipynb` |
| Black-76 swaption | `01_options/02_bs_family_and_asset_classes.ipynb` + `bs_cheatsheet.md` |
| Bachelier (low-rate regime) | `01_options/02_bs_family_and_asset_classes.ipynb` + `bachelier_cheatsheet.md` |
| SABR per-cell | `01_options/02_bs_family_and_asset_classes.ipynb` + `sabr_cheatsheet.md` |
| LMM (BGM) for Bermudans, predictor-corrector | `03_fixed_income/05_libor_market_model.ipynb` + `lmm_cheatsheet.md` |
| LSMC backward induction | `06_stoch_calc/03_lsmc_american.ipynb` |
| Greeks | `01_options/03_greeks.ipynb` |
| Attribution methodology | `04_portfolio/05_performance_attribution.ipynb` |

## Success criteria

The sprint is done when **all** of the following hold:

1. **LMM caplet repricing test** passes: simulator vs Black-76 within ±2σ MC for every caplet in the strip
2. **Vanilla B&R breaks** all under tolerance ($100/MM swaps, $50/MM bonds), except the deliberately injected breaks (which your reconciliation correctly flags)
3. **At least 3 Bermudans** priced via LMM and cross-checked against Hull-White; the two methods bracket within ~10%
4. **P&L unexplained < 5% of |total P&L|** on at least 2 of the 4 days (typically the curve-shape and reversion days); explanation documented for the days where it isn't (typically crisis where second-order cross-terms matter)
5. **SABR fit residuals** all under 50bp of vol on cube grid points; off-grid breach acknowledged in the dashboard
6. **`ANALYSIS.md`** exists and addresses the three writeup questions

## What this prepares you for

- **FO quant interview at IB rates desk**: SABR vs LMM choice, cap-vol vs swaption-vol calibration, Bermudan exercise boundary, why predictor-corrector
- **MO product control interview**: P&L attribution methodology, B&R reconciliation, model validation thinking
- **Risk-management roles**: KRDs, multi-curve discounting, vol-surface arbitrage detection
- **Quant developer at vendor / fintech**: many shops outsource pricing to Numerix / MX.3, but need someone who can challenge the numbers

The single biggest interview signal: you can speak about **why a Bermudan price disagrees by 5% between LMM and Hull-White, and which one you'd trust for which product**. The 2-day sprint is engineered so you've done it once with your own hands — that's what the interview is testing for.

Two days is a **fluency capstone**, not a production system. A real bank build takes quarters and has institutional context this sprint can't reproduce. What it does deliver: you've run the daily batch end-to-end on realistic infrastructure, made the calibration choices yourself, and have something concrete and defensible to walk through in interview.
