# Quant Finance Capstone Trilogy (2-Day Sprints)

Three IB-realistic capstone projects, each scoped as a **2-day focused sprint** (~16h each, ~6 days total). Production-grade scaffolding is provided so the time goes into the actual quant work — calibration, root-finding, P&L explain, model validation — not the plumbing.

## The three projects

| # | Project | Portfolio | The quant work YOU do | Time |
|---|---|---|---|---|
| **1** | [Rates portfolio calibration & FV P&L](01_rates_portfolio/README.md) | ~200 rates trades — **own desk portfolio** | SABR per-cell calibration, LMM caplet bootstrap, LMM Bermudan + LSMC + Hull-White cross-check, P&L attribution waterfall, B&R reconciliation | 2 days |
| **2** | [FX VaR + reconciliation + stress](02_fx_var/README.md) | ~200 FX positions — **separate FX desk portfolio** | Brent IV inversion, t-copula df MLE, Cornish-Fisher tail correction, 4-method VaR + Kupiec/Christoffersen backtest, FO/MO reconciliation waterfall, PCA | 2 days |
| **3** | [XVA on the combined book](03_xva/README.md) | **Both portfolios above + cross-currency swaps**, aggregated to ~30 counterparties | CDS hazard bootstrap (Brent), joint rates-FX correlation calibration, CVA/DVA/FVA/MVA/KVA decomposition, FD sensitivities, CDS hedge LP, optional WWR copula calibration | 2 days |

**Critical**: Projects 1 and 2 are independent portfolios on different desks (rates vs FX). Project 3 takes both as input and aggregates by **counterparty** — netting/CSAs operate per legal entity, not per asset class.

## What's pre-built vs what you build

The compression vs the original 5-week-each scope is achieved by pre-building the plumbing and preserving the quant decisions as user work.

**Pre-built across the trilogy** (in each project's `data/` and `src/`):
- Sample portfolios (parquet on disk) — you load, you don't generate
- Synthetic market data with realistic vol clustering and named crisis episodes — you load, you don't generate
- Pricer libraries: Black-76, Bachelier, Garman-Kohlhagen, vanilla bond, callable bond, LMM simulator, LSMC backward induction, Hull-White Bermudan helper
- Risk-factor decomposition engines, FD Greek bumpers, KRD bucketers
- VaR engines (parametric, HS, MC samplers — **you supply the calibrated copula df**)
- Backtest utilities (Kupiec, Christoffersen, traffic light)
- XVA calculators (CVA/DVA/FVA/MVA/KVA — **you supply the bootstrapped hazards and calibrated correlations**)
- LP solver wrapper for the hedge construction (**you supply objective + constraints**)
- Reporting boilerplate (HTML/CSV generators)

**You build (the genuine quant work)**:
- **Calibration**: SABR per-cell, LMM caplet bootstrap, t-copula df MLE, joint rates-FX correlation, CDS hazard stripping
- **Root-finding**: Brent IV inversion (FX), Brent hazard bootstrap (XVA)
- **Validation**: LMM caplet repricing test, Hull-White cross-check on Bermudans, hazard bootstrap vs reference
- **P&L explain**: rates attribution waterfall, FX FO/MO reconciliation waterfall
- **Model risk**: Cornish-Fisher tail correction, WWR Gaussian-copula calibration (optional stretch)
- **Hedging**: CDS hedge LP construction (objective + constraint formulation)
- **Decision-making**: which VaR method as primary, which counterparties to hedge, which XVA dominates
- **Written analysis**: 1-page ANALYSIS.md per project

## Why this trilogy

A real IB rates desk doesn't price options in isolation, doesn't compute VaR without backtesting, and doesn't quote a trade without the XVA desk's CVA charge. These three projects together model what a desk produces in **a single overnight batch**:

- **Project 1** → today's mark-to-market, Greeks, P&L attribution, certified for the firm's books
- **Project 2** → tomorrow's market-risk capital number, regulator-defensible
- **Project 3** → CVA / FVA charge applied to every new trade quoted today, plus hedge book updates

You'll exercise essentially every technique covered in `quant_finance/`:

| Module | Used in |
|---|---|
| `01_options/` (BS family, Greeks, MC, Heston) | Project 1 (swaption pricing), Project 2 (FX option Greeks + IV inversion), Project 3 (option exposure) |
| `02_risk/` (VaR, ES, Merton, CVA) | Project 2 (VaR methods + backtests + CF), Project 3 (Merton, hazard bootstrap, CVA integral) |
| `03_fixed_income/` (curves, swaps, LMM) | Project 1 (entire curve + Bermudan stack), Project 3 (rates exposure) |
| `04_portfolio/` (PCA, attribution) | Project 1 (P&L attribution methodology), Project 2 (PCA on risk factors) |
| `05_volatility/` (GARCH, realised, Dupire) | Project 1 (vol surface construction), Project 2 (GARCH for stress data) |
| `06_stoch_calc/` (BM, Itô, LSMC) | Project 1 (LSMC for Bermudans), Project 3 (joint MC, exposure-measure LSMC) |

Plus all four cheatsheets: `bs_cheatsheet.md`, `bachelier_cheatsheet.md`, `sabr_cheatsheet.md`, `lmm_cheatsheet.md`.

## Recommended order

**Do them in order.** Each project introduces patterns the next one builds on:

1. **Project 1 first**. Establishes the trade representation, market-data layout, and curve/vol calibration patterns. By the end you can price every rates product.
2. **Project 2 second**. Independent FX portfolio. Reuses your data-loading pattern from Project 1 but introduces VaR machinery, copulas, stress-scenario engines. Doesn't depend on Project 1's portfolio.
3. **Project 3 last**. Takes the trade books from Projects 1 and 2 plus cross-currency swaps. The exposure simulation is the heaviest computational task; the netting/CSA logic is the most operationally complex.

**Total time estimate**: 2 days per project × 3 = ~6 days of focused work (or ~2 weeks of evenings at 3-4 hours per evening). The 2-day compression assumes you're not also building the scaffolding — that's pre-provided.

## What "done" looks like for the trilogy

By the time you've completed all three:

1. **6 notebooks** spanning the three projects (2 per project)
2. **3 portfolios** fully priced under realistic synthetic market data
3. **All-up reports**: P&L attribution, B&R reconciliation, daily VaR (4 methods), backtest, stress, XVA breakdown, hedge proposal — generated automatically from your pipelines
4. **3 written analyses** (~1 page each) capturing dominant findings, model approximations, production gaps
5. **Cross-references** back to module notebooks documented per technique used

This is **portfolio-quality work**: you can show this to an interviewer / hiring manager as "here's how I demonstrate I can run the daily output of an IB rates / market-risk / XVA desk." Every technique is grounded in a real workflow with named deliverables — and the calibration / root-finding / P&L-explain steps are visibly **your work**, not pre-built calls.

## What this prepares you for (in interview)

You'll be able to answer fluently:

- *"Walk me through what happens overnight on the rates desk batch."* → Project 1
- *"How do you reconcile FO and MO VaR? What's the typical break source?"* → Project 2
- *"How do you invert market option marks to implied vols, and what are the gotchas?"* → Project 2
- *"How do you bootstrap a hazard curve from CDS spreads?"* → Project 3
- *"Why is CVA for an uncollateralised swap with a sovereign higher than for the same trade with an IG corporate?"* → Project 3
- *"Walk me through what happens overnight in the XVA desk's batch."* → Project 3
- *"What's the difference between SABR per-cell and SABR-LMM, and when would you use each?"* → Project 1 stretch
- *"How do you handle wrong-way risk in the CVA computation?"* → Project 3 stretch
- *"Why does FVA + DVA → 0 for fully-collateralised trades?"* → Project 3 success criterion

These are the kinds of questions a senior rates / market-risk / XVA interviewer will ask. After this trilogy, you'll have lived the answers — not just read them.

## Honest about what 2 days isn't

A real bank's daily batch is the result of years of engineering, regulatory back-and-forth, and institutional context none of these capstones reproduce. What 2 days **does** deliver: you've made the calibration choices yourself, run the analytical pipelines end-to-end, and have something concrete (notebooks + reports + ANALYSIS.md) to walk an interviewer through. That's a **fluency capstone**, not a production system.

Two days won't make you the desk's senior quant. It'll make you sound like you've thought about being one — which is enough for a first-round interview screen, and gives you something concrete to keep extending. Each README has a Stretch Goals section pointing to the next layer (SABR-LMM, GARCH-filtered HS, WWR copula, CSA optimisation, FRTB-CVA capital).

## Getting started

```bash
# From the repo root
cd quant_finance/capstones/01_rates_portfolio
cat README.md          # read the spec
# Day 1: bootstrap curves, calibrate SABR + LMM, price the book including Bermudans, compute greeks
# Day 2: P&L attribution waterfall, B&R reconciliation, ANALYSIS.md
```

Each project's README has hourly milestones for both days. Aim to land Day 1's deliverables end-of-day-1 and you're calibrated for the pace of Day 2.

Good luck.
