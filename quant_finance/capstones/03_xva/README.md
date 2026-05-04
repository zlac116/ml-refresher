# Project 3 — XVA Computation, Sensitivities & Hedging on a Multi-Asset Portfolio (2-Day Sprint)

## Project goal

You're the new joiner on the **XVA desk** (a.k.a. CCR Trading) at a tier-1 bank. The desk computes, charges, hedges and reports Valuation Adjustments across the firm's full counterparty book. You inherit a **combined book** stitched from Projects 1 (rates) and 2 (FX) — ~400 trades plus 10 cross-currency swaps — aggregated to **30 counterparties**. In 2 days you bootstrap hazards, calibrate the joint correlation, run the full XVA stack (CVA + DVA + FVA + MVA + KVA), produce CVA sensitivities, and propose a CDS hedge book sized via LP.

Plumbing is pre-built. You do calibration, exposure analysis, hedge construction and the desk-level calls.

## Why this matters

XVA is the single biggest pricing addition introduced in derivatives markets since 2008. CVA alone runs to billions on tier-1 balance sheets. DVA sparked accounting wars (FASB ASC 820 vs IFRS 13). FVA forced Goldman to take a one-time $850M charge in 2014; JPM took $1.5B. The XVA desk is where credit, market and capital risk meet — and where post-Lehman regulation lives.

You'll exercise:
- **Hazard-rate stripping from CDS via Brent** — `02_risk/04_cva_intro.ipynb`
- **Joint rates-FX correlation calibration** from historical regression — required input to the simulator
- Joint MC of rates + FX under Q (`06_stoch_calc/02_ito_and_gbm.ipynb`, `01_options/05_monte_carlo_pricing.ipynb`)
- Path-by-path repricing using Project 1's swap/swaption pricers and Garman-Kohlhagen for FX
- LSMC for Bermudan exposure (`06_stoch_calc/03_lsmc_american.ipynb`)
- Netting, CSA mechanics, and the full XVA stack
- Finite-difference sensitivities and **CDS hedge LP construction**
- (Stretch) Wrong-way risk via Gaussian copula

## The realistic IB context

XVA sits in **FICC** at most banks. It charges every other desk for the credit / funding / capital cost of their trades — every new trade gets an XVA quote, and only goes live if the originating desk wears the cost.

**Why each component exists**:

- **CVA**: expected loss from counterparty default before maturity. Always a cost. Universally adopted; biggest XVA component.
- **DVA**: symmetric benefit when **you** default. Controversial — IFRS allows, US GAAP restricts.
- **FVA**: cost of funding the uncollateralised in-the-money exposure. Goldman 2014 was the watershed moment.
- **MVA**: cost of funding initial margin (CCP / SIMM).
- **KVA**: lifetime cost of regulatory capital (Basel III RWA × CoE). Most controversial; not all banks compute.

**Typical XVA sizes** (rough industry numbers): uncollateralised 5y vanilla swap, IG corp: CVA ~5-15bp; HY: ~30-100bp; two-way CSA (daily, MTA $250k): ~0-1bp; FVA on uncollateralised: 1-5bp; KVA under SA-CVA: 5-20bp.

## What's pre-built vs what you build

The 5-week version had you building everything from the joint simulator up. The 2-day version gives you production-grade scaffolding so you spend the time on **calibration, analysis and hedging**, not plumbing.

**Pre-built (in `src/`, `data/`) — load and use, do not rebuild**:

| Component | Path | What it does |
|---|---|---|
| Combined trade book | `data/trades_combined.parquet` | ~400 trades from Projects 1+2 + 10 xccy swaps, tagged with `counterparty_id` |
| Counterparty universe | `data/counterparties.parquet`, `data/csas.parquet` | 30 counterparties + CSA terms |
| **Raw CDS spreads** | `data/cds_quotes/cds_<cpty>.parquet` | Spreads at 1y/3y/5y/7y/10y — **you bootstrap from these** |
| **Reference hazards** | `data/hazards_reference.parquet` | Sanity-check target |
| **Historical rates+FX panel** | `data/historical_rates_fx.parquet` | 2y daily — **you calibrate correlation from this** |
| Joint MC simulator | `src/exposure/joint_simulator.py` | Correlated rates + FX paths under Q; predictor-corrector LMM |
| Repricing engine | `src/exposure/reprice.py` | Path-by-path; LSMC for Bermudans |
| Netting + CSA | `src/exposure/netting.py` | Per-counterparty aggregation with threshold/MTA/frequency |
| Profile computation | `src/exposure/profiles.py` | EE, EPE, ENE, PFE |
| XVA calculators | `src/xva/{cva,dva,fva,mva,kva}.py` | Black-box; MVA via SIMM proxy ($k\sigma_{1d}\sqrt{10}$); KVA configurable CoE |
| FD sensitivity engine | `src/sensitivities.py` | Bump-revert harness; you supply bumps |
| Hedge LP harness | `src/hedging.py` | scipy.optimize.linprog wrapper; you supply objective + constraints |
| Reporting | `src/reports.py` | HTML templates |

**You build (the actual quant work)**:

1. **Hazard-rate bootstrap** (Brent root-finding) per counterparty under ISDA standard ($R = 40\%$, piecewise-constant $\lambda$). Match `hazards_reference.parquet` within 10bp at every node
2. **Joint correlation calibration**: from `historical_rates_fx.parquet`, daily-return correlation across IR rates (3M, 1Y, 5Y, 10Y) and FX (USD-EUR, USD-JPY, USD-GBP). Eigenvalue-clip if not PSD; document the choice
3. **Per-counterparty exposure analysis**: read EPE/PFE shapes; identify outliers
4. **Netting impact study**: with vs without netting → 60-80% reduction
5. **XVA decomposition**: which component dominates per counterparty? Per asset class?
6. **Sensitivity table** (FD bumps): bump CDS / IR vol / FX spot ±1%, recompute CVA, build the table; verify $\partial \text{CVA} / \partial \text{LGD} = \text{CVA}/\text{LGD}$
7. **CDS hedge LP**: minimise total CDS premium spend subject to neutralising CVA delta to ±5% per top-10 counterparty (single-name) and ±10% on index buckets (CDX IG / CDX HY / iTraxx Main)
8. **(Optional stretch) WWR**: Gaussian-copula $\rho$ between sovereign hazard and FX from rank-transformed historical changes; recompute CVA
9. `ANALYSIS.md` (~1 page): dominant XVA + why; model approximations; production additions

## Synthetic data + scaffolding plan

30 counterparties (compressed from the 5-week version's 50), ~400 trades — enough to make netting/concentration effects clear without burning the budget on MC runtime.

**Counterparty universe**: 8 IG financials (50-150bp, two-way daily MTA $250k), 10 IG corps (30-100bp, weekly MTA $1M threshold $5M), 6 HY (200-500bp, weekly MTA $500k threshold $0), 3 sovereigns (5-50bp, uncollateralised), 3 hedge funds (rating proxy, daily MTA $100k).

**Trade book**: ~390 trades from Projects 1+2 + 10 xccy basis swaps in USD-EUR / USD-JPY / USD-GBP, $50-500MM notional. Xccy is the canonical FVA driver.

**MC sizing**: 5,000 paths, weekly grid out to 30y, monthly thereafter. ~5-10 min per full revaluation on a laptop. Use 1,000 paths for development, 5,000 for the report.

## Project tree structure

```
quant_finance/capstones/03_xva/
├── README.md
├── data/                               (PRE-BUILT)
│   ├── trades_combined.parquet
│   ├── counterparties.parquet, csas.parquet
│   ├── cds_quotes/cds_<cpty>.parquet
│   ├── hazards_reference.parquet
│   ├── historical_rates_fx.parquet
│   └── funding_curve.parquet
├── src/                                (PRE-BUILT)
│   ├── exposure/{joint_simulator,reprice,netting,profiles}.py
│   ├── xva/{cva,dva,fva,mva,kva}.py
│   ├── sensitivities.py
│   ├── hedging.py
│   └── reports.py
├── notebooks/                          (YOU BUILD)
│   ├── 01_calibration_exposure_xva.ipynb   (Day 1)
│   └── 02_sensitivities_and_hedge.ipynb    (Day 2)
├── reports/                            (generated)
│   ├── hazard_bootstrap.html
│   ├── exposure_profiles.html
│   ├── xva_breakdown.html
│   ├── sensitivities.html
│   └── hedge_proposal.html
└── ANALYSIS.md
```

## 2-day milestones

### Day 1 — Calibration + exposure + netting + full XVA stack (~8h)

Goal: do the calibration work, get the engine running on all 30 counterparties, output the full XVA breakdown.

- **Hour 0-0.5**: load pre-built data. Sanity-check trade count per counterparty, gross notional per sector
- **Hour 0.5-2.5**: **Hazard-rate bootstrap (Brent)**. For each counterparty, for each tenor (1y/3y/5y/7y/10y), invert the CDS spread under the ISDA standard model (PV premium = PV contingent given piecewise-constant $\lambda$). Bootstrap forward: $\lambda_1$ from 1y, $\lambda_2$ from 3y given $\lambda_1$, etc. Compare to `hazards_reference.parquet` — should match within 10bp at every node. Output: `reports/hazard_bootstrap.html`
- **Hour 2.5-3.5**: **Joint correlation calibration**. Daily-return correlation across IR rates (3M, 1Y, 5Y, 10Y) + FX (USD-EUR, USD-JPY, USD-GBP). Eigenvalue-clip negatives to 1e-6 (or Ledoit-Wolf shrink) if not PSD. Document. Note where the rates-FX block is large (typically negative for USD-funded EM)
- **Hour 3.5-5**: run `joint_simulator.py` with 5,000 paths on all 30 counterparties using the calibrated correlation. Cache the path tensor — every downstream call (XVA, sensitivities) reuses it. Reprice via `reprice.py`; aggregate via `netting.py`; profiles via `profiles.py`
- **Hour 5-6**: **netting impact study** — re-aggregate with netting **off** vs **on**. Quantify per-counterparty reduction. Sanity check: vanilla swap exposure peaks early-life and decays linearly; xccy rises with FX vol
- **Hour 6-8**: **full XVA stack**. Call `cva.py / dva.py / fva.py / mva.py / kva.py` on all 30 using cached exposure paths. Build the breakdown: total = Σ(CVA + DVA + FVA + MVA + KVA); split by counterparty (top 10) and asset class (rates / FX / xccy). Sanity bands: collateralised CVA <10% of uncollateralised; xccy dominates FVA per dollar; sovereigns highest CVA per dollar of notional
- **Deliverable**: `notebooks/01_calibration_exposure_xva.ipynb` + `reports/hazard_bootstrap.html` + `reports/exposure_profiles.html` + `reports/xva_breakdown.html`

### Day 2 — Sensitivities + hedge proposal + (optional) WWR + writeup (~8h)

Goal: build the CVA risk dashboard, propose a hedge book sized via LP, optional WWR, write up.

- **Hour 0-3**: **FD sensitivities** via `sensitivities.py`. Bump each of {parallel IR ±1bp, IR vol +1%, USDEUR/USDJPY/USDGBP spot +1%, parallel CDS +1bp per counterparty} and recompute CVA. Build the per-counterparty sensitivity table. Verify the analytic identity $\partial \text{CVA}/\partial \text{LGD} = \text{CVA}/\text{LGD}$ within MC noise per counterparty
- **Hour 3-5.5**: **Hedge LP construction**. Use `hedging.py` to solve:
  - **Objective**: minimise Σ (CDS premium × notional) across single-name + index hedges
  - **Constraints**: per top-10 counterparty, post-hedge CVA delta within ±5% of pre-hedge (single-name CDS); per index bucket (CDX IG / CDX HY / iTraxx Main), aggregated residual delta within ±10%; non-negative notionals
  - Report pre vs post CVA delta; estimated annual hedge carry (premium × notional summed)
- **Hour 5.5-7 (optional stretch)**: **WWR demo on one EM sovereign**. Calibrate $\rho$ between sovereign hazard and local FX rate via Pearson on rank-transformed historical changes (typically -0.3 to -0.6 for distressed sovereigns). Plug into the joint simulator; recompute CVA under the WWR Gaussian copula. Expect 10-30% uplift; document. **Skip if Day 2 is tight — sensitivities + hedge are core, WWR is bonus**
- **Hour 7-8**: write `ANALYSIS.md` (~1 page): (a) which XVA dominates and why; (b) where the model is approximate (correlation calibration, WWR treatment, KVA CoE, SIMM proxy); (c) what production XVA adds (intraday revaluation, FRTB-CVA capital, CSA optimisation, full SIMM, cross-asset WWR)
- **Deliverable**: `notebooks/02_sensitivities_and_hedge.ipynb` + `reports/sensitivities.html` + `reports/hedge_proposal.html` + `ANALYSIS.md`

## Final deliverables

1. **`reports/hazard_bootstrap.html`** — bootstrapped vs reference hazards, residuals per counterparty per tenor
2. **`reports/exposure_profiles.html`** — EPE(t), PFE(t), collateralised vs uncollateralised, netting reduction % for top 10
3. **`reports/xva_breakdown.html`** — desk-level CVA / DVA / FVA / MVA / KVA; top 10 contributors; split by asset class
4. **`reports/sensitivities.html`** — d(CVA)/d(IR), d(CVA)/d(IR vol), d(CVA)/d(CDS spread), d(CVA)/d(FX); LGD identity check
5. **`reports/hedge_proposal.html`** — single-name CDS notionals (top 10) + index buckets; pre vs post CVA delta; annual carry
6. **`ANALYSIS.md`** (~1 page) — dominant XVA + why; 3 model approximations; 3 production additions

## Hints

- **Hazard bootstrap**: under ISDA standard, the spread $s_T$ at tenor $T$ satisfies the breakeven condition between premium and protection legs. Bootstrap forward — $\lambda_1$ from 1y, $\lambda_2$ from 3y given $\lambda_1$, etc. Brent works directly on the breakeven equation per tenor. If your residuals are >10bp on the 1y but small further out, check day-count and accrual-on-default convention before blaming the solver
- **Correlation matrix PSD-ness**: historical correlations are noisy; eigenvalue-clip negatives to 1e-6 and renormalise, or use Ledoit-Wolf shrinkage. Document
- **Cache the path tensor.** Every downstream step (XVA, sensitivities) reuses the same simulated paths. Re-running MC for each component is the single biggest way to blow the day budget
- **Wrong-way risk** is the elephant. Standard CVA undervalues sovereign WWR by 20-50%. The canonical case: USD swap with EM sovereign — in-the-money to you precisely when local FX has tanked. The optional WWR demo is worth the 1.5h if Day 2 is on track
- **Netting matters enormously**. One ISDA Master = one netting set — get the boundary wrong and CVA is 10x off
- **Exposure vs pricing measure**: Bermudans simulated under the **exposure measure** (T-forward) with LSMC for early-exercise. `reprice.py` handles it — but understand the distinction for the interview
- **MVA via SIMM proxy**: don't reimplement SIMM. $\text{IM} = k \cdot \sigma_{1d} \cdot \sqrt{10}$ with $k \approx 3.6$ for 99% gives ballpark IM ~3-5% of notional on a 10y IRS
- **KVA controversy**: Goldman, JPM, MS compute it; some European banks don't. CoE assumption (10-15%) is itself a major model choice — flag in `ANALYSIS.md`
- **CDS hedge LP**: in production, two layers — single-name CDS for top 10-20 contributors (~80% of CVA), index CDS for the long tail. IR / FX delta of CVA is hedged with the same instruments as the underlying trades ("CVA market hedges")
- **CVA capital ≠ CVA P&L**. Basel III SA-CVA / BA-CVA is separate. Mention in `ANALYSIS.md` as a "production addition"
- **Sanity bands** (no commercial pricer to check against): netting reduces single-counterparty CVA 60-80%; collateralised CVA <10% of uncollateralised; sovereign CVA highest per dollar; xccy generates most FVA per dollar. If you're outside these bands, the issue is upstream (netting set boundary, CSA application, correlation matrix)

## Cross-references

| Concept | Where it's covered |
|---|---|
| Hazard-rate stripping from CDS | `02_risk/04_cva_intro.ipynb` |
| EE / EPE / ENE / PFE | `02_risk/04_cva_intro.ipynb` |
| CVA integral form | `02_risk/04_cva_intro.ipynb` |
| Merton structural credit | `02_risk/03_credit_merton.ipynb` |
| GBM joint simulation | `06_stoch_calc/02_ito_and_gbm.ipynb` |
| Monte Carlo + variance reduction | `01_options/05_monte_carlo_pricing.ipynb` |
| LSMC backward induction | `06_stoch_calc/03_lsmc_american.ipynb` |
| LMM (path-dependent rates) | `03_fixed_income/05_libor_market_model.ipynb` + `lmm_cheatsheet.md` |
| Garman-Kohlhagen FX option exposure | `01_options/02_bs_family_and_asset_classes.ipynb` Part 4 |
| SABR vol calibration | `01_options/02_bs_family_and_asset_classes.ipynb` Part 2 + `sabr_cheatsheet.md` |
| Black-76 swap / swaption pricing | `01_options/02_bs_family_and_asset_classes.ipynb` + `bachelier_cheatsheet.md` |
| Multi-curve discounting | `03_fixed_income/03_curve_building.ipynb` |
| Swap pricing / par rate | `03_fixed_income/04_swaps_swaptions.ipynb` |
| Brent root-finding | `01_options/01_black_scholes.ipynb` (IV inversion uses same pattern) |
| Black-Scholes core (Greeks etc.) | `01_options/01_black_scholes.ipynb` + `bs_cheatsheet.md` |

## Success criteria

The 2-day project is "done" when **all** hold:

1. **Hazard bootstrap** matches reference within 10bp of hazard at every node for every counterparty
2. **Calibrated correlation matrix** is PSD; method documented (raw / clipped / shrunk)
3. **Sanity bounds met**: netting reduces single-counterparty CVA ≥60% on at least 3 counterparties; fully-collateralised CVA <10% of uncollateralised; FVA + DVA → 0 for fully-collateralised trades
4. **CVA / LGD identity** holds within MC noise: $\partial \text{CVA}/\partial \text{LGD} = \text{CVA}/\text{LGD}$ per counterparty
5. **CVA / CDS spread sensitivity** is positive and non-linear (1bp ≠ 10bp scaled)
6. **Top-10 hedge proposal** neutralises CVA delta to ±5% per counterparty using LP-optimised CDS notionals; cost reported
7. **Cross-currency swaps** are the largest single FVA driver per dollar of notional
8. (If WWR stretch attempted) Sovereign CVA increases 10-30% under the Gaussian copula link; calibrated $\rho$ documented
9. `ANALYSIS.md` exists, names the dominant XVA component, lists 3 model approximations and 3 production additions

## What this prepares you for

- **XVA desk interview**: you can speak about every component, why each matters, and the genuine industry debates (DVA recognition, KVA inclusion, FVA vs OIS-with-funding-curve)
- **Counterparty risk regulatory roles**: BA-CVA vs SA-CVA, FRTB-CVA, ISDA SIMM
- **Risk strategist** at a fintech / hedge fund trading uncollateralised OTC products
- **Quant developer building exposure engines** at banks (XVA system rebuilds are ongoing) or vendors (Numerix, Calypso, FIS)

The single biggest interview signal: you can articulate the **operational complexity** of XVA — that it's not just a mathematical adjustment, but an entire desk function with its own P&L, hedging book, regulatory capital allocation, and accounting recognition rules. The killer question is *"Walk me through what happens overnight in the XVA desk's batch."* You should spend 5 minutes on that with follow-up depth on every choice.

Two days is a **fluency capstone**, not a production system. Real XVA desks run dozens of engineers and revalue intraday. What you've built has realistic infrastructure and the genuine quant decisions (Brent bootstrap, correlation calibration, hedge LP, optional WWR copula) preserved as user work — enough to speak fluently to anyone who runs the real thing.
