# Just Group — Valuation Quant Interview Cheatsheet
### Python · Derivatives · Liquidity · Stress Testing  ·  *revise out loud*
*(Expert-reviewed: the actuarial core — NNEG, Matching Adjustment, longevity — is where Just will dig deepest.)*

## ⭐ MUST-KNOW COLD
1. **Just** = UK annuity specialist — **bulk-purchase annuities (BPAs)**, **medically-underwritten/enhanced individual annuities**, and **lifetime mortgages (equity release)**; now **Brookfield-owned**.
2. **Annuity liability** = long-dated (often **RPI/CPI-linked**) cashflows; value by **discounting**; hedge **rate + inflation** sensitivity (**LDI**). **Longevity is the dominant life risk** (annuitants living longer; deaths *help*, so mortality-CAT is immaterial here).
3. **PV01/DV01** = value change per **1bp** rate move; **IE01** = per 1bp **inflation** move. Core risk numbers.
4. **NNEG (No-Negative-Equity Guarantee)** on lifetime mortgages = the lender is **short a put on the property** → biggest LTM risk. *(Section 2 — most likely deep question.)*
5. **Matching Adjustment (MA)** = uplift to the **discount rate** for **eligible, cashflow-matched, ring-fenced** assets → **lowers BEL → raises Own Funds** (and dampens spread-SCR). `MA ≈ asset spread − Fundamental Spread`.
6. **Lifetime mortgages must be restructured/securitised** (senior note with fixed cashflows + junior tranche taking NNEG/prepayment risk) **to become MA-eligible**. This is *the* Just-specific mechanic.
7. **Liquidity risk = collateral / variation-margin calls on hedges** (+ repo roll, reinsurance collateral) — **not** claims (annuities are predictable).
8. **2022 LDI crisis** = leveraged **gilt repo/TRS** in *DB-pension* LDI → yields spiked → margin calls → gilt fire-sales → spiral → **BoE intervened**. **Insurers (incl. Just) were largely insulated** — MA buy-and-maintain, low leverage, not forced sellers.
9. **SCR** = capital to survive **1-in-200 (99.5%, 1-yr VaR)**. **Solvency ratio = eligible Own Funds / SCR** *(Just ~200%+ — say "check latest SFCR", don't quote a number you can't source).*
10. **Technical Provisions = Best-Estimate Liability (BEL) + Risk Margin.** **RM = CoC × Σ SCR(t)·DF(t)**; **Solvency UK cut CoC 6%→4%** → lower RM on long annuities.
11. **Longevity basis** = base tables (**CMI SAPS / S-series**) + **CMI improvement model** (long-term rate + initial addition). Just's edge = **enhanced/medically-underwritten** annuities (impaired lives → priced sharper).
12. **Reinsurance** of longevity (quota-share / longevity swaps; **funded vs funds-withheld** — PRA **SS5/24**) reduces longevity SCR but adds **counterparty** risk.
13. **Par swap rate** `S = (DF₀ − DF_N)/Σ τᵢ·DFᵢ` (single-curve approx); **receiver swap value = PV(fixed) − PV(float)** (payer = opposite sign).
14. **Real rate ≈ nominal − breakeven inflation** (Fisher approx).
15. **Python tested = applied quant coding** (discount, bump curve, stress, NNEG put) — not dev trivia.

---

## 1. Annuity liabilities, BEL & longevity  🔴
- **BEL** = probability-weighted PV of future benefit cashflows, best-estimate assumptions, discounted at risk-free **+ MA**.
- **Assumptions that move it:** **mortality/longevity** (base table + improvements), **expenses**, **inflation**, **discount rate**.
- **Longevity basis:** base = **SAPS/S-series (CMI)** tables; improvements = **CMI_20xx model** (long-term improvement rate + A-parameter). **Experience analysis** recalibrates to the book.
- **Just's edge:** **medically-underwritten / enhanced annuities** — individual health/lifestyle/postcode → higher annuity for impaired lives, better-matched longevity.

## 2. Lifetime Mortgages & NNEG  🔴🔴 (Just core — expect depth)
- **Lifetime mortgage (equity release):** lump sum to a homeowner; **interest rolls up**; repaid on **death / move into care / sale**. No regular payments.
- **NNEG = No-Negative-Equity Guarantee:** repayment capped at the **property sale value** → if rolled-up loan > house value at redemption, the lender eats the shortfall ⇒ **lender is SHORT a put on the house**.
- **NNEG valuation (option approach):** risk-neutral **Black-Scholes/Black-76 put** on the property. Inputs: house price, **house-price volatility (PRA floor ~13%)**, **deferment rate / net rental yield `q`** (PRA **floor**), roll-up rate (→ strike = rolled-up loan), and **redemption timing** (stochastic via mortality/morbidity/prepayment). Forward `F = S·e^(r−q)T`.
- **PRA SS3/17 — Effective Value Test (EVT):** prescribes minimum `q` (deferment-rate floor) and min vol; Just was central to the industry debate.
- **MA eligibility:** raw LTM cashflows are **uncertain** (NNEG + prepayment) → **not MA-eligible**. Insurers **restructure/securitise** the pool: a **senior note** with fixed, predictable cashflows (goes in the MA fund) + a **junior/residual tranche** absorbing NNEG/prepayment. *(Tie this to §3.)*
- **Risks:** house-price level & vol, voluntary **prepayment**, **longevity/morbidity** (longer occupancy → more NNEG exposure), illiquidity.

## 3. Matching Adjustment — mechanics & eligibility  🔴
- **What:** add an **MA uplift** to the risk-free curve to discount eligible annuity liabilities → **lower BEL, higher Own Funds**, and **less spread-risk SCR** (the matched book isn't marked through P&L on spread moves).
- **`MA = (asset portfolio spread over risk-free) − Fundamental Spread (FS)`.** **FS = PD allowance + cost-of-downgrade**, **prescribed and floored by the PRA** (not a free parameter).
- **Eligibility tests:** assets with **fixed/predictable cashflows**, **no issuer prepayment optionality** (callables problematic), **cashflow-matched** to liabilities (pass the **MA cashflow-matching test**), **ring-fenced portfolio**, credit-quality/sub-IG limits.
- **Downgrade hurts twice:** (i) **FS widens → MA falls → BEL rises**; (ii) the asset's **market value falls**.
- **Solvency UK reform:** wider eligible-asset universe (incl. **highly-predictable cashflows**), **senior-actuary attestation** of the MA, reduced EU constraints.

## 4. Reinsurance & longevity transfer
- **Longevity swap:** the **insurer pays fixed** (pre-agreed expected benefit payments + fee), **receives floating = actual** benefit payments → hedges **longevity-improvement** risk.
- **Funded (asset-intensive) vs funds-withheld** reinsurance; **PRA SS5/24** governs **funded reinsurance** (collateral, recapture, counterparty concentration).
- Reduces **longevity SCR**; adds **counterparty default** risk (SCR counterparty module + collateral).

## 5. Derivatives & valuation (your strength — be spoken-fluent)
- **Discounting:** `PV = Σ CFₜ·DFₜ`, `DFₜ = e^(−rₜt)` (or `1/(1+rₜ)ᵗ`). Collateralised hedges → **OIS/SONIA discounting**.
- **IR swap:** **receiver value = PV(fixed) − PV(float)**; **payer = opposite**. Par rate in §13.
- **Par swap rate:** single-curve `S=(DF₀−DF_N)/Σ τᵢDFᵢ`; under **dual-curve** value the float leg explicitly `Σ τᵢ·fᵢ·DFᵢ`.
- **PV01/DV01** = |ΔPV| per 1bp rate shift; **IE01** = per 1bp inflation; **key-rate** durations for curve shape. *(PV01 = par-rate shift; IE01 = inflation; SII **interest-rate stress** is a separate, larger shock.)*
- **Duration/convexity:** `ModDur = −(1/P)dP/dy`; `ΔP ≈ −ModDur·P·Δy + ½·Conv·P·Δy²`. Liabilities are **long-duration & convex**.
- **Swaptions:** **Black-76** on the forward swap rate. **Inflation:** RPI/CPI **zero-coupon swaps**; breakeven = nominal − real.

## 6. LDI / ALM
- Match **rate + inflation sensitivity** of liabilities with **receiver IR swaps, gilts (+ repo), inflation swaps**, longevity reinsurance. **Close cashflow matching** (needed for MA), not just duration matching.

## 7. Liquidity & 2022 LDI  🔴
- **Insurer's liquidity drains:** **variation-margin** on rate/inflation/FX hedges, **repo roll**, **reinsurance collateral** — fast, cash.
- **2022:** rates ↑ → MtM losses on **leveraged gilt/repo** LDI (DB pensions) → margin calls → fire-sales → spiral → **BoE TECRF**. **Just/insurers resilient:** low leverage, **MA buy-and-maintain = not forced sellers**, liquidity buffers.
- **Buffer = unencumbered cash + gilts** (you **repo gilts to raise cash** — repo is a *means*, not a buffer asset). Insurers run **survival-horizon / liquidity-coverage** frameworks (not bank LCR).
- **Liquidity stress test:** rate shock **+100/+200bp → size the margin call (≈ hedge DV01 × shock)** → check buffer covers it over the **survival horizon**; assess **time-to-liquidate**.

## 8. Solvency II / Solvency UK — SCR, RM, stress
- **SCR = 99.5% 1-yr VaR** of own funds; **MCR** = lower floor. **Standard formula vs internal model.**
- **Risk modules** (each a calibrated stress, aggregated via **correlation matrix**): **Market** (interest up/down, **spread**, equity, property, FX, concentration); **Life** (**longevity** dominant, mortality, lapse, expense, mortality-CAT); **Counterparty**; **Operational**.
- **Risk Margin = CoC × Σₜ SCR(t)·DF(t)** (**CoC 6%→4% under Solvency UK** → big RM cut on annuities). **TMTP** transitional runs off to **2032** (drives headline vs fully-loaded ratio).
- **Reverse stress testing** = find the scenario that breaches solvency/liquidity. **PRA Life Insurance Stress Test (LIST)** + **liquidity** stress are live PRA focuses.

## 9. Python — what they'll make you do
Clean, vectorised **functions**; explain, debug, discuss complexity. **Practise (fill-in, graded):** `just_valuation_drills.py` — (1) annuity/liability PV, (2) **PV01/DV01**, (3) **par swap rate**, (4) **duration/convexity** vs Taylor, (5) **rate stress + variation-margin sizing**, (6) **NNEG put** (Black-76); plus `just_pandas_drills.py` — liability panel → PV & portfolio DV01. Keys in the matching `*_SOLUTIONS.py`. Run stdlib drills with `python3`, pandas drill with `uv run python`; know `numpy.interp`, `scipy.optimize`.

---

## 🃏 Flashcards — "They ask → You say"
- **"Value an annuity in Python."** → discount each cashflow on the curve, sum; show DV01 by bumping the curve 1bp.
- **"Firm's biggest risks?"** → **longevity** + **NNEG/house-price** on lifetime mortgages + **rate/spread** on the matched book; managed by LDI, reinsurance, MA.
- **"What's the NNEG and how do you value it?"** → lender short a **put on the property**; **Black-76 put**, forward at the **deferment rate**, vol-floored; **SS3/17 EVT**. Higher with lower house growth, higher vol, longer term, higher roll-up.
- **"Why restructure an equity-release mortgage?"** → raw cashflows uncertain (NNEG/prepayment) → not MA-eligible; **senior note (fixed CFs)** earns the MA, **junior tranche** holds the risk.
- **"Explain the Matching Adjustment."** → discount uplift = **asset spread − fundamental spread** on eligible matched assets → lower BEL/higher Own Funds; **downgrade widens FS and hits asset value**.
- **"How does liquidity stress arise & why were insurers OK in 2022?"** → **VM calls on hedges**; insurers low-leverage, MA buy-and-maintain, buffered — not forced gilt sellers.
- **"Stress-test the book?"** → rates ±, spread widening, **longevity improvement**, MA/FS stress → impact on **Own Funds / SCR**.
- **"Risk Margin?"** → **CoC × Σ discounted future SCRs**; Solvency UK cut CoC to 4%.

## 📐 Formula quick-ref
```
PV            = Σ CFₜ·DFₜ            DFₜ = e^(−rₜt)  or  1/(1+rₜ)ᵗ
Par swap rate = (DF₀ − DF_N) / Σ τᵢ·DFᵢ      (single-curve approx)
Receiver swap = PV(fixed) − PV(float)        (payer = opposite sign)
PV01/DV01     = |PV(curve) − PV(curve+1bp)|   ;  IE01 = same on inflation curve
Mod duration  = −(1/P)·dP/dy ;  ΔP ≈ −ModDur·P·Δy + ½·Conv·P·Δy²
NNEG put      = e^(−rT)[K·N(−d₂) − F·N(−d₁)],  F = S·e^(r−q)T,  K = rolled-up loan
MA            ≈ asset spread − Fundamental Spread (FS = PD + cost-of-downgrade, floored)
Risk Margin   = CoC × Σₜ SCR(t)·DF(t)        (CoC 4% under Solvency UK)
Solvency      = eligible Own Funds / SCR     (SCR = 99.5% 1-yr VaR)
Real rate     ≈ nominal − breakeven inflation
```

*Prep only — confirm Just's exact methodology/figures where you can. Tonight's priority (per expert review): **NNEG/lifetime-mortgage valuation, Matching-Adjustment eligibility, longevity basis, and the Python drills** — your derivatives maths is already strong; just say it fluently and don't freeze on the actuarial vocabulary.*
