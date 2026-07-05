# LIBOR–OIS Curve Migration — Cheatsheet
### Multi-curve · RFRs · fallbacks · valuation/P&L/risk impact  ·  *revise out loud*

> **20-sec version:** *"I led the risk-and-valuation side of moving our IR books off LIBOR-based curves onto the OIS/RFR multi-curve framework — rebuilding the discount and forecast curves, quantifying the one-off P&L from the discounting and fallback changes, and re-estimating risk so delta moved from LIBOR onto SONIA/SOFR plus basis."*

---

## 1. WHY it happened (two linked drivers — know both)
**(a) OIS discounting (post-2008).** Pre-GFC one **single LIBOR curve** both *projected* forwards and *discounted*. In 2008 the **LIBOR–OIS spread blew out** (LIBOR carries bank credit + liquidity premium; OIS ≈ risk-free). That exposed the truth: a **collateralised (CSA) trade must be discounted at the collateral rate (OIS/overnight)** — the actual funding cost of posted cash. → birth of **multi-curve**.

**(b) LIBOR cessation → RFRs (2017–2023).** LIBOR was **submission-based** (thin real transactions) and **manipulated** (2012). FCA (2017) ended it → replaced by transaction-based **Risk-Free Rates**. GBP/EUR/CHF/JPY LIBOR ceased **end-2021**, USD LIBOR **mid-2023**. So the curve infrastructure had to *migrate* from LIBOR to RFR.

## 2. RFRs (Risk-Free Rates)
Near risk-free, **overnight, transaction-based** benchmarks that replaced LIBOR:
| RFR | Ccy | Secured? |
|---|---|---|
| **SONIA** | GBP | Unsecured overnight |
| **SOFR** | USD | Secured (repo) |
| **€STR** | EUR | Unsecured |
| **SARON** | CHF | Secured |
| **TONA** | JPY | Unsecured |
vs **LIBOR** = forward-looking **term** rate, **credit-sensitive**, submission-based. RFRs are overnight → you **compound them** to make a term rate.

## 3. Compounding in arrears (turn overnight → term rate)
- **LIBOR = "in advance"**: the 3M rate is known at the **start** of the period.
- **RFR = "in arrears"**: no term rate exists, so you **compound the daily overnight fixings across the period** → known only at the **end**.
```
Compounded rate = [ Π (1 + rᵢ·dᵢ/365) − 1 ] × 365/D
```
`rᵢ` = overnight RFR on day i, `dᵢ` = calendar days it applies (Fri covers Sat+Sun), `D` = period days.
- **In-arrears headache & fixes:** you don't know the coupon until period-end → **lookback**, **observation shift**, **lockout**, **payment delay**; or a forward-looking **Term SOFR** (from SOFR futures) where a known-in-advance rate is needed.

## 4. Multi-curve framework (the technical core)
**Single-curve (old):** one LIBOR curve projects *and* discounts (assumes LIBOR ≈ risk-free).
**Multi-curve (new):** **project on one curve, discount on another.**
- **Discount curve = OIS/RFR** (bootstrapped from **OIS swaps**) — the CSA collateral rate.
- **Projection/forecast curve** = per-tenor LIBOR (1M/3M/6M) or **compounded-RFR**, **bootstrapped from FRAs, futures & basis swaps that reference that index** — so its forwards reproduce the market's fixings.

**Why you can't derive LIBOR forwards from the OIS curve:** LIBOR = expected overnight (≈OIS) **+ a credit/liquidity premium**. OIS at 4.00% implies a 4.00% forward; 3M LIBOR actually trades ~4.30% (a 30bp spread). Each tenor has its **own** projection curve (tenor basis) — impossible in single-curve.
```
forward L(T₁,T₂) = (P_proj(T₁)/P_proj(T₂) − 1)/τ      ← forwards off the PROJECTION curve
Swap PV = Σ fixed·DF_OIS − Σ (projected float)·DF_OIS  ← discount BOTH legs on OIS
```
> **Projection = forward curve** (two names, one object: produces the fixings). **Discount curve** = a *different* curve (PVs cashflows). Post-LIBOR, for the RFR itself the two largely **re-converge** onto one SONIA/SOFR curve.

## 5. Fallbacks (legacy LIBOR trades that outlived LIBOR)
A **fallback** = the contractual replacement that kicks in automatically on cessation.
- **ISDA fallback (derivatives):** `LIBOR → compounded RFR (in arrears) + fixed spread adjustment`.
- **Spread adjustment** = the **5-year median of the historical (LIBOR − compounded-RFR) gap**, **fixed once on 5 March 2021** (compensates for LIBOR's credit/term premium so neither party gains/loses at switch). E.g. **GBP 3M ≈ 11.9bp**, **USD 3M ≈ 26.2bp**.
- Delivered via the **ISDA 2020 IBOR Fallbacks Protocol**.
- **Active transition** = *voluntarily* convert to RFR before cessation; **fallback** = the *automatic backstop* for trades not transitioned.

## 6. Valuation / P&L / risk impact (your three deliverables)
- **Valuation:** switching discounting LIBOR→OIS **re-prices every trade** (most for off-market/long-dated); the LIBOR–OIS spread feeds directly into PV. Collateralised vs uncollateralised → different discounting (→ FVA/XVA).
- **P&L:** at the switch, books **revalue → a one-off P&L jump**; fallback spread adjustments crystallise an economic transfer. Job: **measure, attribute, explain** it.
- **Risk:** DV01/delta **reallocates from LIBOR onto SONIA/SOFR + LIBOR–OIS basis + tenor basis**; new risk buckets; hedges re-struck; **basis risk** during the transition window.

## 7. Your project narrative (first person)
*"From the risk-and-valuation seat I owned the curve-framework migration: redesigned and rebuilt the **OIS-discount and RFR-forecast curves** and validated consistency; **quantified the P&L** of the discounting switch and the ISDA fallback spread adjustments across the affected IR books; and **re-estimated risk** so exposures, sensitivities and hedges moved onto the RFR-plus-basis world. Coordinated across desks, market risk and finance, and signed off the methodology."*

## 8. Likely Q&A
- **"Why OIS discounting?"** → collateral (CSA) rate is the true funding cost of a collateralised trade; the blown-out LIBOR–OIS spread proved LIBOR discounting mis-values them.
- **"Multi-curve framework?"** → separate **OIS discount** + **LIBOR/RFR forecast** curves; project on one, discount on the other.
- **"How do RFR fallbacks work?"** → compounded RFR in arrears **+ fixed 5y-median spread adjustment** (set Mar-2021).
- **"SONIA vs LIBOR?"** → SONIA overnight, transaction-based, **compounded in arrears**; LIBOR forward-looking term, submission-based.
- **"P&L impact & why?"** → discount-curve change + fallback spread → revaluation; plus basis P&L.
- **"Transition risks?"** → **basis risk** (old vs new curve), conduct/value-transfer, operational, liquidity differences.

## 📐 Formula quick-ref
```
LIBOR–OIS basis  = LIBOR − OIS   (credit + liquidity premium)
Compounded RFR   = [Π(1 + rᵢ·dᵢ/365) − 1]·365/D
Forward (proj)   = (P_proj(T₁)/P_proj(T₂) − 1)/τ
Swap PV          = Σ fixed·DF_OIS − Σ (projected float)·DF_OIS
Fallback rate    = compounded RFR + spread adj (5y median of LIBOR − RFR, fixed Mar-2021)
```

## Glossary
| Term | One-liner |
|---|---|
| **OIS** | Overnight Index Swap — discount/collateral (RFR) curve |
| **RFR** | Risk-Free Rate: overnight, transaction-based (SONIA/SOFR/€STR) |
| **Projection/forward curve** | Produces the index's future fixings (per tenor) |
| **Discount curve** | PVs cashflows (OIS/collateral rate) |
| **Compounding in arrears** | Build a term rate from realised daily RFR fixings |
| **Fallback** | Auto contractual replacement on LIBOR cessation |
| **Spread adjustment** | Fixed 5y-median LIBOR−RFR gap added on fallback |
| **Tenor basis** | Spread between different LIBOR tenor projection curves |
| **CSA** | Credit Support Annex — collateral terms → OIS discounting |

*Prep only. The multi-curve "project on one, discount on OIS" idea and the fallback "compounded RFR + fixed spread" are the two things interviewers probe most.*
