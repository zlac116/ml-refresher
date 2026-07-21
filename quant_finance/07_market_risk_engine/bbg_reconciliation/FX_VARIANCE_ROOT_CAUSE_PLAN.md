# FX Forward/Swap Variance vs BBG — Root Cause Plan

Context: base PV / stressed PV / impacts >7% off BBG for FX forwards & swaps,
outside any bid-offer threshold. Xccy basis ruled out (correctly applied on
non-USD DFs). Suspected cause: no DCC applied, and DF interpolation uses "DF
for the month" rather than the exact cashflow date. Same method is used for
IRS/bonds/xccy basis swaps, which reconcile fine — production engine refactor
is high-risk/slow (matrix multiplication, cashflow dates jettisoned), so root
cause is isolated in a small reference re-pricer, not fixed in production.

## Step 0 — BBG DF confirmation (fastest, do first)
- Pull BBG's per-cashflow `DF(T)` from SWPM for the sample trades.
- Reprice using in-house cashflow amounts × BBG's DF.
- **Matches BBG PV** → cashflows/dates/amounts are correct; 100% of the gap is
  discounting methodology (interp/DCC).
- **Doesn't match** → a cashflow-generation issue also exists; investigate
  before touching discounting.

## Step 1 — controlled ablation (not a single "fixed" engine)
Build 3 variants off the **same DF curve pillars** production uses:

| Variant | Method | Purpose |
|---|---|---|
| 0 | Replicate production: month-bucket DF, no DCC | Calibration check — must reproduce the ~7% gap |
| 1 | Exact cashflow-date interpolation, no DCC | Isolates interpolation-granularity effect |
| 2 | Exact-date interpolation + correct DCC (ACT/365 GBP, ACT/360 USD/EUR) | Should land within bid-offer/noise floor of BBG |

- Produces a quantified bridge: X bp from interpolation, Y bp from DCC, Z bp
  unexplained.
- Prediction to test: **interpolation dominates, DCC is minor** — month-level
  date rounding can be ~15–30 days out of a ~60–120 day tenor; DCC mismatch is
  a much smaller relative effect.
- Reuse `recon/pv_engine.py` as the scaffold; add DCC/interp-granularity
  toggles.

## Step 2 — explain why only FX fwd/swap is affected
- FX fwd/swap PV is a near-cancelling residual of two large legs concentrated
  in 1–2 cashflows on a short tenor — a fixed date error hits full notional,
  undiluted, and is a large fraction of the tenor.
- IRS/bonds/xccy basis swaps use the same method but have periodic, smaller
  cashflows and/or longer tenors — the same absolute error is diluted.
- **Control case**: xccy basis swaps share the cross-currency mechanics but
  are longer-dated/multi-cashflow and reconcile fine — strong supporting
  evidence.
- Optional/stretch: run Variant 0→2 on one IRS trade to show a much smaller
  improvement.

## Step 3 — package for model validation
- **Root cause**: quantified via Steps 0–1 (DF-level match + ablation bridge).
- **Materiality**: gap vs justified thresholds (bid-offer / empirical noise
  floor / IPV policy).
- **Interim mitigant**: flag FX fwd/swap marks as elevated model risk, or
  apply a documented reserve/AVA until remediated.
- **Remediation item**: narrow-scope engine fix (DCC + exact-date
  interpolation for FX fwd/swap only) tracked with owner + target date —
  separate from this recon, not attempted now.
- Production refactor of the shared cashflow/matrix engine stays **out of
  scope** — too high-risk to rush and could destabilize products that
  currently reconcile fine.
