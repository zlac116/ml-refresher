# Reconciling BBG vs In-house — FX Forwards & Swaps

How to run the reconciliation and diagnose breaks. Scope: FX forwards & swaps;
scenarios = BASE, +100bp all, +100bp GBP-only, −100bp GBP-only.

## Run order
- `python capture_inhouse.py` — harvest in-house base + 3 stressed PVs → `data/inhouse/inhouse_canonical.csv`.
- Export the same trades + scenarios from BBG → `data/bbg/bbg_canonical.csv` (match `scenario_id` exactly).
- `python -m pytest -q` — pass/fail gate per (trade, scenario, metric).
- `python run.py` — emit the pack (`data/output/recon_pack.html`).

## Reconcile method (per trade)
- **Confirm the inputs match first:** same `fx0`, same cashflow amounts, same settlement date. Rule these out before touching curves.
- **Reconcile base PV, then each stress impact** (`stressed − base`) separately — level vs sensitivity are different questions.
- **Measure the break in bp of notional, not % of PV.** FX-fwd PV is a small residual of two large legs, so tiny DF diffs show as huge % — % will mislead you.
- **Isolate to the DF:** pull BBG's per-leg `DF(T)` from SWPM and compare directly to in-house `DF(T)` at the same date.
  - DFs match → base PV must match; any residual is mechanics (dates / conversion path).
  - DFs differ → it's curve / basis / interp (below).

## Reading BBG
- Value/inspect in **SWPM** (Cashflow / Details tab) to see per-cashflow **DF + discount curve number**; drill the curve via `ICVS <curve#>`. FXFA/FRD (forward-points) won't show explicit DFs.
- **xccy basis is never a line item.** Either it's baked into the FX forward points (FXFA mode) or into the non-GBP leg's discount curve (SWPM dual-curve mode). **Establish which mode BBG uses** — mismatch vs in-house is the most likely break.

## Break drivers (most-likely first)
- **Discounting curve** — OIS/CSA (SOFR/€STR/SONIA) vs in-house funding/single curve → different `DF(T)` on both legs.
- **Xccy basis** — in forward points vs in the discount curve; handled differently on each side.
- **Day-count on t** — GBP ACT/365 vs EUR/USD ACT/360 for the year fraction.
- **DF interpolation** — log-linear-on-DF vs linear-on-zero-rate; matters because settlement rarely lands on a pillar. Interp at the **exact cashflow date**.
- **Settlement date / calendars** — T+2 spot lag, GBP/TARGET/USD holidays shift `t` and `DF(T)`.
- **Conversion path** — discount foreign leg on its own curve, convert at **spot** (not forward), else the rate differential is double-counted.

## Stress-impact specifics
- **Bump convention** — +100bp on the zero rate vs re-bootstrapped par shift; additive-to-rate vs to-DF. At 100bp the gap is second-order-large.
- **Convexity** — `DF=exp(−z·t)` is nonlinear, so +100 and −100 GBP impacts are **asymmetric**; a linear in-house approximation won't match BBG's full reval.
- **"GBP only" scope** — does it bump only the GBP curve, or also the forward points / basis linking the legs? Define identically on both sides.
- **Fastest localiser** — if `+100bp all` reconciles but GBP-only doesn't, it's the leg-scope/basis definition, not the curve.
