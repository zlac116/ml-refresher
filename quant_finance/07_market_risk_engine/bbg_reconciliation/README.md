# FX Reconciliation Pack — in-house vs Bloomberg

Independent benchmarking of the in-house valuation/stress engine against
Bloomberg, for **model validation**. Reconciles **base PVs** and **stress
impacts** for a sample of trades.

**Minimal scope: FX forwards & swaps only.** (The structure extends to other
products by adding a product adapter + config entries — not built yet.)

This is **benchmarking / outcomes-analysis evidence** (SR 11-7 / PRA SS1/23),
not a full independent revaluation.

## Pipeline

```
 input files ──▶ engine ──▶ data/inhouse/inhouse_canonical.csv ┐
 (cashflows +   (capture)                                      ├─▶ merge ─▶ diff ─▶ classify ─▶ PACK
  DF curves)                                                   │
 Bloomberg export ─────────normalise──▶ data/bbg/bbg_canonical.csv ┘
```

Both sides are flattened into ONE tidy schema (`recon/schema.py`), one row per
`(trade_id, scenario_id, metric)`, `metric ∈ {base_pv, stressed_pv, impact}`,
`impact = stressed_pv − base_pv`. Everything downstream is source-agnostic.

## The capture layer mirrors your engine

Your framework: a **base class** that PVs cashflows off discount factors,
stresses the market, and re-PVs; **product classes** that filter the flat
cashflow table to their deal and PV/stress it. `recon/engine_adapter.py` mirrors
that so wiring is a call into your real classes:

| your engine | capture layer |
|-------------|---------------|
| base class: `revalue()` / stress / `revalue()` | `ProductAdapter.collect()` — base→stress→revalue loop + base/stressed/impact bookkeeping |
| base class stress method | `ProductAdapter.apply_stress()` — **seam 2** |
| product class (filters cashflows to its deal) | `Fx*Adapter.build_engine()` — **seam 1** |
| cashflows / DF-curve input files | `data/inhouse/*.csv` + `config/market.yaml` |

`recon/pv_engine.py` is a **reference** DCF engine (`PV = Σ CF·DF·fx0`; stress =
shift DF curve in bp and/or scale FX) so the pack runs end-to-end today. Swap
`build_engine` to point at your real product class; keep the `revalue()` +
`stress()` contract. The loop **rebuilds per scenario** so shocks hit fresh base
market data — never cumulatively.

## Two things that make this a real VC recon

1. **Reconcile the *impact* (stressed − base) separately from the base PV.** Base
   PV = "is our mark right?"; impact = "is our *sensitivity* right?" — what a risk
   engine is validated on.
2. **Align scenario conventions exactly** — the #1 false-break source.
   `config/scenarios.yaml` pins the shock (FX spot mult / DF bp shift) and its BBG
   equivalent, with the convention gotcha (FX quote direction, additive bp).

Tolerances per `(trade_type, metric)` in `config/tolerances.yaml`, in bp of
notional. Breaks classify PASS / WARN / FAIL / **N/A** (present in only one
source — a coverage break, never a silent pass).

## Layout

```
bbg_reconciliation/
├── recon/
│   ├── schema.py         canonical table + validation (the contract)
│   ├── pv_engine.py      reference DCF engine (swap for your real class)
│   ├── engine_adapter.py ▶ SEAMS: base + product mirror (FX fwd/swap)
│   ├── adapters.py       load the frozen snapshots
│   ├── reconcile.py      merge → diff (abs/rel/bp) → classify
│   ├── report.py         emit the PACK (detail csv + summary csv + html)
│   └── config.py         load YAML
├── config/
│   ├── market.yaml      ▶ base market: cashflow/DF files, fx0, as_of
│   ├── trades.yaml      ▶ sample deals (trade_id == deal_no)
│   ├── scenarios.yaml   ▶ stresses + in-house↔BBG convention map
│   └── tolerances.yaml  ▶ per trade_type × metric thresholds
├── tests/               test_fx.py + test_pack_summary.py (gate)
├── data/{inhouse,bbg,output}/
├── capture_inhouse.py   run engine ONCE → freeze in-house snapshot
├── run.py               generate the pack outside pytest
└── pytest.ini
```

## Run it

```bash
# from bbg_reconciliation/ using the repo venv
../../.venv/bin/python capture_inhouse.py   # input files → data/inhouse snapshot
../../.venv/bin/python -m pytest -q          # pass/fail gate, per (trade,scenario,metric)
../../.venv/bin/python run.py                # → data/output/recon_pack.html
```

Ships with demo `DEMO_FX*` deals + input files so it runs green immediately.

## Wiring your real data

1. **In-house** — drop your real `cashflows.csv` (`deal_no, instrument, ccy,
   cashflow_date, amount`) and `discount_curves.csv` (`ccy, date, df`) into
   `data/inhouse/`, list deals in `config/trades.yaml`, set fx0/as_of in
   `config/market.yaml`. To use your production pricer instead of the reference
   engine, point `Fx*Adapter.build_engine` at it. Then `capture_inhouse.py`.
2. **Bloomberg** — export base + each scenario PV per deal, remap scenario names
   to the canonical `scenario_id`, write `data/bbg/bbg_canonical.csv`.

## Presenting to Model Validation

The deliverable is `data/output/recon_pack.html`, front-loaded for a reviewer:
a **RAG banner** (overall PASS/WARN/FAIL) + KPI counts + **pass-rate matrix
(trade type × metric)**; then a **breaks-only** table (inhouse, bbg, diff in ccy
**and bp of notional**, tolerance applied); then full detail as audit trail
(`recon_detail.csv`); with a reproducibility stamp (val date, benchmark, scope,
model-risk note). The green pytest gate is the automatable evidence; the HTML
pack is the exhibit attached to the validation ticket.
