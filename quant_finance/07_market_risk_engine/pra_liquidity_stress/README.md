# PRA Liquidity Stress — end-to-end template population

A self-contained worked example of the kind of engine used to populate the PRA's
insurer **liquidity cashflow-mismatch templates**, base and stressed:

| Template | Level | What it is |
|---|---|---|
| **LQR.01.01** | per compartment (each MAP + the "remaining part") | detailed cashflow-mismatch ladder |
| **LQR.02.01** | per compartment | short (~condensed) mismatch — daily-capable in stress |
| **LQ.01.01** | solo / legal entity | roll-up of the compartments |

Regime context: PRA **PS15/25** "Closing liquidity reporting gaps", live **30 Sep 2026**.
See `../../../../job-search/interview/just_PRA_liquidity_reporting_MAP_LQR.md` for the
terminology and template background this drill implements.

> ⚠️ **Synthetic data.** All inputs are fabricated by `generate_inputs.py`. Numbers are
> illustrative, not a real firm's positions. The point is the *pipeline shape* and the
> *mechanics*, which stay identical when you swap in real CSV extracts.

---

## The mental model

```
        discount_factors.csv          deals.csv            cashflows.csv
        (720m × 4 curves)        (static, per instrument)   (nominal cash)
               │                        │                        │
               └──────────┬─────────────┴───────────┬────────────┘
                          ▼                          ▼
                 shock the curve            bucket nominal cash
                 → reprice swaps/gilts       onto the maturity ladder
                 → variation margin (VM)         (BASE position)
                 → haircut top-ups
                 → CBC erosion
                          │                          │
                          └────────────┬─────────────┘
                                       ▼
                      LQR.01.01 (per compartment)  ── sum ──▶ LQ.01.01 (solo)
                      LQR.02.01 (short)
                                       │
                                       ▼
                          deep-dive analysis
             survival horizon · driver attribution · fungibility check
```

Two things happen to produce a **stressed** template:

1. **Base ladder** — project each deal's *nominal* cash and drop it into a maturity
   bucket. That is the position "as reported".
2. **Stress overlay** — shock the discount curve and reprice instruments. The *change*
   in value that must be settled as cash — **derivative variation margin** and **repo
   haircut top-ups** — lands in the immediate (`<=1m`) bucket, and the **counterbalancing
   capacity** (cash + repo-able gilts) is eroded by the same shock.

**Survival** at each bucket = cumulative net mismatch + opening counterbalancing capacity.
Negative ⇒ the compartment can't fund its cumulative outflows from its own liquid resources.

---

## The calculations (every formula in the engine)

All figures are in GBP; foreign legs are multiplied by the (shocked) FX rate. `t = month/12`.

**1. Discount curve** — **annual (discrete) compounding** throughout (Just's convention,
not continuous). Spot rate implied from a DF, and DF rebuilt from a shocked spot:

```
z(t)          = DF(t)^(−1/t) − 1                      # DF → annually-compounded spot
DF_shocked(t) = (1 + z(t) + s/10000)^(−t)             # spot → DF after an s-bp shock
```
(Contrast the continuous convention `z = −ln DF / t`, `DF = e^(−z·t)` — *not* used here.)

**2. Receiver swap MTM** (receive-fixed / pay-float, single curve, from DFs only):

```
fixed_pv = Σ_i  N · k · τ · DF(t_i)                  # k = par fixed rate, τ = freq/12
float_pv = N · (1 − DF(T))                            # spot-start float leg
MTM      = fixed_pv − float_pv
```
`k` is set to the base-curve **par rate** so base MTM ≈ 0. The stress cash is the
**variation margin** = the *change* in MTM under the shocked curve:

```
VM = MTM(shocked) − MTM(base)     →  <=1m bucket   (VM<0 = post cash / outflow O2)
```
Rates up ⇒ a receiver swap loses value ⇒ posts VM ⇒ day-1 outflow. This is the LDI margin-call channel.

**3. Inflation swap VM** (receive-inflation zero-coupon swap, repriced on the shocked
**nominal** curve — so it responds to **both** rate and inflation shocks):

```
MTM = N · DF_nom(T) · [ (1+π)^T − (1+b)^T ]
   b = contracted breakeven (deal.rate),  π = b + Δinfl/10000
VM  = MTM(rate_bp, Δinfl) − MTM(0, 0)     →  <=1m bucket
```
The rate shock enters through `DF_nom(T)`; the inflation shock through `π`. A deflation
shock (`Δinfl < 0`) is a loss to the receiver ⇒ outflow. Note: at par (`π=b`) a *pure*
rate move gives ≈0 — correct, a par inflation swap has ~no standalone rate DV01; the rate
effect shows up by discounting the inflation-driven payoff (i.e. in combined scenarios).

**3b. FX forward** (SELL N foreign forward at strike K, maturity T) — GBP MTM:

```
MTM = N·K·DF_dom(T) − N·S·DF_for(T)          # receive GBP strike, deliver foreign
```
`S` = spot (GBP per foreign); `K` = par forward `S·DF_for(T)/DF_dom(T)` so base MTM ≈ 0.
Two liquidity items: **(i)** stress VM = ΔMTM (GBP depreciation ⇒ a sold-foreign forward
loses ⇒ posts VM ⇒ `<=1m` outflow); **(ii)** the **gross settlement** at T — receive `K·N`
GBP *and* deliver `N` foreign — both legs hit the ladder (settlement/Herstatt risk), even
though they nearly net.

**3c. Cross-currency (XCCY) swap** (pay foreign / receive GBP) — GBP MTM:

```
MTM = PV_gbp − S · PV_for
   PV_gbp = Σ c_gbp·N_gbp·τ·DF_dom(t_i) + N_gbp·DF_dom(T)     # receive-GBP leg
   PV_for = Σ c_for·N_for·τ·DF_for(t_i) + N_for·DF_for(T)     # pay-foreign leg
   N_gbp  = N_for · S₀   (notionals matched at inception spot)
```
Liquidity items: **(i)** stress VM = ΔMTM, dominated by the **principal re-exchange** and
so highly FX-sensitive (GBP depreciation ⇒ the pay-foreign leg costs more ⇒ VM outflow);
**(ii)** periodic coupons in each currency *and* the **gross principal re-exchange at T**
all flow through the ladder in their own currency.

**4. Bond / gilt value** (own cashflows discounted on shocked curve + spread):

```
V = Σ_j  CF_j · DF_shocked(t_j ; rate_bp + spread_bp)
    spread = credit_bp for corp bonds,  gilt_bp for gilts
```

**5. Counterbalancing capacity (CBC)** — the liquid buffer, a *stock*:

```
CBC = Σ cash  +  Σ gilt_value · (1 − haircut) ,   haircut = base + stress_addon
```
Under stress CBC shrinks two ways: gilt `V` falls (rate+gilt shock) and the haircut widens.

**6. Repo haircut top-up** — lender raises the haircut ⇒ post extra collateral now:

```
top_up = notional · repo_haircut_addon           →  <=1m bucket (outflow O3)
```

**7. The ladder** — per compartment, per bucket:

```
net_b  = Σ inflows_b + Σ outflows_b              # outflows carried negative
cum_b  = Σ_{k≤b} net_k
surv_b = cum_b + CBC                             # the pass/fail line: surv_b ≥ 0 ?
```

**8. Roll-up (LQ.01.01)** = sum the compartment ladders, then **recompute** `cum` and
`surv` from the summed flows (never sum `cum`/`surv` twice). The naive sum treats cash as
fully fungible — which is precisely the assumption the fungibility check flags as false.

---

## Why the compartment split is the whole point

Cash inside a **Matching Adjustment Portfolio (MAP)** is *not* freely fungible with the rest
of the firm. So the engine reports **each MAP and the remaining part separately (LQR)** and
also **rolls them up (LQ)**. The headline result of this drill is exactly that tension:

```
SOLO 1-year survival  : +£534m   (PASS)
  MAP2_BULK           : −£116m   (BREACH on day 1)   ← trapped: SOLO surplus can't rescue it
```

A firm can look liquid in aggregate while a single MAP is under water. That is the risk the
regime is built to surface.

---

## Files — each one's job

| File | Job |
|---|---|
| `generate_inputs.py` | Fabricate the CSV inputs (curves, FX, deals, cashflows, scenarios, blank templates). **Swap this for real extracts** and nothing downstream changes. |
| `engine.py` | `LiquidityStressEngine`: curve shocking, instrument repricing, counterbalancing, ladder construction, and the three template builders. All closed-form off the supplied DFs. |
| `run.py` | Orchestrates inputs → populated templates → deep-dive analysis. The entry point. |
| `data/inputs/` | Generated inputs (incl. the blank template pack). |
| `data/outputs/` | Populated templates + analysis. |

---

## Run it

```bash
cd pra_liquidity_stress
uv run python run.py          # generates inputs on first run, then populates + analyses
```

Outputs land in `data/outputs/`:

- `template_LQR_01_01_populated.csv` — detailed, per compartment × scenario
- `template_LQR_02_01_populated.csv` — short, per compartment × scenario
- `template_LQ_01_01_populated.csv` — solo roll-up × scenario
- `analysis_deep_dive.md` — the narrative read (survival, drivers, fungibility)
- `analysis_scenario_summary.csv`, `analysis_cbc_erosion.csv` — the underlying tables

Regenerate inputs from scratch (e.g. after editing `generate_inputs.py`):

```bash
rm -f data/inputs/*.csv && uv run python run.py
```

---

## Input data dictionary

| CSV | Columns | Notes |
|---|---|---|
| `discount_factors.csv` | `month` (1..720), `GBP`,`USD`,`EUR`,`JPY` | continuously-compounded DFs from a Nelson-Siegel zero curve per ccy |
| `fx_rates.csv` | `currency`, `gbp_per_unit` | base spot; FX shock scales the foreign legs |
| `deals.csv` | `deal_id`,`compartment`,`currency`,`product_type`,`notional`,`rate`,`start_month`,`maturity_month`,`freq_months`,`haircut`,`rate2` | static data for revaluation; `compartment` drives LQR vs LQ. `rate2` = GBP-leg coupon (XCCY) or strike (FX forward) |
| `cashflows.csv` | `deal_id`,`month`,`amount` (+in/−out),`currency`,`flow_category` | projected nominal cash → the base ladder. **Per-flow `currency`** because FX/XCCY legs settle in two currencies |
| `stress_scenarios.csv` | `scenario_id`, per-factor shocks, `description` | rate/credit/gilt/FX/inflation + haircut add-ons |
| `template_*_blank.csv` | line codes + empty buckets | the PRA "blank pack" the engine fills |

---

## Scenarios

| id | shock |
|---|---|
| `BASE` | none |
| `RATES_UP_100` | +100bp rates (LDI-style margin call) |
| `RATES_DOWN_100` | −100bp rates |
| `CREDIT_WIDEN` | credit +150 / gilt +50, haircuts widen |
| `INFLATION_DOWN` | −75bp inflation (inflation-swap VM) |
| `COMBINED_PRA` | headline: rates +100, credit +150, gilt +50, GBP −10%, infl −50, haircuts widen |

---

## Assumptions to replace with real methodology

The drill is honest about its simplifications (also listed in `analysis_deep_dive.md`):

- **Ladder buckets** — 11 buckets to 60y here; confirm the PRA template's exact buckets,
  especially the short end (daily / weekly) the short template cares about.
- **Template taxonomy & code mapping** — line items and the exact LQ/LQR codes are
  reconstructed; confirm against the real pack.
- **Annuity run-off** — linear-decline proxy, not actuarial mortality.
- **Swaps** — single-curve valuation (no OIS/tenor basis). Inflation swap uses a flat
  breakeven projection (no seasonality / inflation term structure), repriced on the
  nominal curve.
- **FX forwards / XCCY swaps** — priced off covered interest parity with a flat cross-currency
  basis of zero; VM assumed fully cash-collateralised daily (real CSAs have thresholds, MTAs,
  and non-cash collateral). Gross settlement legs are shown in full (no PvP netting assumed).
- **Gilts** — treated purely as counterbalancing (monetise-now), so their redemptions are
  *not* also added to the inflow ladder (avoids double counting).

These are exactly the inputs you'd harden first when wiring the engine to production data.
