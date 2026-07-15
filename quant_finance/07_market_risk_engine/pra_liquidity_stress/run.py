"""
run.py  —  End-to-end: CSV inputs  ->  populated templates + deep-dive analysis.

    uv run python run.py

Steps
  1. (re)generate the synthetic inputs if they are missing
  2. populate LQ.01.01 / LQR.01.01 / LQR.02.01 for BASE + all stress scenarios
  3. write the populated templates to data/outputs/
  4. build the deep-dive analysis: per-compartment survival, driver attribution,
     and the fungibility check (does a healthy SOLO number hide a trapped MAP?)
"""
import numpy as np
import pandas as pd
from pathlib import Path

import generate_inputs
from engine import LiquidityStressEngine, BUCKET_LABELS, IMMEDIATE

HERE = Path(__file__).parent
INPUTS = HERE / "data" / "inputs"
OUTPUTS = HERE / "data" / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)

GBP_M = 1e6                       # report figures in £m
LIQ_HORIZON = 4                   # buckets <=1m..6-12m == the 1-year liquidity horizon


def ensure_inputs():
    if not (INPUTS / "discount_factors.csv").exists():
        print("Inputs missing — generating synthetic inputs...")
        generate_inputs.main()


def survival_bucket(surv_row: np.ndarray) -> str:
    """First ladder bucket where cumulative-mismatch + counterbalancing goes < 0."""
    neg = np.where(surv_row < 0)[0]
    return BUCKET_LABELS[neg[0]] if len(neg) else "covered (>60y)"


def build_analysis(engine, lqr01, lq01):
    """Return (scenario_summary_df, markdown_report)."""
    rows = []
    # per-compartment + SOLO, per-scenario headline metrics
    frames = {"detail": lqr01, "solo": lq01}
    for label, frame in frames.items():
        for (comp, scen), g in frame.groupby(["compartment", "scenario"], sort=False):
            gi = g.set_index("line_code")[BUCKET_LABELS]
            surv = gi.loc["SURV"].values
            cbc = gi.loc["CBC"].values.sum()
            vm_out = gi.loc["O2"].values[IMMEDIATE] if "O2" in gi.index else 0.0
            repo_out = gi.loc["O3"].values[IMMEDIATE] if "O3" in gi.index else 0.0
            rows.append({
                "compartment": comp, "scenario": scen,
                "opening_CBC_£m": cbc / GBP_M,
                # headline is the 1-year liquidity horizon; the full 60y terminal is
                # an ALM/matching read-out, not a near-term liquidity signal
                "min_surv_1y_£m": surv[:LIQ_HORIZON].min() / GBP_M,
                "min_surv_full_£m": surv.min() / GBP_M,
                "survival_bucket": survival_bucket(surv),
                "immediate_VM_out_£m": vm_out / GBP_M,
                "immediate_repo_topup_incl_£m": repo_out / GBP_M,
            })
    summary = pd.DataFrame(rows)

    # CBC erosion under the headline combined scenario vs base
    cbc_erosion = []
    for comp in engine.compartments + ["SOLO"]:
        if comp == "SOLO":
            base = lq01[(lq01.scenario == "BASE")].set_index("line_code").loc["CBC", BUCKET_LABELS].sum()
            strs = lq01[(lq01.scenario == "COMBINED_PRA")].set_index("line_code").loc["CBC", BUCKET_LABELS].sum()
        else:
            b = engine.counterbalancing(comp, next(s for s in engine.scenarios if s.is_base))
            s = engine.counterbalancing(comp, next(s for s in engine.scenarios if s.scenario_id == "COMBINED_PRA"))
            base, strs = b["total"], s["total"]
        cbc_erosion.append({"compartment": comp, "CBC_base_£m": base / GBP_M,
                            "CBC_stressed_£m": strs / GBP_M,
                            "erosion_£m": (base - strs) / GBP_M})
    erosion = pd.DataFrame(cbc_erosion)

    # ---- fungibility check on the headline scenario -----------------------
    combined = summary[summary.scenario == "COMBINED_PRA"]
    solo_ok = combined[combined.compartment == "SOLO"]["min_surv_1y_£m"].iloc[0] >= 0
    breached = combined[(combined.compartment != "SOLO") &
                        (combined["min_surv_1y_£m"] < 0)]["compartment"].tolist()

    # ---- markdown report --------------------------------------------------
    md = []
    md.append("# PRA Liquidity Stress — Deep-Dive Analysis\n")
    md.append("Cashflow-mismatch templates populated base + stressed. Figures in £m. "
              "Survival = cumulative net mismatch + opening counterbalancing capacity (CBC); "
              "negative = the compartment cannot fund its cumulative outflows from its own "
              "liquid resources by that bucket. **Headline uses the 1-year liquidity horizon** "
              "(buckets <=1m..6-12m); the full 60y terminal is an ALM/matching read-out, not a "
              "near-term liquidity signal.\n")

    md.append("## 1. Headline: fungibility check (COMBINED_PRA, 1-year horizon)\n")
    md.append(f"- SOLO (LQ.01.01) minimum 1y survival buffer: "
              f"**£{combined[combined.compartment=='SOLO']['min_surv_1y_£m'].iloc[0]:,.0f}m** "
              f"({'PASS' if solo_ok else 'BREACH'}).")
    if solo_ok and breached:
        md.append(f"- ⚠️ **Fungibility trap:** SOLO looks healthy, but compartment(s) "
                  f"**{', '.join(breached)}** breach on a standalone basis. Cash surplus "
                  f"in NON_MAP/other compartments is *not* freely available to a MAP — "
                  f"this is exactly why the PRA requires the per-compartment LQR return.")
    elif breached:
        md.append(f"- ⚠️ SOLO also breaches; compartment breaches: {', '.join(breached)}.")
    else:
        md.append("- No compartment breaches on the headline scenario.")
    md.append("")

    md.append("## 2. Per-compartment / SOLO 1-year survival by scenario\n")
    piv = summary.pivot_table(index=["compartment"], columns="scenario",
                              values="min_surv_1y_£m").round(0)
    # order columns base-first
    order = [s.scenario_id for s in engine.scenarios if s.scenario_id in piv.columns]
    md.append(piv[order].to_markdown())
    md.append("\n_Minimum survival buffer (£m) within the 1-year liquidity horizon. "
              "Negative = breach._\n")

    md.append("## 3. Immediate (<=1m) stress cash drivers — COMBINED_PRA\n")
    drv = combined[["compartment", "immediate_VM_out_£m",
                    "immediate_repo_topup_incl_£m", "opening_CBC_£m"]].round(0)
    md.append(drv.to_markdown(index=False))
    md.append("\n_Derivative variation margin dominates the day-1 outflow for the MAPs; "
              "gilts + cash are the buffer against it._\n")

    md.append("## 4. Counterbalancing capacity erosion (base -> COMBINED_PRA)\n")
    md.append(erosion.round(0).to_markdown(index=False))
    md.append("\n_CBC falls because gilt market values drop (rate + gilt-spread shock) "
              "and repo haircuts widen — the buffer shrinks exactly when outflows spike._\n")

    md.append("## 5. Assumptions to confirm against the real spec\n")
    md.append("- Ladder bucket boundaries (here 11 buckets to 60y) — confirm the PRA "
              "template's exact buckets, especially the short end (daily/weekly).")
    md.append("- Template line-item taxonomy and the exact LQ/LQR code mapping.")
    md.append("- Annuity run-off is a linear-decline proxy (real: actuarial mortality).")
    md.append("- Single-curve swap valuation; inflation swap via a duration approximation.")
    md.append("- Gilts treated purely as counterbalancing (monetise-now) rather than "
              "adding their redemptions to the inflow ladder — avoids double counting.")
    md.append("")
    return summary, erosion, "\n".join(md)


def main():
    ensure_inputs()
    engine = LiquidityStressEngine(INPUTS)

    print(f"Compartments: {engine.compartments}")
    print(f"Scenarios:    {[s.scenario_id for s in engine.scenarios]}\n")

    # 2-3. populate + write the three templates
    lqr01 = engine.populate_LQR_01_01()
    lqr02 = engine.populate_LQR_02_01()
    lq01 = engine.populate_LQ_01_01()

    lqr01.to_csv(OUTPUTS / "template_LQR_01_01_populated.csv", index=False)
    lqr02.to_csv(OUTPUTS / "template_LQR_02_01_populated.csv", index=False)
    lq01.to_csv(OUTPUTS / "template_LQ_01_01_populated.csv", index=False)

    # 4. deep-dive analysis
    summary, erosion, md = build_analysis(engine, lqr01, lq01)
    summary.round(2).to_csv(OUTPUTS / "analysis_scenario_summary.csv", index=False)
    erosion.round(2).to_csv(OUTPUTS / "analysis_cbc_erosion.csv", index=False)
    (OUTPUTS / "analysis_deep_dive.md").write_text(md)

    print("Outputs written to", OUTPUTS)
    for p in sorted(OUTPUTS.glob("*")):
        print(f"  {p.name}")

    # quick console read-out of the headline
    print("\n--- Headline (COMBINED_PRA) 1-year survival buffer £m ---")
    hl = summary[summary.scenario == "COMBINED_PRA"][
        ["compartment", "min_surv_1y_£m", "min_surv_full_£m", "survival_bucket"]]
    print(hl.to_string(index=False))


if __name__ == "__main__":
    main()
