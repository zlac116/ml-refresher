"""
generate_inputs.py  —  Synthesise the CSV INPUTS for the liquidity stress engine.

In production these CSVs come from the firm's data warehouse (curve service,
policy admin / ALM cashflow projection, derivative sub-ledger, collateral system,
and the PRA's blank template pack). Here we fabricate *plausible* versions so the
whole pipeline runs end-to-end with nothing external. Swap this file for your real
extracts and `engine.py` / `run.py` are unchanged.

Produces, in data/inputs/:
  discount_factors.csv        720 monthly DFs per core currency (GBP/USD/EUR/JPY)
  fx_rates.csv                GBP per 1 unit of each currency (base spot)
  deals.csv                   one row per deal: static data for revaluation
  cashflows.csv               projected NOMINAL cash per deal per month (the ladder)
  stress_scenarios.csv        the PRA-style shock scenarios
  template_LQ_01_01_blank.csv     blank solo template   (row labels, empty buckets)
  template_LQR_01_01_blank.csv    blank detailed per-compartment template
  template_LQR_02_01_blank.csv    blank short (~150 data point) template

Design choices you should sanity-check against the real spec are called out with
[ASSUMPTION].
"""
import numpy as np
import pandas as pd
from pathlib import Path

INPUTS = Path(__file__).parent / "data" / "inputs"
INPUTS.mkdir(parents=True, exist_ok=True)

N_MONTHS = 720                       # 60 years of monthly discount factors
CCYS = ["GBP", "USD", "EUR", "JPY"]  # core currencies


# --------------------------------------------------------------------------- #
# 1. DISCOUNT FACTORS  — a Nelson-Siegel zero curve per currency, then DF=e^-zt #
# --------------------------------------------------------------------------- #
# Nelson-Siegel: z(t) = b0 + b1*(1-e^-t/τ)/(t/τ) + b2*((1-e^-t/τ)/(t/τ) - e^-t/τ)
# b0 ~ long rate, b1 ~ short-minus-long slope, b2 ~ curvature/hump.
NS_PARAMS = {                    # (b0,      b1,      b2,     tau)
    "GBP": (0.0450, -0.0050,  0.0100, 2.5),
    "USD": (0.0470, -0.0030,  0.0080, 2.0),
    "EUR": (0.0300, -0.0080,  0.0060, 2.5),
    "JPY": (0.0120, -0.0060,  0.0040, 3.0),
}


def nelson_siegel(t_years, b0, b1, b2, tau):
    """Zero (spot) rate at t years — interpreted as ANNUALLY compounded."""
    x = t_years / tau
    # limit of (1-e^-x)/x as x->0 is 1; guard the t=0 point.
    loading = np.where(x == 0, 1.0, (1 - np.exp(-x)) / np.where(x == 0, 1.0, x))
    return b0 + b1 * loading + b2 * (loading - np.exp(-x))


def build_discount_factors():
    months = np.arange(1, N_MONTHS + 1)
    t = months / 12.0
    df = pd.DataFrame({"month": months})
    for ccy in CCYS:
        z = nelson_siegel(t, *NS_PARAMS[ccy])
        df[ccy] = (1.0 + z) ** (-t)       # ANNUAL (discrete) compounding: DF = (1+z)^-t
    df.to_csv(INPUTS / "discount_factors.csv", index=False)
    return df


# --------------------------------------------------------------------------- #
# 2. FX  — GBP per 1 unit of foreign currency (base spot)                      #
# --------------------------------------------------------------------------- #
def build_fx():
    fx = pd.DataFrame(
        {"currency": CCYS, "gbp_per_unit": [1.0, 0.79, 0.855, 0.0053]}
    )
    fx.to_csv(INPUTS / "fx_rates.csv", index=False)
    return fx


# --------------------------------------------------------------------------- #
# 3. DEALS  — static data. compartment is the KEY dimension for LQR vs LQ.     #
# --------------------------------------------------------------------------- #
# Compartments:
#   MAP1_RETIREMENT  – matching-adjustment portfolio (individual annuities)
#   MAP2_BULK        – matching-adjustment portfolio (DB bulk annuities)
#   NON_MAP          – the "remaining part": shareholder / surplus / non-MA
#
# product_type semantics:
#   ANNUITY_LIABILITY  monthly outflow (policy payments)          -> ladder
#   CORP_BOND          coupon+principal inflow                    -> ladder + asset value
#   GILT               liquid buffer (counterbalancing capacity)  -> CBC only, revalued
#   CASH               liquid buffer                              -> CBC only
#   RECEIVER_SWAP      receive-fixed/pay-float rate hedge         -> stress VM only
#   INFLATION_SWAP     receive-inflation hedge                    -> stress VM only
#   REPO               secured borrowing                          -> ladder + haircut top-up
DEALS = [
    # deal_id, compartment, ccy, product_type, notional, coupon/fixed_rate, start_m, maturity_m, freq_m, haircut
    ("D01", "MAP1_RETIREMENT", "GBP", "ANNUITY_LIABILITY", 5.0e9, 0.0,    1, 480,  1, 0.0),
    # asset nominal ~matches the annuity so the long end nets out; timing gaps and
    # the day-1 stress VM are what drive the liquidity signal, not structural funding.
    ("D02", "MAP1_RETIREMENT", "GBP", "CORP_BOND",         3.5e9, 0.040,  1, 480,  6, 0.0),
    ("D03", "MAP1_RETIREMENT", "USD", "CORP_BOND",         1.5e9, 0.045,  1, 300,  6, 0.0),
    ("D04", "MAP1_RETIREMENT", "GBP", "GILT",              0.60e9, 0.035, 1, 120,  6, 0.03),
    ("D05", "MAP1_RETIREMENT", "GBP", "RECEIVER_SWAP",     3.0e9, None,    0, 300,  6, 0.0),
    ("D06", "MAP1_RETIREMENT", "GBP", "REPO",              0.80e9, 0.045, 1,   3,  0, 0.02),
    ("D07", "MAP1_RETIREMENT", "GBP", "CASH",              0.15e9, 0.0,    0,   0,  0, 0.0),

    ("D08", "MAP2_BULK",       "GBP", "ANNUITY_LIABILITY", 4.0e9, 0.0,    1, 420,  1, 0.0),
    ("D09", "MAP2_BULK",       "EUR", "CORP_BOND",         1.2e9, 0.040,  1, 216,  6, 0.0),
    ("D18", "MAP2_BULK",       "GBP", "CORP_BOND",         2.8e9, 0.042,  1, 420,  6, 0.0),
    ("D10", "MAP2_BULK",       "GBP", "GILT",              0.40e9, 0.035, 1,  60,  6, 0.025),
    ("D11", "MAP2_BULK",       "GBP", "RECEIVER_SWAP",     2.5e9, None,    0, 240,  6, 0.0),
    ("D12", "MAP2_BULK",       "GBP", "INFLATION_SWAP",    1.5e9, 0.030,   0, 360,  6, 0.0),
    ("D13", "MAP2_BULK",       "GBP", "CASH",              0.10e9, 0.0,    0,   0,  0, 0.0),

    ("D14", "NON_MAP",         "GBP", "CORP_BOND",         0.50e9, 0.045,  1,  60,  6, 0.0),
    ("D15", "NON_MAP",         "JPY", "RECEIVER_SWAP",     5.0e10, None,    0, 120,  6, 0.0),
    ("D16", "NON_MAP",         "GBP", "CASH",              0.60e9, 0.0,    0,   0,  0, 0.0),
    ("D17", "NON_MAP",         "GBP", "REPO",              0.30e9, 0.045, 1,   1,  0, 0.02),

    # --- FX forwards & cross-currency swaps -------------------------------- #
    # XCCY hedges the USD bond D03 back to GBP (pay USD coupons+principal /
    # receive GBP). FX forwards hedge the FX on foreign asset holdings by
    # SELLING the foreign currency forward. `rate` = foreign leg coupon (XCCY);
    # `rate2` (added in build_deals) = GBP leg coupon (XCCY) or strike (FX fwd).
    ("D19", "MAP1_RETIREMENT", "USD", "XCCY_SWAP",         1.5e9, 0.045,  1, 300,  6, 0.0),
    ("D20", "MAP2_BULK",       "EUR", "FX_FORWARD",        1.2e9, 0.0,     1,   6,  0, 0.0),
    ("D21", "NON_MAP",         "USD", "FX_FORWARD",        0.5e9, 0.0,     1,   3,  0, 0.0),
]

DEAL_COLS = ["deal_id", "compartment", "currency", "product_type", "notional",
             "rate", "start_month", "maturity_month", "freq_months", "haircut"]


def par_swap_rate(df_ccy, maturity_m, freq_m):
    """Par fixed rate so the receiver swap has ~zero MTM on the base curve.
    Using single-curve DFs: k* = (1 - DF_T) / sum(tau * DF_ti)."""
    pay_months = np.arange(freq_m, maturity_m + 1, freq_m)
    tau = freq_m / 12.0
    dfs = df_ccy.set_index("month").reindex(pay_months)["_df"].values
    annuity = (tau * dfs).sum()
    df_T = df_ccy.set_index("month").loc[maturity_m, "_df"]
    return (1.0 - df_T) / annuity


def build_deals(dfs, fx):
    rows = []
    for d in DEALS:
        row = dict(zip(DEAL_COLS, d))
        row["rate2"] = np.nan                     # 2nd rate: GBP leg (XCCY) / strike (FX fwd)
        pt = row["product_type"]
        T, freq, ccy = int(row["maturity_month"]), int(row["freq_months"] or 0), row["currency"]

        if pt == "RECEIVER_SWAP":
            # par fixed rate -> BASE MTM ~ 0, so stress VM is a clean read of the shock
            cdf = dfs[["month", ccy]].rename(columns={ccy: "_df"})
            row["rate"] = round(float(par_swap_rate(cdf, T, freq)), 6)

        elif pt == "FX_FORWARD":
            # par forward strike (GBP per foreign) via covered interest parity:
            #   K = S * DF_for(T) / DF_dom(T)   -> BASE MTM ~ 0
            s = float(fx.set_index("currency").loc[ccy, "gbp_per_unit"])
            k = s * float(dfs.loc[dfs.month == T, ccy].iloc[0]) / float(dfs.loc[dfs.month == T, "GBP"].iloc[0])
            row["rate2"] = round(k, 6)

        elif pt == "XCCY_SWAP":
            # GBP leg coupon = GBP par swap rate for the maturity (foreign leg
            # coupon is the stored `rate`). Notionals matched at inception spot.
            gdf = dfs[["month", "GBP"]].rename(columns={"GBP": "_df"})
            row["rate2"] = round(float(par_swap_rate(gdf, T, freq)), 6)

        rows.append(row)
    out = pd.DataFrame(rows, columns=DEAL_COLS + ["rate2"])
    out.to_csv(INPUTS / "deals.csv", index=False)
    return out


# --------------------------------------------------------------------------- #
# 4. CASHFLOWS  — projected NOMINAL cash per deal per month.                   #
#    Sign convention: +inflow, -outflow.  These populate the BASE ladder.      #
#    Swaps/gilts/cash produce NO base ladder cash (their liquidity risk is     #
#    stress VM / is a counterbalancing stock) — they live only in deals.csv.   #
# --------------------------------------------------------------------------- #
def expand_cashflows(deals, fx):
    # each cashflow carries its OWN currency, because FX forwards and XCCY swaps
    # settle two legs in two different currencies within a single deal.
    spot = fx.set_index("currency")["gbp_per_unit"]
    rows = []
    for _, d in deals.iterrows():
        pt, n, ccy = d["product_type"], d["notional"], d["currency"]
        if pt == "ANNUITY_LIABILITY":
            # [ASSUMPTION] linearly-declining monthly payment to zero at maturity
            # (crude run-off proxy for mortality). p0 = 2N/T so payments sum to N.
            T = int(d["maturity_month"])
            p0 = 2.0 * n / T
            for m in range(1, T + 1):
                amt = -p0 * (1 - (m - 1) / T)
                rows.append((d["deal_id"], m, round(amt, 2), ccy, "INSURANCE_OUT"))
        elif pt == "CORP_BOND":
            cpn = n * d["rate"] * (d["freq_months"] / 12.0)
            mat = int(d["maturity_month"])
            for m in range(int(d["freq_months"]), mat + 1, int(d["freq_months"])):
                amt = cpn + (n if m == mat else 0.0)         # coupon; +principal at maturity
                rows.append((d["deal_id"], m, round(amt, 2), ccy, "ASSET_IN"))
        elif pt == "REPO":
            # cash raised now, repaid (with interest) at maturity
            mat = int(d["maturity_month"])
            rows.append((d["deal_id"], int(d["start_month"]), round(n, 2), ccy, "REPO_IN"))
            repay = n * (1 + d["rate"] * mat / 12.0)
            rows.append((d["deal_id"], mat, round(-repay, 2), ccy, "REPO_OUT"))
        elif pt == "FX_FORWARD":
            # SELL foreign forward at strike K: at T receive K*N in GBP, deliver N foreign.
            # Gross legs are a real liquidity (settlement) event even though they ~net.
            mat, K = int(d["maturity_month"]), d["rate2"]
            rows.append((d["deal_id"], mat, round(K * n, 2), "GBP", "FX_SETTLE_IN"))
            rows.append((d["deal_id"], mat, round(-n, 2), ccy, "FX_SETTLE_OUT"))
        elif pt == "XCCY_SWAP":
            # pay foreign / receive GBP. Notionals matched at inception spot.
            mat, freq = int(d["maturity_month"]), int(d["freq_months"])
            tau = freq / 12.0
            n_gbp = n * float(spot[ccy])                     # GBP notional (fixed at inception)
            for m in range(freq, mat + 1, freq):
                gbp_cf = d["rate2"] * n_gbp * tau + (n_gbp if m == mat else 0.0)
                for_cf = d["rate"] * n * tau + (n if m == mat else 0.0)
                rows.append((d["deal_id"], m, round(gbp_cf, 2), "GBP", "XCCY_IN"))
                rows.append((d["deal_id"], m, round(-for_cf, 2), ccy, "XCCY_OUT"))
        # GILT / CASH / RECEIVER_SWAP / INFLATION_SWAP -> no base ladder cashflow
    cf = pd.DataFrame(rows, columns=["deal_id", "month", "amount", "currency", "flow_category"])
    cf.to_csv(INPUTS / "cashflows.csv", index=False)
    return cf


# --------------------------------------------------------------------------- #
# 5. STRESS SCENARIOS  — the PRA-style single- and combined-factor shocks.     #
# --------------------------------------------------------------------------- #
# All shocks are instantaneous. Sign: rate_bp>0 = rates up; fx_gbp_depr_pct>0 =
# sterling weakens (foreign assets/VM worth more in GBP); infl_bp>0 = inflation up.
SCENARIOS = [
    # id, rate_bp, credit_bp, gilt_bp, fx_gbp_depr_pct, infl_bp, gilt_hc_addon, repo_hc_addon, description
    ("BASE",            0,    0,    0,  0.0,    0, 0.00, 0.00, "No shock — as-reported base position"),
    ("RATES_UP_100",  100,    0,   10,  0.0,    0, 0.01, 0.01, "Parallel +100bp rates (LDI-style margin call)"),
    ("RATES_DOWN_100",-100,   0,  -10,  0.0,    0, 0.00, 0.00, "Parallel -100bp rates"),
    ("CREDIT_WIDEN",    0,  150,   50,  0.0,    0, 0.03, 0.03, "Credit +150 / gilt +50, haircuts widen"),
    ("INFLATION_DOWN",  0,    0,    0,  0.0,  -75, 0.00, 0.00, "Deflation shock -75bp (inflation-swap VM)"),
    ("COMBINED_PRA",  100,  150,   50, 10.0,  -50, 0.04, 0.04,
     "Headline combined: rates+100, credit+150, gilt+50, GBP -10%, infl-50, haircuts widen"),
]
SCEN_COLS = ["scenario_id", "rate_bp", "credit_bp", "gilt_bp", "fx_gbp_depr_pct",
             "infl_bp", "gilt_haircut_addon", "repo_haircut_addon", "description"]


def build_scenarios():
    sc = pd.DataFrame(SCENARIOS, columns=SCEN_COLS)
    sc.to_csv(INPUTS / "stress_scenarios.csv", index=False)
    return sc


# --------------------------------------------------------------------------- #
# 6. BLANK TEMPLATES  — row labels + empty bucket columns (the PRA pack).      #
# --------------------------------------------------------------------------- #
BUCKET_LABELS = ["<=1m", "1-3m", "3-6m", "6-12m", "1-2y", "2-3y",
                 "3-5y", "5-10y", "10-20y", "20-30y", "30-60y"]

# Detailed template line items (LQ.01.01 and LQR.01.01 share this layout).
DETAILED_ROWS = [
    ("I1", "Inflow", "Insurance & reinsurance inflows"),
    ("I2", "Inflow", "Asset cashflows (bond coupons & redemptions)"),
    ("I4", "Inflow", "Derivative & FX receipts (VM + FX/XCCY settlement)"),
    ("I5", "Inflow", "Secured financing raised (repo cash-in)"),
    ("O1", "Outflow", "Annuity & claims payments"),
    ("O2", "Outflow", "Derivative & FX payments (VM + FX/XCCY settlement)"),
    ("O3", "Outflow", "Repo repayments & haircut top-ups"),
    ("N",  "Derived", "Net contractual mismatch (period)"),
    ("CUM", "Derived", "Cumulative net mismatch"),
    ("CBC", "Derived", "Counterbalancing capacity (opening stock)"),
    ("SURV", "Derived", "Cumulative mismatch + counterbalancing"),
]

# Short template (~150 data points): condensed lines x 11 buckets + headline block.
SHORT_ROWS = [
    ("S1", "Net insurance flow"),
    ("S2", "Net derivative & FX flow (VM + FX/XCCY settlement)"),
    ("S3", "Net secured-financing flow"),
    ("S4", "Counterbalancing capacity (opening stock)"),
    ("S5", "Net liquidity position (cumulative + counterbalancing)"),
]


def build_blank_templates():
    det = pd.DataFrame(
        [(c, s, lbl, *[np.nan] * len(BUCKET_LABELS)) for c, s, lbl in DETAILED_ROWS],
        columns=["line_code", "section", "line_item", *BUCKET_LABELS],
    )
    det.to_csv(INPUTS / "template_LQ_01_01_blank.csv", index=False)
    det.to_csv(INPUTS / "template_LQR_01_01_blank.csv", index=False)

    short = pd.DataFrame(
        [(c, lbl, *[np.nan] * len(BUCKET_LABELS)) for c, lbl in SHORT_ROWS],
        columns=["line_code", "line_item", *BUCKET_LABELS],
    )
    short.to_csv(INPUTS / "template_LQR_02_01_blank.csv", index=False)


def main():
    dfs = build_discount_factors()
    fx = build_fx()
    deals = build_deals(dfs, fx)
    expand_cashflows(deals, fx)
    build_scenarios()
    build_blank_templates()
    print(f"Inputs written to {INPUTS}")
    for p in sorted(INPUTS.glob("*.csv")):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
