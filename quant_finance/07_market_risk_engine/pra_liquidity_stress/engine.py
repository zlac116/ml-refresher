"""
engine.py  —  LiquidityStressEngine

Takes the CSV inputs (discount factors, deals, cashflows, fx, scenarios) and
populates the PRA cashflow-mismatch templates, base and stressed:

    LQR.01.01   detailed mismatch, ONE per compartment (each MAP + the remaining part)
    LQR.02.01   short (~150 data point) mismatch, ONE per compartment
    LQ.01.01    detailed mismatch at SOLO / legal-entity level (roll-up of compartments)

The engine is deliberately dependency-light (numpy + pandas) and every pricing
step is a small, auditable closed form driven off the supplied discount factors.

Key mechanics
-------------
* BASE ladder      : nominal contractual cashflows bucketed on a maturity ladder.
* STRESS overlay   : reprice instruments off a SHOCKED discount curve to get the
                     extra cash that moves — derivative variation margin (VM) and
                     repo haircut top-ups land in the <=1m bucket (they are immediate).
* Counterbalancing : cash + repo-able gilts (post-haircut), eroded under stress.
* Survival metric  : cumulative net mismatch + opening counterbalancing >= 0 ?
"""
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass

# ---- maturity ladder --------------------------------------------------------
# (lo_month, hi_month) inclusive, aligned to the template bucket labels.
BUCKET_BOUNDS = [(1, 1), (2, 3), (4, 6), (7, 12), (13, 24), (25, 36),
                 (37, 60), (61, 120), (121, 240), (241, 360), (361, 720)]
BUCKET_LABELS = ["<=1m", "1-3m", "3-6m", "6-12m", "1-2y", "2-3y",
                 "3-5y", "5-10y", "10-20y", "20-30y", "30-60y"]
N_BUCKETS = len(BUCKET_BOUNDS)
IMMEDIATE = 0                      # index of the <=1m bucket — where VM/haircut land


def month_to_bucket(m: int) -> int:
    """Map a cashflow month (1..720) to a ladder-bucket index."""
    m = max(1, int(m))
    for i, (lo, hi) in enumerate(BUCKET_BOUNDS):
        if lo <= m <= hi:
            return i
    return N_BUCKETS - 1           # anything beyond 720m falls in the last bucket


@dataclass
class Scenario:
    scenario_id: str
    rate_bp: float
    credit_bp: float
    gilt_bp: float
    fx_gbp_depr_pct: float
    infl_bp: float
    gilt_haircut_addon: float
    repo_haircut_addon: float
    description: str = ""

    @property
    def is_base(self) -> bool:
        return self.scenario_id == "BASE"


class LiquidityStressEngine:
    def __init__(self, inputs_dir: str | Path):
        self.dir = Path(inputs_dir)
        self.dfs = pd.read_csv(self.dir / "discount_factors.csv").set_index("month")
        self.deals = pd.read_csv(self.dir / "deals.csv")
        self.cashflows = pd.read_csv(self.dir / "cashflows.csv")
        self.fx = pd.read_csv(self.dir / "fx_rates.csv").set_index("currency")["gbp_per_unit"]
        sc = pd.read_csv(self.dir / "stress_scenarios.csv")
        self.scenarios = [Scenario(**r) for r in sc.to_dict("records")]
        self.compartments = list(self.deals["compartment"].unique())

    # ------------------------------------------------------------------ #
    # Curve helpers: shocked discount factors from the supplied base DFs. #
    # ------------------------------------------------------------------ #
    def _base_df(self, ccy: str, month: int) -> float:
        month = int(np.clip(month, 1, self.dfs.index.max()))
        return float(self.dfs.loc[month, ccy])

    def zero_rate(self, ccy: str, month: int) -> float:
        """Annually-compounded spot rate implied by the base DF:  z = DF^(-1/t) - 1."""
        month = int(np.clip(month, 1, self.dfs.index.max()))
        t = month / 12.0
        return self._base_df(ccy, month) ** (-1.0 / t) - 1.0

    def shocked_df(self, ccy: str, month: int, spread_bp: float = 0.0) -> float:
        """DF under a parallel spot-rate shift of `spread_bp` bp, ANNUAL compounding.
        spot z = DF^(-1/t) - 1  ==>  DF' = (1 + z + s)^(-t).  (t in years)
        Matches Just's discrete-compounding DF<->spot convention (not continuous)."""
        month = int(np.clip(month, 1, self.dfs.index.max()))
        if spread_bp == 0.0:
            return self._base_df(ccy, month)              # exact, no round-trip
        t = month / 12.0
        z = self.zero_rate(ccy, month)
        return (1.0 + z + spread_bp / 1e4) ** (-t)

    def gbp(self, ccy: str, scn: Scenario) -> float:
        """GBP per unit of `ccy` after applying the FX shock (sterling depreciation
        lifts the GBP value of foreign currency)."""
        base = float(self.fx[ccy])
        if ccy == "GBP":
            return 1.0
        return base * (1.0 + scn.fx_gbp_depr_pct / 100.0)

    # ------------------------------------------------------------------ #
    # Instrument revaluation.                                            #
    # ------------------------------------------------------------------ #
    def receiver_swap_mtm(self, deal, rate_bp: float) -> float:
        """MTM (in deal currency) of a receive-fixed / pay-float swap off a
        single curve. rates up -> receiver loses value (posts VM)."""
        N = deal["notional"]
        k = deal["rate"]
        freq = int(deal["freq_months"])
        mat = int(deal["maturity_month"])
        tau = freq / 12.0
        pay_months = np.arange(freq, mat + 1, freq)
        fixed_pv = sum(N * k * tau * self.shocked_df(deal["currency"], m, rate_bp)
                       for m in pay_months)
        # spot-start float leg PV (pay float) = N * (DF(0) - DF(T)); DF(0)=1
        float_pv = N * (1.0 - self.shocked_df(deal["currency"], mat, rate_bp))
        return fixed_pv - float_pv

    def inflation_swap_mtm(self, deal, rate_bp: float, infl_bp: float) -> float:
        """MTM (deal ccy) of a receive-inflation zero-coupon inflation swap:

            MTM = N · DF_nom(T) · [ (1+π)^T − (1+b)^T ]

        b = contracted breakeven (deal['rate']); π = b + inflation shock (the projected
        breakeven). Discounted on the NOMINAL curve, so it responds to BOTH the rate
        shock (through DF_nom) and the inflation shock (through π).

        Note: for a par swap (π=b) a *pure* rate move gives ~0 — which is correct. A par
        inflation swap has ~no standalone rate DV01; the rate effect materialises by
        discounting the inflation-driven payoff, i.e. in any scenario that also moves
        inflation (e.g. COMBINED_PRA). Deflation (infl_bp<0) is a loss to the receiver."""
        N = deal["notional"]
        T = deal["maturity_month"] / 12.0
        b = deal["rate"]                                  # contracted breakeven
        pi = b + infl_bp / 1e4                            # shocked projected breakeven
        df = self.shocked_df(deal["currency"], int(deal["maturity_month"]), rate_bp)
        return N * df * ((1 + pi) ** T - (1 + b) ** T)

    def bond_value(self, deal, rate_bp: float, extra_spread_bp: float) -> float:
        """PV (deal ccy) of a bond/gilt's own cashflows discounted at the shocked
        curve + an extra spread (credit for corp bonds, gilt spread for gilts)."""
        N = deal["notional"]
        cpn = N * deal["rate"] * (deal["freq_months"] / 12.0)
        mat = int(deal["maturity_month"])
        freq = int(deal["freq_months"])
        s = rate_bp + extra_spread_bp
        pv = 0.0
        for m in range(freq, mat + 1, freq):
            cf = cpn + (N if m == mat else 0.0)
            pv += cf * self.shocked_df(deal["currency"], m, s)
        return pv

    def fx_forward_mtm(self, deal, spot: float, rate_bp: float) -> float:
        """MTM (GBP) of a SOLD foreign forward (receive GBP strike, deliver foreign)
        at maturity T:  MTM = N*K*DF_dom(T) - N*spot*DF_for(T).
        `spot` = GBP per foreign (pass base for base MTM, shocked for stressed)."""
        N, K, T, ccy = deal["notional"], deal["rate2"], int(deal["maturity_month"]), deal["currency"]
        return (N * K * self.shocked_df("GBP", T, rate_bp)
                - N * spot * self.shocked_df(ccy, T, rate_bp))

    def xccy_swap_mtm(self, deal, spot: float, rate_bp: float) -> float:
        """MTM (GBP) of a pay-foreign / receive-GBP cross-currency swap:
        MTM = PV(GBP leg) - spot * PV(foreign leg), both legs incl. principal.
        Notionals matched at inception spot (N_gbp = N_for * base spot)."""
        N_for, ccy = deal["notional"], deal["currency"]
        freq, T = int(deal["freq_months"]), int(deal["maturity_month"])
        tau = freq / 12.0
        c_for, c_gbp = deal["rate"], deal["rate2"]
        N_gbp = N_for * float(self.fx[ccy])                  # fixed at inception spot
        pay_months = np.arange(freq, T + 1, freq)
        gbp_pv = sum(c_gbp * N_gbp * tau * self.shocked_df("GBP", m, rate_bp) for m in pay_months)
        gbp_pv += N_gbp * self.shocked_df("GBP", T, rate_bp)
        for_pv = sum(c_for * N_for * tau * self.shocked_df(ccy, m, rate_bp) for m in pay_months)
        for_pv += N_for * self.shocked_df(ccy, T, rate_bp)
        return gbp_pv - spot * for_pv

    # ------------------------------------------------------------------ #
    # Counterbalancing capacity (a STOCK, in GBP) per compartment.        #
    # ------------------------------------------------------------------ #
    def counterbalancing(self, compartment: str, scn: Scenario) -> dict:
        """Cash + repo-able gilts (post-haircut), all in GBP, under scenario `scn`.
        Returns a breakdown so the analysis can attribute the erosion."""
        dd = self.deals[self.deals["compartment"] == compartment]
        cash = 0.0
        gilt_value = 0.0
        for _, d in dd.iterrows():
            if d["product_type"] == "CASH":
                cash += d["notional"] * self.gbp(d["currency"], scn)
            elif d["product_type"] == "GILT":
                mv = self.bond_value(d, scn.rate_bp, scn.gilt_bp)     # shocked MV
                haircut = d["haircut"] + scn.gilt_haircut_addon
                gilt_value += mv * (1.0 - haircut) * self.gbp(d["currency"], scn)
        return {"cash": cash, "gilt_repo_value": gilt_value,
                "total": cash + gilt_value}

    # ------------------------------------------------------------------ #
    # Build one compartment's ladder (11 buckets x line items) for a scn. #
    # ------------------------------------------------------------------ #
    def compartment_ladder(self, compartment: str, scn: Scenario) -> pd.DataFrame:
        codes = ["I1", "I2", "I4", "I5", "O1", "O2", "O3"]
        L = {c: np.zeros(N_BUCKETS) for c in codes}

        # ---- (a) BASE contractual cashflows from cashflows.csv -------------
        deal_cmp = self.deals.set_index("deal_id")["compartment"].to_dict()
        cat_to_code = {"INSURANCE_OUT": "O1", "ASSET_IN": "I2",
                       "REPO_IN": "I5", "REPO_OUT": "O3",
                       # FX forward / XCCY gross settlement legs -> derivative lines
                       "FX_SETTLE_IN": "I4", "FX_SETTLE_OUT": "O2",
                       "XCCY_IN": "I4", "XCCY_OUT": "O2"}
        cf = self.cashflows[self.cashflows["deal_id"].map(deal_cmp) == compartment]
        for _, r in cf.iterrows():
            code = cat_to_code[r["flow_category"]]
            b = month_to_bucket(r["month"])
            # base ladder is nominal; each flow converted at its OWN currency's (shocked) FX
            L[code][b] += r["amount"] * self.gbp(r["currency"], scn)

        # ---- (b) STRESS overlay: derivative VM + repo haircut top-ups ------
        if not scn.is_base:
            dd = self.deals[self.deals["compartment"] == compartment]
            for _, d in dd.iterrows():
                pt = d["product_type"]
                if pt == "RECEIVER_SWAP":
                    dmtm = (self.receiver_swap_mtm(d, scn.rate_bp)
                            - self.receiver_swap_mtm(d, 0.0))
                    vm = dmtm * self.gbp(d["currency"], scn)     # +receive / -post
                    (L["I4"] if vm >= 0 else L["O2"])[IMMEDIATE] += vm
                elif pt == "INFLATION_SWAP":
                    # reprice on the shocked NOMINAL curve -> responds to rate AND inflation
                    dmtm = (self.inflation_swap_mtm(d, scn.rate_bp, scn.infl_bp)
                            - self.inflation_swap_mtm(d, 0.0, 0.0))
                    vm = dmtm * self.gbp(d["currency"], scn)
                    (L["I4"] if vm >= 0 else L["O2"])[IMMEDIATE] += vm
                elif pt == "FX_FORWARD":
                    # FX-sensitive VM: revalue at shocked spot & curves vs base
                    dmtm = (self.fx_forward_mtm(d, self.gbp(d["currency"], scn), scn.rate_bp)
                            - self.fx_forward_mtm(d, float(self.fx[d["currency"]]), 0.0))
                    (L["I4"] if dmtm >= 0 else L["O2"])[IMMEDIATE] += dmtm
                elif pt == "XCCY_SWAP":
                    dmtm = (self.xccy_swap_mtm(d, self.gbp(d["currency"], scn), scn.rate_bp)
                            - self.xccy_swap_mtm(d, float(self.fx[d["currency"]]), 0.0))
                    (L["I4"] if dmtm >= 0 else L["O2"])[IMMEDIATE] += dmtm
                elif pt == "REPO":
                    # lender raises haircut -> post extra collateral now (outflow)
                    topup = d["notional"] * scn.repo_haircut_addon
                    L["O3"][IMMEDIATE] += -topup * self.gbp(d["currency"], scn)

        # ---- (c) derived rows ----------------------------------------------
        inflow = L["I1"] + L["I2"] + L["I4"] + L["I5"]
        outflow = L["O1"] + L["O2"] + L["O3"]                    # outflows are negative
        net = inflow + outflow
        cum = np.cumsum(net)
        cbc = self.counterbalancing(compartment, scn)["total"]
        cbc_row = np.zeros(N_BUCKETS)
        cbc_row[IMMEDIATE] = cbc                                 # opening stock
        surv = cum + cbc                                         # cumulative + buffer

        rows = {**L, "N": net, "CUM": cum, "CBC": cbc_row, "SURV": surv}
        df = pd.DataFrame(rows, index=BUCKET_LABELS).T
        df.index.name = "line_code"
        df.insert(0, "compartment", compartment)
        df.insert(1, "scenario", scn.scenario_id)
        return df

    # ------------------------------------------------------------------ #
    # Assemble the three populated templates across compartments/scenarios#
    # ------------------------------------------------------------------ #
    def populate_LQR_01_01(self, scenarios=None) -> pd.DataFrame:
        """Detailed mismatch — one block per (compartment, scenario)."""
        scenarios = scenarios or self.scenarios
        blocks = [self.compartment_ladder(c, s)
                  for s in scenarios for c in self.compartments]
        return pd.concat(blocks).reset_index()

    def populate_LQ_01_01(self, scenarios=None) -> pd.DataFrame:
        """Solo / legal-entity roll-up = sum of compartments (per scenario).
        NB: a naive sum treats cash as fully fungible; the analysis flags where
        that masks a trapped-cash breach in an individual MAP."""
        scenarios = scenarios or self.scenarios
        out = []
        detail = self.populate_LQR_01_01(scenarios)
        for s in scenarios:
            sub = detail[detail["scenario"] == s.scenario_id]
            agg = sub.groupby("line_code")[BUCKET_LABELS].sum()
            # CUM / SURV must be recomputed from summed flows, not summed twice
            net = agg.loc["N"].values
            agg.loc["CUM"] = np.cumsum(net)
            cbc = agg.loc["CBC"].values
            agg.loc["SURV"] = np.cumsum(net) + cbc.sum()
            agg = agg.reset_index()
            agg.insert(0, "compartment", "SOLO")
            agg.insert(1, "scenario", s.scenario_id)
            out.append(agg)
        return pd.concat(out).reset_index(drop=True)

    def populate_LQR_02_01(self, scenarios=None) -> pd.DataFrame:
        """Short template — condensed net lines per (compartment, scenario)."""
        scenarios = scenarios or self.scenarios
        detail = self.populate_LQR_01_01(scenarios)
        rows = []
        for (comp, scen), g in detail.groupby(["compartment", "scenario"], sort=False):
            gi = g.set_index("line_code")[BUCKET_LABELS]
            s1 = gi.loc["I1"] + gi.loc["O1"]                         # net insurance
            s2 = gi.loc["I4"] + gi.loc["O2"]                         # net derivative/VM
            s3 = gi.loc["I5"] + gi.loc["O3"] + gi.loc["I2"]         # net financing+assets
            s4 = gi.loc["CBC"]                                       # counterbalancing
            s5 = gi.loc["SURV"]                                      # cumulative + CBC
            for code, lbl, vals in [
                ("S1", "Net insurance flow", s1),
                ("S2", "Net derivative / variation-margin flow", s2),
                ("S3", "Net secured-financing & asset flow", s3),
                ("S4", "Counterbalancing capacity (opening stock)", s4),
                ("S5", "Net liquidity position (cumulative + CBC)", s5),
            ]:
                rows.append([comp, scen, code, lbl, *vals.values])
        return pd.DataFrame(rows, columns=["compartment", "scenario",
                                           "line_code", "line_item", *BUCKET_LABELS])
