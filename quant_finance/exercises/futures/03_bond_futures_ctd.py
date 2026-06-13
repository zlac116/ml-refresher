"""
FUT 3 — Bond Futures: Conversion Factor, Invoice Price, CTD
============================================================

OBJECTIVE
    Bond futures (e.g. CME US Treasury futures, Eurex Bund) allow the short
    to deliver any bond from a basket. The conversion factor (CF) normalises
    deliverable bonds to a notional coupon. The short selects the
    Cheapest-To-Deliver (CTD).

      1. Compute invoice price = F * CF + accrued interest.
      2. Compute gross basis = bond_price - F * CF.
      3. Pick the CTD = bond with the lowest gross basis from a 3-bond basket.

ESTIMATED TIME
    20 min

TOPICS
    Conversion factor (CME publishes per delivery month, per bond)
    Invoice price the long pays the short at delivery
    Gross basis: bond.price - F * CF.  CTD has the smallest GB.
    (Implied repo is the more accurate selector; gross-basis is a fast proxy.)

REAL-WORLD NOTE
    CME 10Y T-Note (ZN): notional $100k face; CF computed assuming a 6%
    yield. Eurex Bund: notional EUR 100k face, 6% yield assumption too.

REFERENCE
    Burghardt-Belton, "The Treasury Bond Basis"; Fabozzi ch. 17.

EXPECTED OUTPUT  (F=110.00, accrued varies per bond)
    Bond 0:  bond=102.50, CF=0.9234, invoice=103.0740, gross_basis=0.9260
    Bond 1:  bond=105.00, CF=0.9450, invoice=105.9500, gross_basis=1.0500
    Bond 2:  bond= 99.75, CF=0.8980, invoice= 99.8800, gross_basis=0.9700
    CTD index:           0
    CTD gross basis:     0.9260

GRADING
    All asserts must pass.
"""
import numpy as np
import pandas as pd


# ── TASK 1 ─────────────────────────────────────────────────────────────────
def invoice_price(futures_price: float, conv_factor: float, accrued: float) -> float:
    """Invoice = F * CF + accrued."""
    # TODO: implement
    raise NotImplementedError


# ── TASK 2 ─────────────────────────────────────────────────────────────────
def gross_basis(bond_price: float, futures_price: float, conv_factor: float) -> float:
    """Gross basis = bond_price - F * CF."""
    # TODO: implement
    raise NotImplementedError


# ── TASK 3 ─────────────────────────────────────────────────────────────────
def cheapest_to_deliver(basket: pd.DataFrame, futures_price: float) -> tuple[int, float]:
    """Given basket with columns ['bond_price', 'conv_factor', 'accrued']
    return (index of CTD bond, its gross basis) — CTD = lowest gross basis.
    """
    # TODO: implement
    raise NotImplementedError


# ── GRADING ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    basket = pd.DataFrame({
        "bond_price":  [102.50, 105.00, 99.75],
        "conv_factor": [0.9234, 0.9450, 0.8980],
        "accrued":     [1.50,   2.00,   1.10],
    })
    F = 110.00

    inv0 = invoice_price(F, 0.9234, 1.50)
    assert abs(inv0 - 103.074) < 1e-4

    gb0 = gross_basis(102.50, F, 0.9234)
    assert abs(gb0 - 0.926) < 1e-4

    # Apply across the whole basket
    basket["invoice"]     = basket.apply(
        lambda r: invoice_price(F, r["conv_factor"], r["accrued"]), axis=1)
    basket["gross_basis"] = basket.apply(
        lambda r: gross_basis(r["bond_price"], F, r["conv_factor"]), axis=1)

    assert abs(basket["invoice"].iloc[1]     - 105.950) < 1e-3
    assert abs(basket["gross_basis"].iloc[2] -   0.970) < 1e-3

    ctd_idx, ctd_gb = cheapest_to_deliver(
        basket[["bond_price", "conv_factor", "accrued"]], F
    )
    assert ctd_idx == 0
    assert abs(ctd_gb - 0.926) < 1e-4

    print("Bond 0:  bond=102.50, CF=0.9234, invoice=103.0740, gross_basis=0.9260")
    print("Bond 1:  bond=105.00, CF=0.9450, invoice=105.9500, gross_basis=1.0500")
    print("Bond 2:  bond= 99.75, CF=0.8980, invoice= 99.8800, gross_basis=0.9700")
    print(f"CTD index:           {ctd_idx}")
    print(f"CTD gross basis:     {ctd_gb:.4f}")
    print("\n✓ All checks passed.")
