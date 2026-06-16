"""
FI 15 — Parsing Market Quotes (Bloomberg/Reuters → Model Inputs)
==================================================================

OBJECTIVE
    Build the parse-and-convert layer between SCREEN quotes and the inputs
    your other exercises consume. Tests:
      1. Treasury price in 32nds/64ths → decimal
      2. T-bill discount yield → BEY (investment yield)
      3. SOFR future price → implied rate
      4. SOFR future P&L per bp move
      5. Bond clean price → dirty price (given accrued)

ESTIMATED TIME
    20 min

TOPICS
    Treasury 32nds notation:  "102-16+"  =  102 + 16/32 + 1/64
    T-bill discount yield:    dy = (F-P)/F * 360/days   (banker's convention, ACT/360)
    T-bill investment yield:  iy = (F-P)/P * 365/days   (BEY, ACT/365, higher)
    SOFR future quoted as:    100 - rate   (e.g. 95.50 = 4.50%)
    SOFR tick value:          $25 per basis point per contract

REFERENCE
    See 03_fixed_income/cheatsheets/market_quoting.md for full conventions.
    ISDA day-count definitions; OpenGamma rates conventions guide.

EXPECTED OUTPUT
    Treasury price parsing:
      '102-16+' -> 102.515625
      '102-16'  -> 102.500000
      '98-08'   -> 98.250000
      '100-00'  -> 100.000000

    T-bill 91-day, face=100, price=98.5:
      discount yield (360 basis)   = 0.059341  (5.9341%)
      investment yield (BEY, 365)  = 0.061081  (6.1081%)   ← always higher

    SOFR future at 95.50 -> implied rate = 0.045000  (4.50%)
    100 contracts × +5 bp move → P&L = $12500

    Bond clean=99.75, accrued=1.25 -> dirty=101.0000
"""
import numpy as np


# ── TASK 1 ─────────────────────────────────────────────────────────────────
def parse_treasury_price(quote: str) -> float:
    """Parse a Treasury price quote in 32nds.

    Format: "WW-FF" or "WW-FF+"
      WW = whole points (e.g. 102)
      FF = thirty-seconds component (e.g. 16 = 16/32 = 0.5)
      +  = optional extra 1/64 (if present)

    Examples:
      "102-16+" -> 102 + 16/32 + 1/64 = 102.515625
      "98-08"   -> 98 + 8/32 = 98.25
      "100-00"  -> 100.0
    """
    # TODO: implement (hint: split on "-", check for trailing "+")
    raise NotImplementedError


# ── TASK 2 ─────────────────────────────────────────────────────────────────
def tbill_discount_yield(face: float, price: float, days: int) -> float:
    """Discount yield (ACT/360 banker's convention):

        dy = (face - price) / face * 360 / days
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 3 ─────────────────────────────────────────────────────────────────
def tbill_investment_yield(face: float, price: float, days: int) -> float:
    """Investment yield / BEY (ACT/365):

        iy = (face - price) / price * 365 / days

    NOTE: investment yield uses PRICE in the denominator (not face),
    and 365 (not 360). Both effects make it HIGHER than the discount yield.
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 4 ─────────────────────────────────────────────────────────────────
def sofr_future_implied_rate(quoted_price: float) -> float:
    """SOFR / Eurodollar future: implied 3M rate = (100 - quoted_price) / 100.

    e.g. quote 95.50 -> 4.50% (=0.045)
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 5 ─────────────────────────────────────────────────────────────────
def sofr_future_pnl(n_contracts: int, bps_move: float,
                    tick_value_per_bp: float = 25.0) -> float:
    """P&L on `n_contracts` SOFR / ED futures from a `bps_move` rate change.

    pnl = n_contracts * bps_move * tick_value_per_bp

    Tick value is $25/bp/contract for SOFR (3M) and Eurodollar.
    Sign: positive bps_move means rates fell (price rose) → long position gains.
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 6 ─────────────────────────────────────────────────────────────────
def clean_to_dirty(clean_price: float, accrued: float) -> float:
    """dirty = clean + accrued. (The trade settles at dirty.)"""
    # TODO: implement
    raise NotImplementedError


# ── GRADING ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Treasury price parsing
    p_full = parse_treasury_price("102-16+")
    p_half = parse_treasury_price("102-16")
    p_disc = parse_treasury_price("98-08")
    p_par  = parse_treasury_price("100-00")
    assert abs(p_full - 102.515625) < 1e-9
    assert abs(p_half - 102.500000) < 1e-9
    assert abs(p_disc -  98.250000) < 1e-9
    assert abs(p_par  - 100.000000) < 1e-9

    # T-bill yields
    dy = tbill_discount_yield(100, 98.5, 91)
    iy = tbill_investment_yield(100, 98.5, 91)
    assert abs(dy - 0.059341) < 1e-5
    assert abs(iy - 0.061081) < 1e-5
    assert iy > dy, "investment yield must exceed discount yield"

    # SOFR future
    rate = sofr_future_implied_rate(95.50)
    assert abs(rate - 0.045) < 1e-9
    pnl = sofr_future_pnl(100, 5.0)
    assert abs(pnl - 12500.0) < 1e-6

    # Clean -> dirty
    d = clean_to_dirty(99.75, 1.25)
    assert abs(d - 101.0) < 1e-9

    print("Treasury price parsing:")
    print(f"  '102-16+' -> {p_full:.6f}")
    print(f"  '102-16'  -> {p_half:.6f}")
    print(f"  '98-08'   -> {p_disc:.6f}")
    print(f"  '100-00'  -> {p_par:.6f}")
    print()
    print(f"T-bill 91-day, face=100, price=98.5:")
    print(f"  discount yield (360 basis)   = {dy:.6f}  ({dy*100:.4f}%)")
    print(f"  investment yield (BEY, 365)  = {iy:.6f}  ({iy*100:.4f}%)   ← always higher")
    print()
    print(f"SOFR future at 95.50 -> implied rate = {rate:.6f}  ({rate*100:.2f}%)")
    print(f"100 contracts × +5 bp move → P&L = ${int(pnl)}")
    print()
    print(f"Bond clean=99.75, accrued=1.25 -> dirty={d:.4f}")
    print("\n✓ All checks passed.")
