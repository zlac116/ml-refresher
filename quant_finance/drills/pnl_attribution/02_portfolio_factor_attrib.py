"""
PNL 2 — Portfolio Factor Attribution
====================================

OBJECTIVE
    A small derivatives book holds 4 option positions. Aggregate the Greeks
    weighted by quantity, then apply a market scenario (dS, dvol, dt) and
    attribute the portfolio P&L to delta/gamma/vega/theta.

ESTIMATED TIME
    20 min

TOPICS
    Linear aggregation of Greeks: Greek_book = sum_i (qty_i * Greek_i)
    Aggregation works ONLY because the shock factors are common (single S,
    single dvol). Idiosyncratic vol-per-option books need per-position vega.
    Reprice all positions and compare to the explain.

REAL-WORLD NOTE
    Real risk systems bucket vega by tenor and strike. Here we use a single
    common vol shock for clarity; the next exercise (risk 03) does buckets.

REFERENCE
    Hull, ch. 19; production "explain" reports.

EXPECTED OUTPUT  (book of 4 positions, dS=+1, dvol=+0.005, dt=1/252)
    portfolio value0 = -985.8418
    portfolio delta  = -54.7624
    portfolio gamma  =   2.7501
    portfolio vega   = 1450.2914
    portfolio theta  = -448.0491
    actual PNL       =  -47.7680
    delta explain    =  -54.7624
    gamma explain    =    1.3750
    vega  explain    =    7.2515
    theta explain    =   -1.7780
    sum   explain    =  -47.9139
    residual         =    0.1458

GRADING
    All asserts must pass; residual must be small relative to actual.
"""
import numpy as np
from scipy.stats import norm


def _bsm(S, K, T, r, sigma, kind: str):
    """Return (price, delta, gamma, vega, theta) for kind in {'call','put'}."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega  = S * norm.pdf(d1) * np.sqrt(T)
    if kind == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
        theta = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1
        theta = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)
    return price, delta, gamma, vega, theta


# ── Position book (qty, kind, K, T, sigma) ─────────────────────────────────
BOOK: list[dict] = [
    {"qty":  100, "kind": "call", "K": 100, "T": 1.0, "sigma": 0.20},
    {"qty":   50, "kind": "call", "K": 105, "T": 0.5, "sigma": 0.22},
    {"qty": -150, "kind": "call", "K":  95, "T": 2.0, "sigma": 0.18},
    {"qty":   75, "kind": "put",  "K": 100, "T": 1.0, "sigma": 0.20},
]


# ── TASK 1 ─────────────────────────────────────────────────────────────────
def book_greeks(book: list[dict], S: float, r: float) -> dict:
    """Return a dict with keys value, delta, gamma, vega, theta — each one
    is sum over positions of qty * per-position Greek/value.
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 2 ─────────────────────────────────────────────────────────────────
def book_explain(greeks: dict, dS: float, dvol: float, dt: float) -> dict:
    """Return delta/gamma/vega/theta/total contributions for the portfolio
    using the linear Taylor formula.
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 3 ─────────────────────────────────────────────────────────────────
def book_actual_pnl(book: list[dict], S0: float, r: float,
                    dS: float, dvol: float, dt: float) -> float:
    """Reprice every position at (S0+dS, T-dt, sigma+dvol) and return the
    book P&L = sum qty * (price_after - price_before).
    """
    # TODO: implement
    raise NotImplementedError


# ── GRADING ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    S0, r = 100.0, 0.05
    dS, dvol, dt = 1.0, 0.005, 1 / 252

    g = book_greeks(BOOK, S0, r)
    assert abs(g["value"] -  -985.8418) < 1e-3
    assert abs(g["delta"] -   -54.7624) < 1e-3
    assert abs(g["gamma"] -     2.7501) < 1e-3
    assert abs(g["vega" ] -  1450.2914) < 1e-3
    assert abs(g["theta"] -  -448.0491) < 1e-3

    ex = book_explain(g, dS, dvol, dt)
    assert abs(ex["delta"] -  -54.7624) < 1e-3
    assert abs(ex["gamma"] -    1.3750) < 1e-3
    assert abs(ex["vega" ] -    7.2515) < 1e-3
    assert abs(ex["theta"] -   -1.7780) < 1e-3
    assert abs(ex["total"] -  -47.9139) < 1e-3

    actual = book_actual_pnl(BOOK, S0, r, dS, dvol, dt)
    assert abs(actual - -47.7680) < 1e-3

    residual = actual - ex["total"]
    # Residual should be small compared to actual
    assert abs(residual / actual) < 0.05, f"residual fraction too large: {residual/actual}"

    print(f"portfolio value0 = {g['value']:.4f}")
    print(f"portfolio delta  = {g['delta']:.4f}")
    print(f"portfolio gamma  = {g['gamma']:.4f}")
    print(f"portfolio vega   = {g['vega']:.4f}")
    print(f"portfolio theta  = {g['theta']:.4f}")
    print(f"actual PNL       = {actual:.4f}")
    print(f"delta explain    = {ex['delta']:.4f}")
    print(f"gamma explain    = {ex['gamma']:.4f}")
    print(f"vega  explain    = {ex['vega']:.4f}")
    print(f"theta explain    = {ex['theta']:.4f}")
    print(f"sum   explain    = {ex['total']:.4f}")
    print(f"residual         = {residual:.4f}")
    print("\n✓ All checks passed.")
