# 01 — Options

Foundational derivatives — Black-Scholes through to local + stochastic vol
models. Heaviest subject in the quant track; do this in order.

## Learning notes (in order)

| # | File | Focus |
|---|---|---|
| 00 | [`00_black_scholes_intuition.md`](00_black_scholes_intuition.md) | Narrative companion — the "why" before the maths |
| 01 | [`01_black_scholes.md`](01_black_scholes.md) | BS formula derivation + pricing |
| 02 | [`02_bs_family_and_asset_classes.md`](02_bs_family_and_asset_classes.md) | Bachelier, Black-76; rates / FX / commodities |
| 03 | [`03_greeks.md`](03_greeks.md) | Delta, gamma, vega, theta, rho, vanna, vomma |
| 04 | [`04_binomial_trees.md`](04_binomial_trees.md) | CRR + Jarrow-Rudd; American exercise |
| 05 | [`05_monte_carlo_pricing.md`](05_monte_carlo_pricing.md) | MC engine, variance reduction (antithetic, control variates) |
| 06 | [`06_implied_vol_surface.md`](06_implied_vol_surface.md) | Smile / skew / term structure; SABR fit |
| 07 | [`07_heston.md`](07_heston.md) | Heston stochastic vol — characteristic-function pricing |

## Cheatsheets

- [`cheatsheets/bs.md`](cheatsheets/bs.md) — Black-Scholes quick reference.
- [`cheatsheets/bachelier.md`](cheatsheets/bachelier.md) — normal-model variant (rates).
- [`cheatsheets/sabr.md`](cheatsheets/sabr.md) — SABR smile model.

## Prerequisites

- `fundamentals/mathematics.ipynb` — stochastic calculus intuition.
- `06_stoch_calc/` — Brownian motion + Itô if you haven't seen them.

## Next

- `02_risk/` — VaR / ES / credit.
- `03_fixed_income/` — bonds, swaps, LMM (uses these BS-family models).
