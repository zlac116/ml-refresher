# 01 — Options

Foundational derivatives — Black-Scholes through to local + stochastic vol
models. Heaviest notebook subject in the quant track; do this in order.

## Notebooks (in order)

| # | Notebook | Focus |
|---|---|---|
| 00 | `00_black_scholes_intuition.ipynb` | Narrative companion — the "why" before the maths |
| 01 | `01_black_scholes.ipynb` | BS formula derivation + pricing |
| 02 | `02_bs_family_and_asset_classes.ipynb` | Bachelier, Black-76; rates / FX / commodities |
| 03 | `03_greeks.ipynb` | Delta, gamma, vega, theta, rho, vanna, vomma |
| 04 | `04_binomial_trees.ipynb` | CRR + Jarrow-Rudd; American exercise |
| 05 | `05_monte_carlo_pricing.ipynb` | MC engine, variance reduction (antithetic, control variates) |
| 06 | `06_implied_vol_surface.ipynb` | Smile / skew / term structure; SABR fit |
| 07 | `07_heston.ipynb` | Heston stochastic vol — characteristic-function pricing |

## Cheatsheets

- [`bs_cheatsheet.md`](bs_cheatsheet.md) — Black-Scholes quick reference.
- [`bachelier_cheatsheet.md`](bachelier_cheatsheet.md) — normal-model variant (rates).
- [`sabr_cheatsheet.md`](sabr_cheatsheet.md) — SABR smile model.

## Prerequisites

- `fundamentals/mathematics.ipynb` — stochastic calculus intuition.
- `06_stoch_calc/` — Brownian motion + Itô if you haven't seen them.

## Next

- `02_risk/` — VaR / ES / credit.
- `03_fixed_income/` — bonds, swaps, LMM (uses these BS-family models).
