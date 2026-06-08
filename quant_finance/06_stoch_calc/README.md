# 06 — Stochastic Calculus

The mathematical engine room. **Reference** material — open these when an
SDE in another notebook isn't clicking, not as a sequential course.

## Notebooks (in order)

| # | Notebook | Focus |
|---|---|---|
| 01 | `01_brownian_motion.ipynb` | Wiener process; quadratic variation; reflection |
| 02 | `02_ito_and_gbm.ipynb` | Itô's lemma; geometric Brownian motion; risk-neutral measure |
| 03 | `03_lsmc_american.ipynb` | Longstaff-Schwartz Monte Carlo for early-exercise options |

## When to use

- Stuck on a derivation in `01_options/` or `03_fixed_income/`? Most likely answer is in `02_ito_and_gbm.ipynb`.
- About to implement an MC pricer with early exercise? Read `03_lsmc_american.ipynb` first.

## Prerequisites

- `fundamentals/mathematics.ipynb` — measure theory + probability basics.

## Related

- [`01_options/`](../01_options/) — applied: prices an option using these tools.
- [`05_volatility/03_local_vol_dupire.ipynb`](../05_volatility/03_local_vol_dupire.ipynb) — the SDE perspective on local vol.
