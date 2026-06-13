# 06 — Stochastic Calculus

The mathematical engine room. **Reference** material — open these when an
SDE in another note isn't clicking, not as a sequential course.

## Learning notes (in order)

| # | File | Focus |
|---|---|---|
| 01 | [`01_brownian_motion.md`](01_brownian_motion.md) | Wiener process; quadratic variation; reflection |
| 02 | [`02_ito_and_gbm.md`](02_ito_and_gbm.md) | Itô's lemma; geometric Brownian motion; risk-neutral measure |
| 03 | [`03_lsmc_american.md`](03_lsmc_american.md) | Longstaff-Schwartz Monte Carlo for early-exercise options |

## When to use

- Stuck on a derivation in `01_options/` or `03_fixed_income/`? Most likely answer is in `02_ito_and_gbm.md`.
- About to implement an MC pricer with early exercise? Read `03_lsmc_american.md` first.

## Prerequisites

- `fundamentals/mathematics.ipynb` — measure theory + probability basics.

## Related

- [`01_options/`](../01_options/) — applied: prices an option using these tools.
- [`05_volatility/03_local_vol_dupire.md`](../05_volatility/03_local_vol_dupire.md) — the SDE perspective on local vol.
