# 03 — Fixed Income

Bonds, curves, swaps, swaptions, LMM. Heaviest subject in the quant
track after options — and the prerequisite for `capstones/04_lmm_nn_surrogate/`.

## Learning notes (in order)

| # | File | Focus |
|---|---|---|
| 01 | [`01_bond_pricing.md`](01_bond_pricing.md) | Discounting, YTM, accrued interest, day-count conventions |
| 02 | [`02_duration_convexity_krd.md`](02_duration_convexity_krd.md) | Modified / Macaulay duration, convexity, key-rate durations |
| 03 | [`03_curve_building.md`](03_curve_building.md) | Bootstrapping a yield curve from market instruments |
| 04 | [`04_swaps_swaptions.md`](04_swaps_swaptions.md) | Vanilla IRS, swaption pricing via Black-76 |
| 05 | [`05_libor_market_model.md`](05_libor_market_model.md) | LMM — forward-rate dynamics, smile via SABR |
| 06 | [`06_loan_amortisation.md`](06_loan_amortisation.md) | Annuity formula, amortisation schedules, mortgage duration, prepayment risk |

## Cheatsheets

- [`cheatsheets/curve_building.md`](cheatsheets/curve_building.md) — bootstrap quick reference.
- [`cheatsheets/lmm.md`](cheatsheets/lmm.md) — LMM quick reference (Rebonato vol curve, correlation).

## Walkthroughs

- [`walkthroughs/lmm_end_to_end.md`](walkthroughs/lmm_end_to_end.md) — production LMM calibration workflow.
- [`walkthroughs/sabr_end_to_end.md`](walkthroughs/sabr_end_to_end.md) — SABR smile fitting end-to-end.

## Prerequisites

- `01_options/02_bs_family_and_asset_classes.md` — Black-76 specifically.
- `06_stoch_calc/` — drift / volatility under change of measure (HJM intuition).

## Next

- `capstones/04_lmm_nn_surrogate/` — NN surrogate for LMM calibration. The natural application of everything in this module.
