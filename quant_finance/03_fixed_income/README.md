# 03 — Fixed Income

Bonds, curves, swaps, swaptions, LMM. Heaviest subject in the quant
track after options — and the prerequisite for `capstones/04_lmm_nn_surrogate/`.

## Notebooks (in order)

| # | Notebook | Focus |
|---|---|---|
| 01 | `01_bond_pricing.ipynb` | Discounting, YTM, accrued interest, day-count conventions |
| 02 | `02_duration_convexity_krd.ipynb` | Modified / Macaulay duration, convexity, key-rate durations |
| 03 | `03_curve_building.ipynb` | Bootstrapping a yield curve from market instruments |
| 04 | `04_swaps_swaptions.ipynb` | Vanilla IRS, swaption pricing via Black-76 |
| 05 | `05_libor_market_model.ipynb` | LMM — forward-rate dynamics, smile via SABR |

## Cheatsheets + walkthroughs

- [`curve_building_cheatsheet.md`](curve_building_cheatsheet.md) — bootstrap quick reference.
- [`lmm_cheatsheet.md`](lmm_cheatsheet.md) — LMM quick reference (Rebonato vol curve, correlation).
- [`lmm_end_to_end_walkthrough.md`](lmm_end_to_end_walkthrough.md) — production calibration workflow.
- [`sabr_end_to_end_walkthrough.md`](sabr_end_to_end_walkthrough.md) — SABR smile fitting end-to-end.

## Prerequisites

- `01_options/02_bs_family_and_asset_classes.ipynb` — Black-76 specifically.
- `06_stoch_calc/` — drift / volatility under change of measure (HJM intuition).

## Next

- `capstones/04_lmm_nn_surrogate/` — NN surrogate for LMM calibration. The natural application of everything in this module.
