# 02 — Risk

Market + credit risk fundamentals. The "what could go wrong" side of
quant finance.

## Notebooks (in order)

| # | Notebook | Focus |
|---|---|---|
| 01 | `01_var_methods.ipynb` | VaR via historical / parametric / Monte Carlo; coherence properties |
| 02 | `02_expected_shortfall.ipynb` | ES (CVaR) — why it's the coherent successor to VaR |
| 03 | `03_credit_merton.ipynb` | Merton structural model; PD from equity prices |
| 04 | `04_cva_intro.ipynb` | Counterparty Valuation Adjustment — the EE/EPE/CVA chain |

## Prerequisites

- `01_options/` — BS-family pricing for the CVA exposure profile.
- `fundamentals/mathematics.ipynb` — distributions, moments.

## Next

- `03_fixed_income/` and `06_stoch_calc/` for the underlying SDE machinery.
- The full XVA picture lives in `capstones/03_xva/`.

## Related

- [`toolkit/eda_decisions.md`](../../toolkit/eda_decisions.md) — distribution selection upstream of VaR.
