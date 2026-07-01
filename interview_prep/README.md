# interview_prep/

Company- and role-specific interview preparation material. Separated from the
core learning path (`ml/`, `quant_finance/`, `api_engineering/`, `llm_pipeline/`,
`toolkit/`) so that general reference stays uncluttered.

## Contents

| Path | Purpose |
|---|---|
| [`STUDY_GUIDE.md`](STUDY_GUIDE.md) | 4-week compressed plan targeting IB + AM quant interviews (~80 hours) |
| [`just_group/`](just_group/) | Role-specific prep for the Just Group valuation-quant interview (UK insurance, actuarial + market-risk focus) |

## just_group/ subfolder

| File | Focus |
|---|---|
| `hist_var_cheatsheet.md/pdf` | Historical-simulation VaR engine — the technical assessment prep |
| `Just_valuation_cheatsheet.md/pdf` | Python / derivatives / liquidity / stress testing — the interview cheatsheet |
| `python_fundamentals_cheatsheet.md/pdf` | Data structures / functions / classes for the coding assessment |

## When this content goes stale

Interview material has a shelf life. If a target role is closed or you're no
longer preparing for it:

1. **Preserve on a branch**: `git checkout -b interview-archive-<year> && git push origin interview-archive-<year>`
2. **Delete from `main`**: `git rm -r interview_prep/<subfolder>/`
3. **Commit** with a note pointing to the archive branch.

This keeps `main` focused on evergreen reference material while retaining the
prep artifacts in git history.

## What lives here vs. what stays in the subject dirs

- **Here**: role-specific talking points, single-company assessment prep,
  targeted 4-week timeboxed plans.
- **Not here**: general-purpose cheatsheets, gold references, tutorial
  narratives. Those live in the subject dirs (e.g.
  `quant_finance/07_market_risk_engine/cheatsheets/risk_methods_cheatsheet.md`
  is the general reference; the interview-focused version was here).
