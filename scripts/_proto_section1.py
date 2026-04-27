"""Prototype: rewrite Section 1 of classification.ipynb in the pipeline-stage format.

Idempotent — finds the existing Section 1 boundaries by header text, captures the
user's four old answer cells, and replaces the section with the new format. The
old answers are preserved verbatim inside a collapsible fold at the bottom.

Run from project root:
    python scripts/_proto_section1.py
"""
from __future__ import annotations
import re
from pathlib import Path
import nbformat as nbf

NB_PATH = Path('classification/classification.ipynb')


# ----------------------------------------------------------------------
# Cell-builder helpers
# ----------------------------------------------------------------------
def md(text: str):
    return nbf.v4.new_markdown_cell(text.lstrip('\n').rstrip() + '\n')


def code(text: str):
    return nbf.v4.new_code_cell(text.lstrip('\n').rstrip())


# ----------------------------------------------------------------------
# Capture the user's four existing answer cells from the live notebook
# ----------------------------------------------------------------------
def capture_old_answers(nb) -> dict[str, str]:
    """Return {old_id: source} for the user's pre-existing answer cells in section 1."""
    out = {}
    EX_RE = re.compile(r'\*\*Exercise\s(1\.[1-4])')
    for i, c in enumerate(nb.cells):
        if c.cell_type != 'markdown':
            continue
        m = EX_RE.match(c.source.strip()[:60])
        if m and i + 1 < len(nb.cells) and nb.cells[i + 1].cell_type == 'code':
            out[m.group(1)] = nb.cells[i + 1].source
    return out


# ----------------------------------------------------------------------
# New Section 1 content
# ----------------------------------------------------------------------
def build_new_cells(old_answers: dict[str, str]) -> list:
    cells = []

    # --- Stage 1 header + Why we're here ---
    cells.append(md(r"""
---
## Stage 1 — Data Quality

**↳ Why we're here.** Every model downstream — every metric, every backtest, every P&L estimate — assumes the input dataframe is a contiguous, deduplicated, properly-typed time series. If a single symbol has a 3-hour gap halfway through, every rolling statistic computed across that gap is silently wrong. If two timestamps duplicate, group-by-symbol counts disagree with date-range counts. **None of these problems raise an exception** — they produce *plausible but lying* output, which we then train models on. By the time we notice the model is "weirdly good", we've wasted weeks.

This stage's job is to prove the data is sound *before* building anything on top of it. We produce two artifacts that the rest of the pipeline depends on:

1. A long-format frame `df` whose `(ts, symbol)` pairs are unique, sorted chronologically inside each symbol, with all expected columns and no surprise NaNs.
2. A reusable **data quality report** function that asserts those properties — so the next time someone (you in two months, or an upstream ETL) breaks them, the pipeline crashes instead of producing fictional results.

We start by loading the raw long-format dataframe (one row per symbol-timestamp).
"""))

    # The data-load + sanity cells already exist (cells 5 and 6 of the original
    # notebook). We don't replicate them here — we splice this block in *after*
    # those cells, leaving them in place.

    # --- The 30-second concept ---
    cells.append(md(r"""
### The 30-second concept

Three pandas patterns do 90% of the data-quality work in a real pipeline:

```python
# 1. Per-group statistics over time using groupby.apply + diff
df.groupby('symbol')['ts'].apply(lambda s: s.sort_values().diff().dt.total_seconds().div(3600).max())
# → Series indexed by symbol; value = the largest gap in hours

# 2. Long → wide reshape
df.pivot(index='ts', columns='symbol', values='close').sort_index()
# → DataFrame with one column per symbol; rectangular, easy to feature-engineer

# 3. Continuity check via expected-grid + reindex
expected = pd.date_range(df['ts'].min(), df['ts'].max(), freq='1h', tz='UTC')
wide.reindex(expected).isna().sum()
# → Per-symbol count of missing bars
```

The *why* behind these specific choices:

- **`groupby(...).apply` rather than a Python loop.** Pandas is two orders of magnitude faster on real data. More importantly, it forces you to think about per-group operations in a vectorised way — that mindset transfers to every later stage.
- **`pivot` rather than concat-and-merge.** Pivot gives you a single rectangular array. Feature engineering downstream (rolling means, cross-asset correlations, lag features) becomes a one-liner per feature. With long-format data, every feature needs its own per-group apply.
- **`reindex` against an expected grid, not the timestamp column.** The *difference between actual and expected* is the missing-data signal. Just trusting whatever timestamps shipped tells you nothing about the bars you didn't observe.

`.sort_values()` inside the lambda and `.sort_index()` after pivot look redundant (both are sorted in current pandas), but they are *defensive* habits. If the source ever ships timestamps out of order — and at some point it will — every `.shift()` and `.rolling()` downstream will silently corrupt without them.
"""))

    # --- Worked example ---
    cells.append(md(r"""
### Worked example — running the three patterns on a deliberately broken toy frame

We build a tiny 3-symbol frame, deliberately introduce a duplicate row, a gap, and a NaN, then run each idiom and watch what it surfaces. Read the comments — they explain not just *what* each line does, but *why* that line is the right tool for the job.
"""))

    cells.append(code(r"""
import pandas as pd
import numpy as np

# Tiny toy with three deliberate problems.
toy = pd.DataFrame({
    'ts': pd.to_datetime([
        '2024-01-01 00:00', '2024-01-01 01:00', '2024-01-01 03:00',  # 'A' has a gap (no 02:00)
        '2024-01-01 00:00', '2024-01-01 01:00', '2024-01-01 02:00',
        '2024-01-01 00:00', '2024-01-01 00:00', '2024-01-01 02:00',  # 'C' has a duplicate at 00:00 and a missing 01:00
    ], utc=True),
    'symbol': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'],
    'close':  [100, 101, np.nan, 50, 51, 52, 30, 30.5, 32],   # 'A' also has a missing close
})

# 1. Largest gap per symbol — surfaces A's missing 02:00 bar.
#    Sorting INSIDE the lambda matters: groupby preserves first-row order, not
#    chronological order. A 2-row group with rows out of order would give a
#    *negative* timedelta — silently wrong.
gaps_h = toy.groupby('symbol')['ts'].apply(
    lambda s: s.sort_values().diff().dt.total_seconds().div(3600).max()
)
print('Largest gap per symbol (hours):')
print(gaps_h, '\n')

# 2. % missing per column — surfaces the NaN close in A.
#    Multiply by 100 so the output reads naturally as a percentage.
print('% missing per column:')
print((toy.isna().mean() * 100).round(2), '\n')

# 3. pivot WILL FAIL on duplicate (ts, symbol) pairs — and that's good.
#    A silent duplicate is far worse than a noisy crash; pivot's strictness
#    forces us to make a deliberate choice about how to dedupe.
try:
    toy.pivot(index='ts', columns='symbol', values='close')
except ValueError as e:
    print(f'pivot raised: {str(e)[:80]}...\n')

# Fix the duplicates explicitly: keep first observation per (ts, symbol).
# Alternatives in production: take the mean, or assert that duplicates are
# never expected (depends on the upstream contract).
toy_dedup = toy.drop_duplicates(subset=['ts', 'symbol'], keep='first')
wide_toy = toy_dedup.pivot(index='ts', columns='symbol', values='close').sort_index()
print('Wide pivot after dedup:')
print(wide_toy, '\n')

# 4. Continuity check via expected grid — surfaces the missing 01:00 row of C.
expected = pd.date_range(toy['ts'].min(), toy['ts'].max(), freq='1h', tz='UTC')
n_missing = wide_toy.reindex(expected).isna().sum()
print('Missing bars per symbol after reindex against the expected hourly grid:')
print(n_missing)
"""))

    # --- Failure mode ---
    cells.append(md(r"""
### Failure mode — what happens when you skip the wide-pivot and do the obvious thing instead

The naive way to compute per-symbol returns from a long frame:

```python
returns = df.groupby('symbol')['close'].apply(np.log).diff()
```

looks correct. **It isn't.** `.apply()` returns a flat Series; `.diff()` then operates on the *whole* series, ignoring group boundaries. At the row where one symbol's last bar gives way to the next symbol's first bar, you get `log(ETH_first) − log(BTC_last)` — a "return" between two different assets. Run the cell below before moving on.
"""))

    cells.append(code(r"""
# 1. The naive (broken) version: groupby.apply followed by .diff() on the result.
bad_returns = df.groupby('symbol')['close'].apply(np.log).diff()

# 2. The correct version: pivot to wide first, then diff per column.
wide_close_demo = df.pivot(index='ts', columns='symbol', values='close').sort_index()
correct_returns_btc = np.log(wide_close_demo['BTC']).diff()

print('Naive cross-symbol "returns":')
print(f'  std        = {bad_returns.std():.4f}')
print(f'  max |ret|  = {bad_returns.abs().max():.4f}    ← look at this number\n')

print('Correct BTC returns:')
print(f'  std        = {correct_returns_btc.std():.4f}')
print(f'  max |ret|  = {correct_returns_btc.abs().max():.4f}\n')

ratio = bad_returns.abs().max() / correct_returns_btc.abs().max()
print(f'The naive version\'s biggest "return" is {ratio:.0f}× the largest real BTC return.')
print('That single garbage value is log(next_symbol_first / prev_symbol_last) at')
print('the symbol boundary. A volatility model trained on this would learn that')
print('returns are wildly larger than they really are; any "edge" it discovers')
print('comes from that one fictional spike. The model will look great in')
print('backtest and lose money in production.')
"""))

    # --- Decisions ---
    cells.append(md(r"""
### Decisions you make at this stage

- **Drop, fill, or assert?** Dropping rows with missing data breaks downstream alignment if other symbols have data at the same timestamps — pivoting first lets you `NaN` one column without dropping the row. Filling (forward-fill, mean) introduces synthetic data that masks regime changes and inflates apparent precision. **Asserting** fails loudly, which is usually the right default at the prototype stage. Switch to filling only when you've quantified the impact on downstream metrics.

- **Per-symbol or jointly?** A gap in one symbol doesn't invalidate the others. Pivot first to a rectangular frame, *then* decide what to do per column.

- **Where to draw the failure threshold?** A single missing bar in 17,000 is fine. A 24-hour gap is not — the model will learn the wrong distribution of overnight moves. Codify these thresholds as numerical asserts (e.g. `assert max_gap_hours < 6`) so the next ETL regression trips the alarm instead of silently producing bad features.
"""))

    # --- Exercise 1.1 ---
    cells.append(md(r"""
### Exercise 1.1 — Build a data quality report

Real pipelines fail silently. The fix is to codify your assumptions as a **report**: a function that computes every quality metric you care about, and *asserts* against thresholds you can defend out loud.

Build `data_quality_report(df)` that returns a dict with these keys:

- `n_rows` — int
- `n_symbols` — int
- `max_gap_hours` — float, the largest per-symbol gap in hours
- `n_duplicate_rows` — int, count of duplicated `(ts, symbol)` pairs
- `pct_missing` — dict mapping column → percent missing (rounded to 2 dp)
- `n_missing_bars_per_symbol` — dict mapping symbol → number of bars missing from the expected hourly grid (after pivot + reindex)

Apply it to `df`. Print the report. Then write `assert` statements that would fail if any quality threshold is breached — pick the threshold values yourself and **justify them in a comment** (e.g. why 6 hours? why 5%? what would the impact be on downstream features?).
"""))

    cells.append(code(r"""
def data_quality_report(df: pd.DataFrame) -> dict:
    # TODO: build the dict with the six keys above, reusing the three patterns
    # from the worked example.
    pass

report = data_quality_report(df)
print(report)

# TODO: add asserts with thresholds you can justify.
# assert ...
"""))

    cells.append(md(r"""
<details><summary>💡 Click to reveal solution & explanation</summary>

```python
def data_quality_report(df: pd.DataFrame) -> dict:
    wide = df.pivot(index='ts', columns='symbol', values='close').sort_index()
    expected = pd.date_range(df['ts'].min(), df['ts'].max(), freq='1h', tz='UTC')

    return {
        'n_rows': len(df),
        'n_symbols': df['symbol'].nunique(),
        'max_gap_hours': float(
            df.groupby('symbol')['ts']
              .apply(lambda s: s.sort_values().diff().dt.total_seconds().div(3600).max())
              .max()
        ),
        'n_duplicate_rows': int(df.duplicated(subset=['ts', 'symbol']).sum()),
        'pct_missing': (df.isna().mean() * 100).round(2).to_dict(),
        'n_missing_bars_per_symbol': wide.reindex(expected).isna().sum().to_dict(),
    }

report = data_quality_report(df)
print(report)

# Justified asserts:
#  - 6 h gap: anything longer crosses a typical overnight regime, which would
#    distort the distribution of "next-bar" returns the model learns from.
#  - duplicates: the upstream contract guarantees uniqueness, so any duplicate
#    is a bug in the join or the source — fix it now, not after model training.
#  - 5% missing per symbol: above this, the symbol's data is too sparse to
#    rely on without re-fetching.
n_bars_expected = len(pd.date_range(df['ts'].min(), df['ts'].max(), freq='1h', tz='UTC'))
assert report['max_gap_hours'] < 6, f"Suspicious gap: {report['max_gap_hours']:.1f}h"
assert report['n_duplicate_rows'] == 0, f"Duplicates: {report['n_duplicate_rows']}"
for sym, n_missing in report['n_missing_bars_per_symbol'].items():
    assert n_missing / n_bars_expected < 0.05, f"{sym} missing {n_missing} bars"
```

**Why this shape works.** Each key in the report is something a downstream stage *would silently fail on* if it regressed. By computing them all at the start of every run, you turn silent corruption into a noisy crash.

The asserts are *thresholds*, not absence-tests — picking them is a judgement call worth thinking about. Setting them too tight gives false alarms; too loose lets corruption through. A team standard might be to start strict (`< 1` hour gap) and relax only when the false-alarm rate justifies it.

**Why pivot-then-reindex inside the function**: it materialises both the actual and the expected grid in one place, making the missing-bar count a single subtraction. The alternative (a per-symbol loop) is slower and reads worse — and you'd write that loop inconsistently every time.

</details>
"""))

    # --- Exercise 1.2 ---
    cells.append(md(r"""
### Exercise 1.2 — Diagnose a buggy pipeline cell

The cell below tries to compute hourly log-returns for BTC. It runs without errors and produces a plausible-looking output. But there is a subtle bug that *could* silently corrupt every feature built on top of it. Run the cell, compare the output to the worked-example version above, identify the bug, and write the fixed version in the answer cell beneath.

(Hint: think about whether this code is correct *because of the data we happen to have*, or *because of the operations being performed*. The first kind of correctness breaks the moment the data changes shape.)
"""))

    cells.append(code(r"""
# BUGGY — run me, then think about what assumption this code is relying on.
btc_rows = df[df['symbol'] == 'BTC']                    # filter to BTC
btc_returns_buggy = np.log(btc_rows['close']).diff()    # log-diff to get returns
print(f"buggy std:        {btc_returns_buggy.std():.6f}")
print(f"buggy max |ret|:  {btc_returns_buggy.abs().max():.4f}")
print(btc_returns_buggy.head(5))
"""))

    cells.append(code(r"""
# Your answer here — write the FIXED version below.
"""))

    cells.append(md(r"""
<details><summary>💡 Click to reveal solution & explanation</summary>

```python
btc_rows = df[df['symbol'] == 'BTC'].sort_values('ts')   # ← critical: ensure chronological order
btc_returns_correct = np.log(btc_rows['close']).diff()
print(f"correct std:        {btc_returns_correct.std():.6f}")
print(f"correct max |ret|:  {btc_returns_correct.abs().max():.4f}")
```

**The bug.** Earlier in the notebook we wrote `df = df.sort_values(['symbol', 'ts'])` — sorted by *symbol first, then ts*. When you filter to BTC, the rows are still in `ts` order *only because BTC happens to come first alphabetically* and was sorted internally. The moment the source ships symbols in a different order — or you join with another frame that doesn't preserve order — the BTC slice could come back un-sorted, and `np.log(close).diff()` would compute "returns" between non-adjacent times.

**The fix.** An explicit `.sort_values('ts')` after the filter. It costs almost nothing and makes the line's correctness *not depend* on the ambient sort order of the parent frame.

**The general lesson.** Pipelines are correct when each step is correct *in isolation*, not when the steps happen to compose nicely *given the current data*. Any operation that depends on ordering, group boundaries, alignment, or dtype is suspect until proven otherwise. Defensive `.sort_index()` / `.sort_values()` / `.copy()` calls are cheap insurance — and they document the invariants the line is relying on. Future-you, looking at the line in isolation, should be able to see that it's correct.

</details>
"""))

    # --- Recap ---
    cells.append(md(r"""
### Recap — what stage 1 produced

You now have:

- `df` — long-format frame, sorted, deduplicated, hourly UTC timestamps verified.
- A reusable `data_quality_report(df)` function. Run it at the start of every pipeline run; if it fails, **stop**.
- A reflex for spotting silent corruption: *any operation that depends on ordering, group boundaries, or alignment is suspect until proven otherwise.* Defensive `sort_*` / `.copy()` calls are cheap insurance.

In **stage 2** we'll pivot `df` to wide format, compute returns, and look at their distributional properties — the inputs to every feature we'll engineer in stage 3.
"""))

    # --- Your previous answers ---
    prev = "<details><summary>📁 Your previously-written answers from the old exercise format</summary>\n\n"
    prev += "The old format had four exercises (1.1 — largest gap; 1.2 — % missing; 1.3 — pivot; 1.4 — continuity check). Their building blocks have been folded into Exercise 1.1 above (the data quality report). Your answers from the previous iteration are preserved here verbatim for reference.\n\n"
    titles = {
        '1.1': 'Old 1.1 — Largest time gap per symbol',
        '1.2': 'Old 1.2 — Percent missing per column',
        '1.3': 'Old 1.3 — Pivot to wide',
        '1.4': 'Old 1.4 — Verify hourly continuity with reindex',
    }
    for ex_id in ['1.1', '1.2', '1.3', '1.4']:
        prev += f"**{titles[ex_id]}**\n\n"
        prev += "```python\n"
        prev += old_answers.get(ex_id, '# (no previous answer captured)').rstrip() + "\n"
        prev += "```\n\n"
    prev += "</details>"
    cells.append(md(prev))

    return cells


# ----------------------------------------------------------------------
# Surgery
# ----------------------------------------------------------------------
def main():
    nb = nbf.read(NB_PATH, as_version=4)

    old_answers = capture_old_answers(nb)
    print(f"captured old answers: {sorted(old_answers)}")

    # Locate Section 1's old header (cell 4) and Section 2's start.
    sec1_idx = None
    sec2_idx = None
    for i, c in enumerate(nb.cells):
        if c.cell_type != 'markdown':
            continue
        first = c.source.split('\n')[0].strip()
        if re.match(r'^##\s*1\.', first) and sec1_idx is None:
            sec1_idx = i
        elif re.match(r'^##\s*2\.', first) and sec2_idx is None:
            sec2_idx = i
            break

    if sec1_idx is None or sec2_idx is None:
        raise RuntimeError("could not locate section 1/2 boundaries")

    # The data-load + sanity cells (cells 5 and 6 in the original) we want to keep.
    # They live between sec1_idx and the "### Exercises — Section 1" header.
    keep_until = None
    for j in range(sec1_idx + 1, sec2_idx):
        if (
            nb.cells[j].cell_type == 'markdown'
            and '### Exercises — Section 1' in nb.cells[j].source[:60]
        ):
            keep_until = j
            break
    if keep_until is None:
        raise RuntimeError("could not locate end of preserved data-load region")

    print(f"section 1 spans cells [{sec1_idx}..{sec2_idx})")
    print(f"keep cells [{sec1_idx + 1}..{keep_until})  (data load + sanity stats)")

    # Build the new section.
    new_cells = build_new_cells(old_answers)
    # Replace the original Section 1 header (cell `sec1_idx`) with the first cell
    # of `new_cells` (the new Stage 1 header + Why), keep the data-load region
    # unchanged, then append the rest of the new cells, and continue from
    # `sec2_idx` onward.
    rebuilt = (
        nb.cells[:sec1_idx]
        + [new_cells[0]]                              # ## Stage 1 header
        + nb.cells[sec1_idx + 1: keep_until]          # data-load + sanity
        + new_cells[1:]                               # rest of new section 1
        + nb.cells[sec2_idx:]                          # section 2 onward
    )
    nb.cells = rebuilt
    nbf.validate(nb)
    nbf.write(nb, NB_PATH)
    print(f"wrote {NB_PATH} — total cells: {len(nb.cells)}")


if __name__ == '__main__':
    main()
