# Data Analysis Drills — pandas / numpy / matplotlib

Short, hands-on Python scripts (~15–20 min each) covering the moves you need
on muscle memory for any quant / data-engineering workflow.

## How to use them

Each file is a self-contained script with:

1. A **docstring** stating the objective, topics covered, estimated time,
   and the **expected output** when implemented correctly.
2. **Synthetic data** generated in-script via `np.random.seed(...)` so the
   assertions are deterministic — no downloads, no `data/` folder.
3. **Function stubs** marked `# TODO:` for you to fill in.
4. **Assertion block** at the bottom that grades your implementation: if it
   prints the expected output and nothing throws, you're correct.

```bash
cd toolkit/exercises/data_analysis_drills
python 01_returns_and_volatility.py    # will raise NotImplementedError until you fill it in
```

## Index

| #  | File                                       | Focus                                              |
|----|--------------------------------------------|----------------------------------------------------|
| 01 | `01_returns_and_volatility.py`             | pct_change, rolling, annualisation                 |
| 02 | `02_resample_and_align.py`                 | resample, asfreq, business calendars, ffill/bfill  |
| 03 | `03_multi_asset_panel.py`                  | pivot, melt, merge, MultiIndex                     |
| 04 | `04_correlation_heatmap.py`                | corr/cov, matplotlib imshow + annotate             |
| 05 | `05_numpy_vectorisation.py`                | broadcasting, where, percentile, einsum            |
| 06 | `06_groupby_aggregations.py`               | groupby, agg, transform, named aggregations        |
| 07 | `07_outlier_detection.py`                  | z-score, IQR, winsorize, boolean masking           |
| 08 | `08_rolling_window_plot.py`                | rolling, expanding, subplots, twin axes            |
| 09 | `09_groupby_power.py`                      | groupby + transform + rolling per group, ranks     |
| 10 | `10_multiindex_panel.py`                   | MultiIndex (date×ticker), xs, unstack/stack, IndexSlice |
| 11 | `11_calendar_timezone.py`                  | US holidays calendar, BDay shift, tz_convert, market hours |
| 12 | `12_merge_asof_pit.py`                     | Point-in-time joins (signals↔quotes), no look-ahead |
| 13 | `13_cross_sectional_ops.py`                | Per-date z-score/rank across assets, top/bottom quintile L/S |

## Conventions used

- Indices are timezone-naive UTC business days unless stated otherwise.
- All `np.random` calls are seeded with `42` for reproducibility.
- `axis=0` (column-wise) is the pandas default — used explicitly when ambiguity matters.
- Returns are simple (`pct_change`) unless the drill calls out log returns.

## References (latest docs)

- pandas user guide: <https://pandas.pydata.org/docs/user_guide/index.html>
- numpy user guide: <https://numpy.org/doc/stable/user/index.html>
- matplotlib tutorials: <https://matplotlib.org/stable/tutorials/index.html>
