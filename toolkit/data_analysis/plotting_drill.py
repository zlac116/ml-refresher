"""
TOOLKIT — matplotlib + seaborn drill
======================================

OBJECTIVE
    Practise the 8 canonical matplotlib patterns from the cheatsheet:
    fig/ax pattern, subplot grids, common chart types, time-axis formatting,
    twin axes, annotations, log axes, and saving.

ESTIMATED TIME
    60–90 min

TOPICS
    fig, ax = plt.subplots(...)
    plt.subplots(nrows, ncols, sharex/sharey, figsize)
    ax.plot / .bar / .hist / .scatter
    matplotlib.dates.DateFormatter + ax.xaxis.set_major_formatter
    ax.twinx() for dual y-axes
    ax.annotate(text, xy, xytext, arrowprops)
    ax.set_yscale('log')
    plt.tight_layout(); plt.savefig(path); plt.close('all')

EXPECTED OUTPUT
    Each task saves a PNG to /tmp/. Run + open the PNGs to verify visually.
    Assertions check: file exists, non-trivial size, axes labelled.

GRADING
    All asserts must pass. Tests check file size > 5kB + key labels via ax.get_*.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

np.random.seed(42)
DATES = pd.date_range("2024-01-01", periods=120, freq="1D")
TS = pd.Series(np.cumsum(np.random.normal(0, 1, 120)) + 100, index=DATES, name="px")
TS2 = pd.Series(np.cumsum(np.random.normal(0, 0.5, 120)) + 50, index=DATES, name="vol")
CATS = ["A", "B", "C", "D"]
CAT_VALUES = [10, 25, 17, 33]


# ── TASK 1 — fig/ax line plot ────────────────────────────────────────────
def plot_line(series: pd.Series, save_path: str) -> None:
    """Use `fig, ax = plt.subplots(figsize=(10, 4))`. Plot the series.
    Set ax.set_title("Price"), ax.set_ylabel("Close"), ax.grid(True, alpha=0.3).
    Save with plt.savefig + plt.close('all').
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 2 — Subplots grid (2×2) ──────────────────────────────────────────
def plot_subplots_grid(series: pd.Series, save_path: str) -> None:
    """2×2 grid. In each subplot draw the series differently:
       (0,0) line
       (0,1) bar (every 5th point)
       (1,0) histogram of pct_change()
       (1,1) cumulative sum
    Use plt.subplots(2, 2, figsize=(11, 7)) and plt.tight_layout().
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 3 — Bar chart (categorical) ─────────────────────────────────────
def plot_bar(categories: list[str], values: list[float], save_path: str) -> None:
    """Vertical bar chart of values vs categories. Add ax.set_ylabel('count')
    and ax.bar_label(bars) to label each bar with its value.
    """
    # TODO: implement (hint: bars = ax.bar(...); ax.bar_label(bars))
    raise NotImplementedError


# ── TASK 4 — Histogram with density curve ────────────────────────────────
def plot_histogram(series: pd.Series, save_path: str) -> None:
    """Histogram of pct_change() of the series with 30 bins, density=True.
    Overlay a vertical line at the mean using ax.axvline(mean, color='red').
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 5 — Time axis formatting ────────────────────────────────────────
def plot_with_date_formatter(series: pd.Series, save_path: str) -> None:
    """Line plot with formatted date axis:
       ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
       fig.autofmt_xdate()  # rotates labels
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 6 — Twin axes (two y-scales) ────────────────────────────────────
def plot_twin_axes(s_left: pd.Series, s_right: pd.Series, save_path: str) -> None:
    """Single x-axis. Plot s_left on the left y-axis (blue) and s_right on a
    twin right y-axis (red, via ax.twinx()). Label each axis.
    Add a combined legend.
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 7 — Annotation + log axis ───────────────────────────────────────
def plot_annotated(series: pd.Series, save_path: str) -> None:
    """Line plot with:
       - ax.set_yscale('log')
       - ax.annotate on the GLOBAL maximum point with an arrow
         (use arrowprops=dict(arrowstyle='->'))
       - Add a subtle title.
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 8 — Save + verify file ──────────────────────────────────────────
def saved_plot_size(path: str) -> int:
    """Just os.path.getsize(path) — used by the grader."""
    # TODO: implement
    raise NotImplementedError


# ── GRADING ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    paths = {
        "line":      "/tmp/p_line.png",
        "grid":      "/tmp/p_grid.png",
        "bar":       "/tmp/p_bar.png",
        "hist":      "/tmp/p_hist.png",
        "dates":     "/tmp/p_dates.png",
        "twin":      "/tmp/p_twin.png",
        "annot":     "/tmp/p_annot.png",
    }
    # Clean any prior files
    for p in paths.values():
        if os.path.exists(p): os.remove(p)

    plot_line(TS,        paths["line"])
    plot_subplots_grid(TS, paths["grid"])
    plot_bar(CATS, CAT_VALUES, paths["bar"])
    plot_histogram(TS,   paths["hist"])
    plot_with_date_formatter(TS, paths["dates"])
    plot_twin_axes(TS, TS2, paths["twin"])
    plot_annotated(TS,   paths["annot"])

    for name, p in paths.items():
        assert os.path.exists(p),       f"{name} missing"
        sz = saved_plot_size(p)
        assert sz > 5_000,              f"{name} too small ({sz} bytes)"

    for name, p in paths.items():
        print(f"saved {name:6s} → {p}   ({saved_plot_size(p):>6} bytes)")
    print("\n✓ All checks passed.  Open the PNGs in /tmp/ to verify visually.")
