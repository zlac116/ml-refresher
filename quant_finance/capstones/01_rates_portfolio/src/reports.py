"""Reporting helpers — HTML / CSV table generators + waterfall chart.

Minimal but standardised so the user's notebooks all produce reports in the same
style. Real banks use Jinja2 templates feeding into a corporate stylesheet; for
the capstone we keep it pandas-to-HTML with a tiny inline style.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd


_CSS = """
body { font-family: -apple-system, Helvetica, Arial, sans-serif; max-width: 1200px;
       margin: 2em auto; color: #222; }
h1, h2 { color: #333; border-bottom: 2px solid #ccc; padding-bottom: 0.3em; }
table { border-collapse: collapse; margin: 1em 0; font-size: 0.9em; }
th, td { padding: 0.4em 0.8em; border: 1px solid #ddd; text-align: right; }
th { background: #f5f5f5; }
td:first-child, th:first-child { text-align: left; }
.highlight { background: #fff3cd; }
.break { background: #f8d7da; }
.ok { background: #d4edda; }
.tag { font-family: monospace; font-size: 0.85em; padding: 0.1em 0.3em; background: #eef; border-radius: 3px; }
"""


def _html_head(title: str) -> str:
    """Build the HTML head — kept separate from CSS so str.format doesn't see {} braces."""
    return (
        f"<!DOCTYPE html>\n<html><head><meta charset='utf-8'><title>{title}</title>\n"
        f"<style>{_CSS}</style></head><body>\n"
    )


def _html_footer() -> str:
    return "</body></html>"


def html_report(title: str, sections: list[tuple[str, pd.DataFrame | str]]) -> str:
    """Build a multi-section HTML report.

    sections: list of (heading, content) — content can be a DataFrame (auto-rendered
    as a table) or a raw HTML/Markdown string.
    """
    out = io.StringIO()
    out.write(_html_head(title))
    out.write(f"<h1>{title}</h1>\n")
    for heading, content in sections:
        out.write(f"<h2>{heading}</h2>\n")
        if isinstance(content, pd.DataFrame):
            out.write(content.to_html(index=False, escape=False, float_format=lambda x: f"{x:,.4f}"))
        else:
            out.write(content)
        out.write("\n")
    out.write(_html_footer())
    return out.getvalue()


def write_html_report(path: str | Path, title: str, sections: list[tuple[str, pd.DataFrame | str]]) -> None:
    """Save an HTML report to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html_report(title, sections))


def write_csv(path: str | Path, df: pd.DataFrame) -> None:
    """Save a DataFrame to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def waterfall_chart(buckets: list[str], values: list[float], title: str = "P&L Attribution") -> str:
    """Render a P&L attribution waterfall chart as an inline-PNG <img> tag.

    Bars stack on top of each other; a final 'TOTAL' bar shows the running sum.
    Returns HTML embeddable inline.
    """
    import base64

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Append a final 'TOTAL' bar for readability
    total = float(sum(values))
    plot_buckets = list(buckets) + ["TOTAL"]
    plot_values = list(values) + [total]
    cum = np.concatenate([[0.0], np.cumsum(values)])
    bottoms = list(cum[:-1]) + [0.0]   # last bar (TOTAL) starts from zero
    bar_values = list(values) + [total]
    colors = ["#4caf50" if v >= 0 else "#e53935" for v in values] + ["#1976d2"]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(plot_buckets, bar_values, bottom=bottoms, color=colors, edgecolor="black")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title(title)
    ax.set_ylabel("$ P&L")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f'<img src="data:image/png;base64,{b64}" style="max-width:100%;height:auto"/>'
