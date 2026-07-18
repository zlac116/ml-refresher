"""Emit the reconciliation PACK — the deliverable handed to Model Validation.

Three artefacts, one call:
  1. recon_detail.csv   — every (trade, scenario, metric) row, full audit trail.
  2. recon_summary.csv  — pass-rate matrix by trade_type x metric.
  3. recon_pack.html    — self-contained, RAG-status pack (front sheet +
                          breaks table + full detail). No external deps.

The HTML is deliberately self-contained (inline CSS) so it can be attached to a
validation ticket / emailed / archived as evidence without a server.
"""

from datetime import datetime
from pathlib import Path

import pandas as pd

from . import reconcile, schema

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "output"

_STATUS_COLOR = {
    reconcile.PASS: "#1a7f37",
    reconcile.WARN: "#9a6700",
    reconcile.FAIL: "#cf222e",
    reconcile.NA: "#57606a",
}


def write_pack(recon: pd.DataFrame, *, as_of: str, out_dir: Path | None = None) -> dict:
    """Write the three artefacts. `as_of` is the valuation date (stamp it, don't
    call Date.now inside the engine — pass it in for reproducibility)."""
    out_dir = out_dir or OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    summ = reconcile.summary(recon)
    breaks = recon[recon["status"].isin([reconcile.FAIL, reconcile.WARN, reconcile.NA])]

    detail_path = out_dir / "recon_detail.csv"
    summary_path = out_dir / "recon_summary.csv"
    html_path = out_dir / "recon_pack.html"

    recon.to_csv(detail_path, index=False)
    summ.to_csv(summary_path, index=False)
    html_path.write_text(_render_html(recon, summ, breaks, as_of=as_of))

    return {"detail": detail_path, "summary": summary_path, "html": html_path}


def _status_badge(s: str) -> str:
    return f'<span style="color:{_STATUS_COLOR.get(s, "#000")};font-weight:600">{s}</span>'


def _df_to_html(df: pd.DataFrame) -> str:
    d = df.copy()
    if "status" in d.columns:
        d["status"] = d["status"].map(_status_badge)
    for c in ("inhouse", "bbg", "diff_abs", "diff_bp", "diff_rel", "threshold", "max_abs_bp"):
        if c in d.columns:
            d[c] = d[c].map(lambda x: "" if pd.isna(x) else f"{x:,.4g}")
    return d.to_html(index=False, escape=False, border=0)


def _render_html(recon, summ, breaks, *, as_of: str) -> str:
    n = len(recon)
    n_fail = (recon["status"] == reconcile.FAIL).sum()
    n_warn = (recon["status"] == reconcile.WARN).sum()
    n_na = (recon["status"] == reconcile.NA).sum()
    n_pass = (recon["status"] == reconcile.PASS).sum()
    overall = "FAIL" if n_fail else ("WARN" if n_warn or n_na else "PASS")
    generated = datetime.strptime(as_of, "%Y-%m-%d").strftime("%Y-%m-%d")

    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>BBG Reconciliation Pack — {generated}</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:2rem;color:#1f2328}}
 h1{{font-size:1.4rem}} h2{{font-size:1.1rem;margin-top:2rem;border-bottom:1px solid #d0d7de;padding-bottom:.3rem}}
 table{{border-collapse:collapse;font-size:.82rem;margin-top:.5rem}}
 th,td{{padding:.3rem .6rem;text-align:right;border-bottom:1px solid #eaecef}}
 th{{background:#f6f8fa;text-align:right}} td:first-child,th:first-child{{text-align:left}}
 .kpi{{display:inline-block;padding:.6rem 1rem;margin-right:.6rem;border-radius:6px;background:#f6f8fa}}
 .big{{font-size:1.6rem;font-weight:700}}
 .banner{{padding:.6rem 1rem;border-radius:6px;font-weight:700;color:#fff;display:inline-block}}
</style></head><body>
<h1>Treasury Valuation — Bloomberg Reconciliation Pack</h1>
<p>Valuation date: <b>{generated}</b> &nbsp;|&nbsp; Independent benchmark: <b>Bloomberg</b>
 &nbsp;|&nbsp; Scope: base PV + stress impacts across all trade types.</p>
<p class="banner" style="background:{_STATUS_COLOR[overall.replace('PASS','PASS')]}">OVERALL: {overall}</p>
<div style="margin-top:1rem">
 <span class="kpi"><div class="big">{n}</div>rows</span>
 <span class="kpi"><div class="big" style="color:{_STATUS_COLOR[reconcile.PASS]}">{n_pass}</div>pass</span>
 <span class="kpi"><div class="big" style="color:{_STATUS_COLOR[reconcile.WARN]}">{n_warn}</div>warn</span>
 <span class="kpi"><div class="big" style="color:{_STATUS_COLOR[reconcile.FAIL]}">{n_fail}</div>fail</span>
 <span class="kpi"><div class="big" style="color:{_STATUS_COLOR[reconcile.NA]}">{n_na}</div>n/a</span>
</div>
<h2>1. Pass-rate matrix (trade type × metric)</h2>
{_df_to_html(summ)}
<h2>2. Breaks &amp; warnings (exceptions only)</h2>
{_df_to_html(breaks) if len(breaks) else "<p><i>No breaks or warnings.</i></p>"}
<h2>3. Full detail</h2>
{_df_to_html(recon)}
<p style="color:#57606a;font-size:.75rem;margin-top:2rem">
 Model-risk note: this pack is benchmarking / outcomes-analysis evidence (SR 11-7 /
 PRA SS1/23). It reconciles the in-house engine to Bloomberg on frozen inputs for a
 sample of trades; it is not a full independent revaluation. Breaks are classified by
 tolerance, not root-caused — see the RCA workflow in the README.</p>
</body></html>"""
