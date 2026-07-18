"""Shared pytest fixtures.

The recon frame is built ONCE per session (load in-house snapshot + BBG export,
merge, classify) and shared across every test module. Each trade-type test just
filters this frame and asserts row-by-row within tolerance.
"""

import functools

import pandas as pd
import pytest

from recon import adapters, config, reconcile, schema


@functools.lru_cache(maxsize=1)
def _load_recon() -> pd.DataFrame:
    """Build the classified recon frame ONCE (cached for both collection-time
    parametrisation and the session fixture). Returns empty frame if no data."""
    inhouse = adapters.load_inhouse()
    bbg = adapters.load_bbg()
    if inhouse.empty or bbg.empty:
        return schema.empty_canonical()
    return reconcile.run(inhouse, bbg, config.load_tolerances())


def pytest_generate_tests(metafunc):
    """Parametrise any test requesting `recon_row` with that module's rows.

    A test module sets module-global `TRADE_TYPE = "irs"` and takes a
    `recon_row` argument; this generates one test per (trade, scenario, metric)
    row for that type — granular pass/fail in the pytest report.
    """
    if "recon_row" not in metafunc.fixturenames:
        return
    trade_type = getattr(metafunc.module, "TRADE_TYPE", None)
    types = [trade_type] if isinstance(trade_type, str) else list(trade_type or [])
    recon = _load_recon()
    sub = recon[recon[schema.TRADE_TYPE].isin(types)] if not recon.empty else recon
    params = [
        pytest.param(
            row,
            id=f"{row[schema.TRADE_ID]}|{row[schema.SCENARIO_ID]}|{row[schema.METRIC]}",
        )
        for _, row in sub.iterrows()
    ]
    if not params:
        params = [pytest.param(None, id="no-data", marks=pytest.mark.skip(reason="no recon rows"))]
    metafunc.parametrize("recon_row", params)


@pytest.fixture(scope="session")
def recon() -> pd.DataFrame:
    """The classified reconciliation frame — the single source of truth."""
    df = _load_recon()
    if df.empty:
        pytest.skip(
            "No data: drop data/inhouse/inhouse_canonical.csv and "
            "data/bbg/bbg_canonical.csv (see README) to enable the pack."
        )
    return df


def assert_within_tolerance(row: pd.Series) -> None:
    """The single assertion every trade-type test funnels through."""
    status = row["status"]
    if status == reconcile.NA:
        pytest.fail(
            f"COVERAGE break — {row[schema.TRADE_ID]} {row[schema.SCENARIO_ID]} "
            f"{row[schema.METRIC]}: present in only one source "
            f"(inhouse={row['inhouse']}, bbg={row['bbg']})."
        )
    msg = (
        f"{status}: {row[schema.TRADE_ID]} {row[schema.SCENARIO_ID]} {row[schema.METRIC]} "
        f"inhouse={row['inhouse']:,.2f} bbg={row['bbg']:,.2f} "
        f"diff={row['diff_abs']:,.2f} ({row['diff_bp']:.2f}bp) "
        f"vs fail>{row['threshold']}{row['tol_basis']}"
    )
    # WARN does not fail the build but is surfaced; FAIL fails.
    assert status != reconcile.FAIL, msg
    if status == reconcile.WARN:
        import warnings
        warnings.warn(msg)
