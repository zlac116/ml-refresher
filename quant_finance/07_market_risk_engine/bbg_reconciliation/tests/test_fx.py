"""FX swaps & forwards reconciliation vs Bloomberg.

Key axes: FX spot shock, each-leg discounting, forward points / xccy basis.
Watch quote direction (domestic-per-foreign) — an inverted convention flips the
sign of the impact and shows as a large false break.
"""

from conftest import assert_within_tolerance

TRADE_TYPE = ["fx_swap", "fx_forward"]


def test_reconciles(recon_row):
    assert_within_tolerance(recon_row)
