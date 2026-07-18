"""Whole-pack gate + artefact emission.

Runs after the per-trade-type tests conceptually: builds the summary, WRITES the
pack (CSV + HTML) to data/output/, and asserts the overall gate — zero FAILs.
This is the test whose green tick you attach to the validation ticket.
"""

from recon import report, reconcile

# Valuation date is passed in, never generated inside the engine, so the pack is
# reproducible. TODO(you): source this from the run manifest / engine as-of.
AS_OF = "2026-07-18"


def test_emit_pack_and_gate(recon):
    paths = report.write_pack(recon, as_of=AS_OF)
    n_fail = (recon["status"] == reconcile.FAIL).sum()
    n_na = (recon["status"] == reconcile.NA).sum()
    assert n_fail == 0, (
        f"{n_fail} FAIL break(s) — see {paths['html']} and {paths['detail']}"
    )
    assert n_na == 0, (
        f"{n_na} coverage break(s): trades present in only one source — see {paths['html']}"
    )
