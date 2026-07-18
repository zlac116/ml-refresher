"""Generate the reconciliation pack outside pytest (for ad-hoc / CI runs).

    python run.py                 # uses data/{inhouse,bbg}/*_canonical.csv
    python run.py --as-of 2026-07-18

Emits data/output/recon_{detail,summary}.csv and recon_pack.html. The pytest
suite (`pytest`) is the pass/fail gate; this is the same computation for when you
just want the artefacts.
"""

import argparse

from recon import adapters, config, reconcile, report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-of", default="2026-07-18", help="valuation date YYYY-MM-DD")
    args = ap.parse_args()

    inhouse = adapters.load_inhouse()
    bbg = adapters.load_bbg()
    if inhouse.empty or bbg.empty:
        print("No data — populate data/inhouse and data/bbg (see README).")
        return 1

    recon = reconcile.run(inhouse, bbg, config.load_tolerances())
    paths = report.write_pack(recon, as_of=args.as_of)

    summ = reconcile.summary(recon)
    print(summ.to_string(index=False))
    n_fail = (recon["status"] == reconcile.FAIL).sum()
    print(f"\nPack written: {paths['html']}")
    print(f"FAILs: {n_fail}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
