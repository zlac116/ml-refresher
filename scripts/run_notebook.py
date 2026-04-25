"""Execute a notebook end-to-end and write outputs back in place.

Used to populate the cached outputs for newly-inserted Worked Example cells
(and refresh stale outputs of any other code cell). Errors do not abort
execution — affected cells will carry their traceback as the cached output.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import nbformat
from nbclient import NotebookClient


def run(nb_path: Path) -> dict:
    nb = nbformat.read(nb_path, as_version=4)
    NotebookClient(
        nb,
        timeout=3600,
        kernel_name="python3",
        resources={"metadata": {"path": str(nb_path.parent)}},
        allow_errors=True,
    ).execute()
    nbformat.write(nb, nb_path)

    n_err = sum(
        1
        for c in nb.cells
        if c.cell_type == "code"
        for o in c.get("outputs", [])
        if o.get("output_type") == "error"
    )
    n_code = sum(1 for c in nb.cells if c.cell_type == "code")
    return {"path": str(nb_path), "code_cells": n_code, "errored_cells": n_err}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("notebooks", nargs="+", type=Path)
    args = ap.parse_args()
    for p in args.notebooks:
        r = run(p)
        print(f"[{p.name}] {r['code_cells']} code cells, {r['errored_cells']} errored")
    return 0


if __name__ == "__main__":
    sys.exit(main())
