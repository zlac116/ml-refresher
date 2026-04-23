"""Inject section-level hint cells.

For every `### Exercises ...` markdown cell, insert a 'Before you start —
techniques you'll use' markdown cell immediately after it, pulled from
`hints_data.HINTS` keyed by the header's first line.

Idempotent: if the next cell already starts with the hints marker
(`**Before you start`), it is overwritten with the latest content. This lets
you edit `hints_data.py` and re-run without duplicating cells.

Usage:
    python scripts/inject_hints.py <notebook.ipynb> [<notebook2.ipynb> ...]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nbformat

from hints_data import HINTS

HINTS_MARKER = "**Before you start"


def inject(nb_path: Path) -> dict:
    nb = nbformat.read(nb_path, as_version=4)
    inserted = 0
    overwritten = 0
    missing = []

    new_cells = list(nb.cells)
    i = 0
    while i < len(new_cells):
        c = new_cells[i]
        if c.cell_type == "markdown" and c.source.lstrip().startswith("###"):
            header_line = c.source.splitlines()[0].strip()
            if "Exercise" in header_line:
                hint_md = HINTS.get(header_line)
                if hint_md is None:
                    missing.append(header_line)
                else:
                    next_cell = new_cells[i + 1] if i + 1 < len(new_cells) else None
                    if (
                        next_cell is not None
                        and next_cell.cell_type == "markdown"
                        and next_cell.source.lstrip().startswith(HINTS_MARKER)
                    ):
                        next_cell.source = hint_md
                        overwritten += 1
                    else:
                        new_cells.insert(i + 1, nbformat.v4.new_markdown_cell(hint_md))
                        inserted += 1
                        i += 1  # skip over the one we just inserted
        i += 1

    nb.cells = new_cells
    nbformat.validate(nb)
    nbformat.write(nb, nb_path)

    return {
        "path": str(nb_path),
        "inserted": inserted,
        "overwritten": overwritten,
        "missing_keys": missing,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("notebooks", nargs="+", type=Path)
    args = ap.parse_args()

    rc = 0
    for p in args.notebooks:
        result = inject(p)
        print(
            f"[{p.name}] inserted={result['inserted']} "
            f"overwritten={result['overwritten']} "
            f"missing={len(result['missing_keys'])}"
        )
        for m in result["missing_keys"]:
            print(f"   MISSING HINT FOR: {m!r}")
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
