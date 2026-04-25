"""Inject a 'Worked example' tutorial cell pair after each section's hint block.

Layout per section after injection:

    ### Exercises — Section N           (untouched)
    **Before you start ...**            (untouched, hint block)
    **Worked example — ...**            (NEW — markdown intro)
    <runnable code demonstrating the techniques>   (NEW — code cell, executed)
    **Exercise N.1 — ...**              (untouched)
    ...

Idempotent: if a "Worked example" markdown cell already follows the hint block,
its content (and the following code cell) is overwritten with the latest version
from worked_examples_data.WORKED_EXAMPLES.

Usage:
    python scripts/inject_worked_examples.py <notebook.ipynb> [<more.ipynb> ...]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nbformat

from worked_examples_data import WORKED_EXAMPLES

HINT_MARKER = "**Before you start"
EXAMPLE_MARKER = "**Worked example"


def inject(nb_path: Path) -> dict:
    nb = nbformat.read(nb_path, as_version=4)
    inserted = 0
    overwritten = 0
    missing: list[str] = []
    cells = list(nb.cells)

    i = 0
    while i < len(cells):
        c = cells[i]
        if c.cell_type == "markdown" and c.source.lstrip().startswith("###"):
            header_line = c.source.splitlines()[0].strip()
            if "Exercise" in header_line:
                entry = WORKED_EXAMPLES.get(header_line)
                if entry is None:
                    missing.append(header_line)
                    i += 1
                    continue

                # Find the position right after the hint block (which sits at i+1).
                # If the hint isn't there for some reason, fall back to inserting
                # immediately after the header.
                target = i + 1
                if target < len(cells) and cells[target].cell_type == "markdown" and cells[target].source.lstrip().startswith(HINT_MARKER):
                    target += 1

                intro_md = entry["intro"].rstrip() + "\n"
                code_src = entry["code"].rstrip() + "\n"

                existing_intro = cells[target] if target < len(cells) else None
                if (
                    existing_intro is not None
                    and existing_intro.cell_type == "markdown"
                    and existing_intro.source.lstrip().startswith(EXAMPLE_MARKER)
                ):
                    # Already-injected pair — overwrite both cells in place.
                    existing_intro.source = intro_md
                    existing_code = cells[target + 1] if target + 1 < len(cells) else None
                    if existing_code is not None and existing_code.cell_type == "code":
                        existing_code.source = code_src
                        # Clear stale outputs so they get re-populated on next execute.
                        existing_code["outputs"] = []
                        existing_code["execution_count"] = None
                    overwritten += 1
                else:
                    cells.insert(target, nbformat.v4.new_markdown_cell(source=intro_md))
                    cells.insert(target + 1, nbformat.v4.new_code_cell(source=code_src))
                    inserted += 1
                    # Skip past the two cells we just inserted.
                    i = target + 1
        i += 1

    nb.cells = cells
    nbformat.validate(nb)
    nbformat.write(nb, nb_path)
    return {"path": str(nb_path), "inserted": inserted, "overwritten": overwritten, "missing": missing}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("notebooks", nargs="+", type=Path)
    args = ap.parse_args()
    rc = 0
    for p in args.notebooks:
        r = inject(p)
        print(f"[{p.name}] inserted={r['inserted']} overwritten={r['overwritten']} missing={len(r['missing'])}")
        for m in r["missing"]:
            print(f"   MISSING: {m!r}")
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
