"""Merge broken <details> triples in the maths notebook into single markdown cells.

Each exercise currently has three cells:
    markdown: <details><summary>...</summary>
    code:     <solution code>
    markdown: </details>

Jupyter only honours <details> inside a single markdown cell — a sandwiched code
cell renders regardless of the tags. We merge each triple into one markdown cell
with the solution as a fenced python block inside the <details>.

Usage:
    python scripts/fix_maths_details.py <notebook.ipynb>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nbformat


def fix(nb_path: Path) -> dict:
    nb = nbformat.read(nb_path, as_version=4)

    new_cells: list = []
    i = 0
    n = len(nb.cells)
    merged = 0

    while i < n:
        c = nb.cells[i]
        is_open = (
            c.cell_type == "markdown"
            and c.source.lstrip().startswith("<details>")
            and "</details>" not in c.source
        )
        if (
            is_open
            and i + 2 < n
            and nb.cells[i + 1].cell_type == "code"
            and nb.cells[i + 2].cell_type == "markdown"
            and "</details>" in nb.cells[i + 2].source
        ):
            open_md = nb.cells[i].source.rstrip()
            code_src = nb.cells[i + 1].source.rstrip()
            close_md = nb.cells[i + 2].source.strip()

            body = (
                f"{open_md}\n\n"
                f"```python\n{code_src}\n```\n\n"
                f"{close_md}\n"
            )
            new_cells.append(nbformat.v4.new_markdown_cell(source=body))
            merged += 1
            i += 3
        else:
            new_cells.append(c)
            i += 1

    nb.cells = new_cells
    nbformat.validate(nb)
    nbformat.write(nb, nb_path)
    return {"merged": merged, "cells_before": n, "cells_after": len(new_cells)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("notebook", type=Path)
    args = ap.parse_args()
    result = fix(args.notebook)
    print(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
