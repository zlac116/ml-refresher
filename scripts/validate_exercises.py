"""Validate that a notebook's exercises follow the 4-cell pattern:

    prompt  →  stub code  →  expected output  →  <details> solution

Checks performed:
  1. Every `**Exercise ...` prompt is followed by: code, markdown (Expected), markdown (<details>).
  2. Every stub code cell contains no executable code — only comment lines.
  3. Every Expected-output cell starts with `**Expected output:**` and has at least one body line.
  4. The <details> block is intact (contains `</details>` closer).
  5. nbformat.validate() passes.

Usage:
    python scripts/validate_exercises.py <notebook.ipynb>
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import nbformat


EXERCISE_RE = re.compile(r"\s*\*\*Exercise\s([0-9.]+)")
DETAILS_RE = re.compile(r"^\s*<details>")
EXPECTED_HEADER = "**Expected output:**"


def validate(nb_path: Path) -> int:
    nb = nbformat.read(nb_path, as_version=4)
    try:
        nbformat.validate(nb)
    except Exception as e:
        print(f"FAIL nbformat.validate: {e}")
        return 1

    problems = []
    exercises_found = []

    i = 0
    n = len(nb.cells)
    while i < n:
        c = nb.cells[i]
        if c.cell_type == "markdown":
            m = EXERCISE_RE.match(c.source)
            if m:
                ex_id = m.group(1)
                exercises_found.append(ex_id)

                # Walk forward past any code cells (scratch cells allowed).
                j = i + 1
                while j < n and nb.cells[j].cell_type == "code":
                    j += 1
                if j == i + 1 or j >= n:
                    problems.append(f"Exercise {ex_id}: no code cell after prompt")
                    i += 1
                    continue

                answer = nb.cells[j - 1]  # last code cell — the stub
                expected = nb.cells[j] if j < n else None
                details = nb.cells[j + 1] if j + 1 < n else None

                code_lines = [
                    ln for ln in answer.source.splitlines() if ln.strip()
                ]
                non_comment = [
                    ln for ln in code_lines if not ln.lstrip().startswith("#")
                ]
                if non_comment:
                    problems.append(
                        f"Exercise {ex_id}: stub code cell still has executable lines: "
                        f"{non_comment[:2]}"
                    )
                if not any("your answer here" in ln.lower() for ln in code_lines):
                    problems.append(
                        f"Exercise {ex_id}: stub code cell missing '# Your answer here'"
                    )

                if (
                    expected is None
                    or expected.cell_type != "markdown"
                    or not expected.source.startswith(EXPECTED_HEADER)
                ):
                    problems.append(
                        f"Exercise {ex_id}: expected-output cell missing or malformed"
                    )
                else:
                    body = expected.source[len(EXPECTED_HEADER):].strip()
                    if not body:
                        problems.append(f"Exercise {ex_id}: expected-output cell is empty")

                if (
                    details is None
                    or details.cell_type != "markdown"
                    or not DETAILS_RE.match(details.source[:50])
                ):
                    problems.append(
                        f"Exercise {ex_id}: <details> solution cell missing"
                    )
                elif "</details>" not in details.source:
                    problems.append(
                        f"Exercise {ex_id}: <details> block not closed"
                    )

                i = j + 2
                continue
        i += 1

    print(f"[{nb_path.name}] exercises found: {len(exercises_found)}")
    if problems:
        print(f"[{nb_path.name}] FAIL — {len(problems)} problems:")
        for p in problems[:20]:
            print(f"  - {p}")
        if len(problems) > 20:
            print(f"  ... (+{len(problems) - 20} more)")
        return 1

    print(f"[{nb_path.name}] OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("notebooks", nargs="+", type=Path)
    args = ap.parse_args()

    rc = 0
    for p in args.notebooks:
        rc |= validate(p)
    return rc


if __name__ == "__main__":
    sys.exit(main())
