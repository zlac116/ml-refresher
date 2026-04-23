"""Transform exercise notebooks in place.

For every prompt → code → <details> triple:
  1. Capture the code cell's outputs (execute if missing).
  2. Stub the code cell source to just the leading comment block
     (`# Your answer here` + any `# Hint: ...` lines).
  3. Insert a new markdown "Expected output" cell between the code
     cell and the <details> reveal.

Usage:
    python scripts/transform_exercises.py <notebook.ipynb> [--no-execute]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient


EXERCISE_RE = re.compile(r"\s*\*\*Exercise\s")
DETAILS_RE = re.compile(r"^\s*<details>")
EXPECTED_HEADER = "**Expected output:**"
DETAILS_PY_RE = re.compile(r"```python\n(.+?)\n```", re.DOTALL)


def extract_details_python(details_source: str) -> str | None:
    """Pull the ```python ... ``` block out of a <details> solution cell."""
    m = DETAILS_PY_RE.search(details_source)
    return m.group(1) if m else None


def answer_is_stub(source: str) -> bool:
    """True if the answer cell has only comment lines (no executable code)."""
    for ln in source.splitlines():
        s = ln.strip()
        if s and not s.startswith("#"):
            return False
    return True


def is_prompt(cell) -> bool:
    return cell.cell_type == "markdown" and bool(EXERCISE_RE.match(cell.source))


def is_details(cell) -> bool:
    return cell.cell_type == "markdown" and bool(DETAILS_RE.match(cell.source[:50]))


def is_expected(cell) -> bool:
    return cell.cell_type == "markdown" and cell.source.startswith(EXPECTED_HEADER)


def find_triples(cells):
    """Locate exercise units.

    Returns a list of (prompt_idx, answer_idx, details_idx) tuples.

    Pattern: `prompt → code+ → details`. The **answer** cell is the last
    code cell before the <details> (the one containing `# Your answer here`).
    Any scratch code cells between the prompt and the answer cell are left
    untouched — only the answer cell gets stubbed.

    Already-transformed units are skipped: if an `**Expected output:**`
    markdown cell sits between the last code cell and the details, that
    unit is treated as done.
    """
    triples = []
    i = 0
    n = len(cells)
    while i < n:
        if is_prompt(cells[i]):
            # Walk forward past any code cells.
            j = i + 1
            while j < n and cells[j].cell_type == "code":
                j += 1
            if j == i + 1:
                i += 1
                continue
            # Optional single Expected-output markdown cell already present.
            if j < n and is_expected(cells[j]):
                i = j + 1  # already transformed; skip
                continue
            if j < n and is_details(cells[j]):
                answer_idx = j - 1
                details_idx = j
                triples.append((i, answer_idx, details_idx))
                i = details_idx + 1
                continue
        i += 1
    return triples


def stub_source(source: str) -> str:
    """Keep `# Your answer here` and any immediately-following `# Hint:` lines.

    Drops solution code that follows the first blank line.
    """
    kept = []
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            kept.append(line.rstrip())
        elif stripped == "" and kept:
            break
        elif stripped == "":
            continue
        else:
            break
    if not kept or not any("your answer here" in k.lower() for k in kept):
        kept.insert(0, "# Your answer here")
    return "\n".join(kept) + "\n"


def _join(value):
    return "".join(value) if isinstance(value, list) else value


def outputs_to_markdown(outputs) -> str:
    """Render a code cell's outputs as a markdown cell body."""
    parts = [EXPECTED_HEADER, ""]
    any_body = False
    for out in outputs:
        ot = out.get("output_type")
        if ot == "stream":
            text = _join(out.get("text", "")).rstrip()
            if text:
                parts.append("```text")
                parts.append(text)
                parts.append("```")
                any_body = True
        elif ot in ("execute_result", "display_data"):
            data = out.get("data", {}) or {}
            if "text/html" in data:
                html = _join(data["text/html"]).rstrip()
                if html:
                    parts.append(html)
                    any_body = True
            elif "image/png" in data:
                img = _join(data["image/png"]).strip()
                parts.append(f'<img src="data:image/png;base64,{img}" alt="expected plot" />')
                any_body = True
            elif "text/plain" in data:
                text = _join(data["text/plain"]).rstrip()
                if text:
                    parts.append("```text")
                    parts.append(text)
                    parts.append("```")
                    any_body = True
        elif ot == "error":
            ename = out.get("ename", "")
            evalue = out.get("evalue", "")
            parts.append(
                "> ⚠️ _The reference solution in the `<details>` block currently errors "
                f"(`{ename}: {evalue}`). Treat the details block's commentary as "
                "the reference, not any output shown here._"
            )
            any_body = True
    if not any_body:
        parts.append("_(no output produced by the reference solution)_")
    return "\n".join(parts)


def execute_notebook(nb, cwd: Path) -> None:
    """Run all code cells in place so exercise answers have fresh outputs.

    `allow_errors=True` — a failing cell doesn't abort the run, so later
    exercises can still capture outputs. Errored cells get their error
    rendered as the Expected Output (which is at least an honest signal).
    """
    client = NotebookClient(
        nb,
        timeout=3600,
        kernel_name="python3",
        resources={"metadata": {"path": str(cwd)}},
        allow_errors=True,
    )
    client.execute()


def transform(nb_path: Path, execute: bool) -> dict:
    nb = nbformat.read(nb_path, as_version=4)

    triples_before = find_triples(nb.cells)
    missing = sum(
        1 for (_p, a, _d) in triples_before if not nb.cells[a].get("outputs")
    )
    # For any answer cell that is just a stub (no executable code), splice in
    # the canonical solution from the <details> reveal so execution actually
    # produces an output. We remember which indices we patched so we can
    # assert the stub back after execution.
    patched = []
    for _p, a, d in triples_before:
        if answer_is_stub(nb.cells[a].source):
            details_py = extract_details_python(nb.cells[d].source)
            if details_py:
                patched.append((a, nb.cells[a].source))
                nb.cells[a].source = details_py

    print(
        f"[{nb_path.name}] {len(triples_before)} exercises, "
        f"{missing} missing outputs, "
        f"{len(patched)} stub cells patched with canonical solution."
    )

    if execute:
        print(f"[{nb_path.name}] Executing notebook to capture outputs...")
        execute_notebook(nb, cwd=nb_path.parent)
        print(f"[{nb_path.name}] Execution complete.")

    # Restore original stub sources so transform sees them as empty
    # (we'll still have the outputs attached from the patched run).
    for idx, original in patched:
        nb.cells[idx].source = original

    # Re-scan after execution in case cell indices shifted (they shouldn't).
    triples = find_triples(nb.cells)
    # Map answer_idx → (prompt_idx, details_idx) for fast membership.
    answers = {a: (p, d) for (p, a, d) in triples}

    new_cells = []
    i = 0
    ex_count = 0
    n = len(nb.cells)
    while i < n:
        if i in answers:
            code = nb.cells[i]
            details_idx = answers[i][1]

            outputs = code.get("outputs", []) or []
            expected_md = outputs_to_markdown(outputs)
            stub = stub_source(code.source)

            new_code = nbformat.v4.new_code_cell(source=stub)
            new_code.metadata = code.metadata  # preserve any tags
            expected_cell = nbformat.v4.new_markdown_cell(source=expected_md)

            new_cells.append(new_code)
            new_cells.append(expected_cell)
            # Emit the details cell, then jump past it.
            new_cells.append(nb.cells[details_idx])
            i = details_idx + 1
            ex_count += 1
        else:
            new_cells.append(nb.cells[i])
            i += 1

    nb.cells = new_cells
    nbformat.validate(nb)
    nbformat.write(nb, nb_path)

    return {
        "path": str(nb_path),
        "exercises": len(triples_before),
        "transformed": ex_count,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("notebook", type=Path)
    ap.add_argument(
        "--no-execute",
        action="store_true",
        help="Skip kernel execution (use existing cached outputs only).",
    )
    args = ap.parse_args()

    result = transform(args.notebook, execute=not args.no_execute)
    print(f"Result: {result}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
