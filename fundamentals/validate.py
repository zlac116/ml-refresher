"""Audit a fundamentals notebook for correctness.

For every exercise:
  1. Find the student cell and the <details> solution code.
  2. Replace student function bodies with the solution's same-named functions
     (AST surgery), keeping the student's check() calls and setup lines.
  3. Execute the notebook in a fresh kernel.
  4. Count ✅ / ❌ from check() output.

Reports any failures so they can be fixed before the notebook ships.

Usage:
    python fundamentals/validate.py <notebook.ipynb> [<more.ipynb> ...]
"""
from __future__ import annotations
import argparse, ast, copy, re, sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient


SOL_RE = re.compile(r"```python\n(.+?)\n```", re.DOTALL)
EX_RE = re.compile(r"^\*\*Exercise\s([0-9.]+)", re.DOTALL)


def find_exercises(nb):
    out = []
    for i, c in enumerate(nb.cells):
        if c.cell_type != "markdown":
            continue
        m = EX_RE.match(c.source.strip()[:80])
        if not m:
            continue
        if i + 2 >= len(nb.cells):
            continue
        if nb.cells[i + 1].cell_type != "code":
            continue
        if nb.cells[i + 2].cell_type != "markdown":
            continue
        if "<details>" not in nb.cells[i + 2].source:
            continue
        sm = SOL_RE.search(nb.cells[i + 2].source)
        if not sm:
            continue
        out.append({
            "id": m.group(1),
            "student_idx": i + 1,
            "student_src": nb.cells[i + 1].source,
            "solution_src": sm.group(1),
        })
    return out


def merge(student_src: str, solution_src: str) -> str:
    """Replace student's `def f: pass` with the solution's same-named def(s);
    keep all other student lines (check calls, setup variables) intact."""
    st = ast.parse(student_src)
    sl = ast.parse(solution_src)
    sol_funcs = {n.name: n for n in sl.body if isinstance(n, ast.FunctionDef)}
    new_body = []
    replaced = 0
    for n in st.body:
        if isinstance(n, ast.FunctionDef) and n.name in sol_funcs:
            new_body.append(sol_funcs[n.name])
            replaced += 1
        else:
            new_body.append(n)
    if replaced == 0:
        # Student cell has no function defs; prepend ALL solution top-level statements
        # (definitions and any helpers) before the student's runnable code.
        new_body = list(sl.body) + new_body
    st.body = new_body
    return ast.unparse(st)


def audit(nb_path: Path) -> dict:
    nb = nbformat.read(nb_path, as_version=4)
    exercises = find_exercises(nb)

    test = copy.deepcopy(nb)
    for ex in exercises:
        try:
            test.cells[ex["student_idx"]].source = merge(ex["student_src"], ex["solution_src"])
        except SyntaxError as e:
            print(f"  syntax error merging Ex {ex['id']}: {e}")

    NotebookClient(
        test, timeout=120, kernel_name="python3",
        resources={"metadata": {"path": str(nb_path.parent)}},
        allow_errors=True,
    ).execute()

    idx2id = {ex["student_idx"]: ex["id"] for ex in exercises}
    passes = 0
    fails = []
    errors = []
    for i, c in enumerate(test.cells):
        if c.cell_type != "code":
            continue
        ex_id = idx2id.get(i, "-")
        for o in c.get("outputs", []):
            ot = o.get("output_type")
            if ot == "error":
                errors.append((ex_id, i, o.get("ename"), str(o.get("evalue"))[:120]))
            elif ot == "stream":
                t = o.get("text", "")
                if isinstance(t, list):
                    t = "".join(t)
                for ln in t.splitlines():
                    if "✅" in ln:
                        passes += 1
                    elif "❌" in ln:
                        fails.append((ex_id, ln.strip()))

    return {
        "exercises": len(exercises),
        "passes": passes,
        "fails": fails,
        "errors": errors,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("notebooks", nargs="+", type=Path)
    args = ap.parse_args()
    rc = 0
    for p in args.notebooks:
        print(f"\n=== {p.name} ===")
        r = audit(p)
        print(f"exercises: {r['exercises']}    checks passing: {r['passes']}")
        if r["fails"]:
            print(f"failing checks: {len(r['fails'])}")
            for ex_id, line in r["fails"]:
                print(f"  Ex {ex_id}: {line}")
            rc = 1
        if r["errors"]:
            print(f"runtime errors: {len(r['errors'])}")
            for ex_id, i, en, ev in r["errors"]:
                print(f"  Ex {ex_id} [cell {i}]: {en}: {ev}")
            rc = 1
        if not r["fails"] and not r["errors"]:
            print("OK — every check passes")
    return rc


if __name__ == "__main__":
    sys.exit(main())
