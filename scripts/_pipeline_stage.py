"""Reusable utility for rewriting a notebook section into the pipeline-stage format.

Each section dict has the shape:

    {
        'old_id':   2,                      # the existing section number (## 2.)
        'new_id':   2,                      # the new stage number (## Stage 2)
        'title':    'EDA — ...',            # text after the dash
        'why':      '**↳ Why we're here.** ...',
        'concept_md':       '### The 30-second concept ...',
        'failure_intro_md': '### Failure mode — ...',
        'failure_code':     'python source',
        'decisions_md':     '### Decisions you make at this stage ...',
        'exercises': [
            {
                'id': '2.1',
                'prompt_md':     '### Exercise 2.1 — ...',
                'pre_code':      None or 'buggy code to run',
                'student_stub':  '# Your answer here\\n...',
                'solution_md':   '<details>...</details>',
            },
            ...,
        ],
        'recap_md':  '### Recap — ...',
    }

The replace_section() function:
  1. Locates the section's old layout in the notebook.
  2. Captures any user solutions from the existing exercise answer cells.
  3. Inserts "Why we're here" right after the section header.
  4. Removes the old hint block + 4 exercises + their solutions.
  5. Inserts the new content after the existing teaching cells (preserved).
  6. Adds a fold-out at the bottom containing the captured user answers.
"""
from __future__ import annotations
import re
from typing import Optional

import nbformat as nbf


EX_RE = re.compile(r"\*\*Exercise\s([0-9.]+)")


def md(text: str):
    return nbf.v4.new_markdown_cell(text.lstrip("\n").rstrip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(text.lstrip("\n").rstrip())


def _find_section_bounds(nb, old_id: int) -> tuple[int, int, int]:
    """Return (header_idx, exercises_start_idx, next_section_idx)."""
    header_idx = None
    next_idx = None
    for i, c in enumerate(nb.cells):
        if c.cell_type != "markdown":
            continue
        first = c.source.split("\n")[0].strip()
        m = re.match(r"^##\s+(\d+)\.", first)
        if m:
            n = int(m.group(1))
            if n == old_id and header_idx is None:
                header_idx = i
            elif n == old_id + 1 and next_idx is None and header_idx is not None:
                next_idx = i
                break
    if header_idx is None:
        raise RuntimeError(f"could not find ## {old_id}. header")
    if next_idx is None:
        next_idx = len(nb.cells)

    ex_idx = None
    for j in range(header_idx + 1, next_idx):
        c = nb.cells[j]
        if c.cell_type == "markdown" and f"### Exercises — Section {old_id}" in c.source[:80]:
            ex_idx = j
            break
    if ex_idx is None:
        raise RuntimeError(f"could not find '### Exercises — Section {old_id}' in section")
    return header_idx, ex_idx, next_idx


def _capture_old_answers(nb, ex_start: int, sec_end: int, section_old_id: int) -> dict[str, str]:
    """Capture {ex_id: source} from user answer cells in the existing exercise area."""
    out = {}
    for i in range(ex_start, sec_end):
        c = nb.cells[i]
        if c.cell_type != "markdown":
            continue
        m = EX_RE.match(c.source.strip()[:60])
        if not m:
            continue
        if not m.group(1).startswith(f"{section_old_id}."):
            continue
        # The answer cell is the FIRST code cell that follows the prompt and
        # before the next markdown cell.
        for k in range(i + 1, min(sec_end, i + 6)):
            if nb.cells[k].cell_type == "code":
                out[m.group(1)] = nb.cells[k].source
                break
    return out


def _build_old_answers_foldout(old_id: int, captured: dict[str, str], titles: dict[str, str]) -> str:
    """Compose the markdown for the 'Your previously-written answers' fold-out."""
    if not captured:
        return ""
    body = "<details><summary>📁 Your previously-written answers from the old exercise format</summary>\n\n"
    body += (
        f"The old format had multiple exercises in this section. Their building blocks have been "
        f"folded into the new exercises above. Your previous answers are preserved here verbatim "
        f"for reference.\n\n"
    )
    for ex_id in sorted(captured.keys(), key=lambda s: tuple(int(x) for x in s.split("."))):
        title = titles.get(ex_id, f"Old {ex_id}")
        body += f"**{title}**\n\n```python\n{captured[ex_id].rstrip()}\n```\n\n"
    body += "</details>"
    return body


def _make_exercise_cells(ex: dict) -> list:
    """Markdown prompt + (optional pre-code) + student stub + solution markdown."""
    cells = [md(ex["prompt_md"])]
    if ex.get("pre_code"):
        cells.append(code(ex["pre_code"]))
    cells.append(code(ex["student_stub"]))
    cells.append(md(ex["solution_md"]))
    return cells


def replace_section(nb, section: dict, old_titles: Optional[dict[str, str]] = None) -> None:
    """In-place rewrite of a section into the new pipeline-stage format."""
    old_id = section["old_id"]
    new_id = section["new_id"]
    header_idx, ex_idx, next_idx = _find_section_bounds(nb, old_id)
    captured = _capture_old_answers(nb, ex_idx, next_idx, old_id)

    # Replace the section header markdown.
    nb.cells[header_idx].source = (
        f"---\n## Stage {new_id} — {section['title']}\n\n{section['why'].strip()}\n"
    )

    # Build the new content cells (everything after the existing teaching cells).
    new_cells: list = []
    new_cells.append(md(section["concept_md"]))
    new_cells.append(md(section["failure_intro_md"]))
    new_cells.append(code(section["failure_code"]))
    new_cells.append(md(section["decisions_md"]))
    for ex in section["exercises"]:
        new_cells.extend(_make_exercise_cells(ex))
    new_cells.append(md(section["recap_md"]))
    fold = _build_old_answers_foldout(old_id, captured, old_titles or {})
    if fold:
        new_cells.append(md(fold))

    # Splice: keep [..ex_idx), drop [ex_idx..next_idx), insert new_cells, then [next_idx..].
    # The existing worked-example cells (added in the previous iteration) live
    # inside the dropped region, so they're removed too. The section's *teaching*
    # code cells (the ones that build wide_close, X_train, etc.) live BEFORE
    # ex_idx — they're preserved.
    nb.cells = (
        nb.cells[: ex_idx]
        + new_cells
        + nb.cells[next_idx:]
    )
