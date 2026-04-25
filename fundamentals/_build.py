"""Shared utilities for building the fundamentals notebooks.

A notebook is described as a list of section dicts. Each section has:
  - title:          "## N. Title"
  - intro:          markdown body explaining the concept
  - worked_intro:   one-line worked-example prompt (markdown)
  - worked_code:    python code demonstrating the concept
  - exercises:      list of {prompt, student, solution, explanation}

Each exercise's solution markdown is emitted as a SINGLE markdown cell
containing a `<details>` block — the only layout that Jupyter actually
collapses. Solutions stay hidden on first open.
"""
from __future__ import annotations
import nbformat as nbf


def md(text: str):
    return nbf.v4.new_markdown_cell(text.rstrip("\n") + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(text.rstrip("\n"))


def make_solution_md(solution_code: str, explanation: str) -> str:
    body = (
        "<details><summary>👉 <b>Click to reveal solution & explanation</b></summary>\n\n"
        f"```python\n{solution_code.rstrip()}\n```\n\n"
        f"{explanation.strip()}\n\n"
        "</details>"
    )
    return body


def build_section(section: dict) -> list:
    """Convert one section dict into a list of nbformat cells."""
    cells = []
    cells.append(md(f"---\n{section['title']}\n\n{section['intro'].strip()}"))
    cells.append(md(f"### Worked example\n{section['worked_intro'].strip()}"))
    cells.append(code(section["worked_code"]))
    cells.append(md("### Exercises\n\nWrite code for each. The `check()` will tell you if you're right."))
    for ex in section["exercises"]:
        cells.append(md(f"**Exercise {ex['id']}** — {ex['prompt'].strip()}"))
        cells.append(code(ex["student"]))
        cells.append(md(make_solution_md(ex["solution"], ex["explanation"])))
    return cells


def assemble_notebook(title_md: str, setup_code: str, sections: list, footer_md: str) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    cells = [md(title_md), code(setup_code)]
    for s in sections:
        cells.extend(build_section(s))
    cells.append(md(footer_md))
    nb["cells"] = cells
    return nb
