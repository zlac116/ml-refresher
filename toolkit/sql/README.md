# SQL

PostgreSQL — beginner → advanced, with one full capstone.

## Contents

- [`sql_cheatsheet.md`](sql_cheatsheet.md) — code-first reference. Lead with this; everything else fills in details.
- [`capstones/`](capstones/) — full exercises.

## Capstones

| # | Capstone | What it teaches |
|---|---|---|
| 01 | [`marketplace_analytics/`](capstones/01_marketplace_analytics/) | CTEs, FILTER aggregates, window funcs, `string_agg`, multi-grain reporting |

## Reading order

1. Skim §0–§3 of `sql_cheatsheet.md` if rusty (basics + intermediate).
2. Read §4–§5 for window functions / advanced.
3. Attempt the marketplace_analytics capstone end-to-end.
4. Re-read §6 (gotchas) after — most mistakes you made will be in there.

## Related

- [`toolkit/eda_decisions.md`](../toolkit/eda_decisions.md) — when SQL meets pandas.
- [`api_engineering/`](../api_engineering/) — same patterns via SQLAlchemy 2.0 in Python.
