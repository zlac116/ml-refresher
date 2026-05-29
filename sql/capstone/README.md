# SQL Capstone — Marketplace Analytics Workbook

> **This is YOUR exercise.** `schema.sql` is provided (load it as-is). You write
> the queries in `tasks.sql` — a skeleton of TODO'd blocks. No reference solution
> is in this repo. When done, ask me to **review** it.

A workbook of **8 tasks** over one dataset, climbing from everyday SQL to an
advanced report. It deliberately covers the **most-used PostgreSQL features** —
not just the billing-report toolkit — so you practise the whole core surface.

⏱️ **Time budget: ~2 hours (hard cap).** Drills 1–7 are short (~8 min each); the
finale (Task 8) is the big one (~40 min).

---

## The data (`schema.sql`)

```
sellers(id, name, category, gmv_tier)      -- category may be 'n/a'; gmv_tier like '$2M','$500K','n/a'
orders(seller_id, ordered_at, net_amount)  -- ordered_at is TEXT 'YYYY-MM-DD HH:MI:SS'; net_amount can be negative
```
6 sellers (one, `ZetaToys`, has **no orders** — for LEFT JOIN / "none" cases) and
10 orders (one in **2024**, which most tasks must exclude).

## Run it
```bash
docker run -d --name sqlcap -e POSTGRES_PASSWORD=pw -e POSTGRES_DB=cap -p 5432:5432 postgres:17
sleep 5
psql postgresql://postgres:pw@localhost:5432/cap -f schema.sql
psql postgresql://postgres:pw@localhost:5432/cap -f tasks.sql      # your answers
# teardown: docker rm -f sqlcap
```
(Or `psql -d <yourdb> -f schema.sql` against your own Postgres.)

---

## Part 1 — Drills (each = one query in `tasks.sql`)

Each drill names the **features** it trains and shows the **expected output** so you
can self-check.

### Task 1 — SELECT / WHERE / ORDER BY / LIMIT
The 3 largest orders by `net_amount` in **2025** (`seller_id, ordered_at, net_amount`),
biggest first, ties broken by `ordered_at`.
```
 seller_id |     ordered_at      | net_amount
         3 | 2025-03-30 08:00:00 |    3000.00
         1 | 2025-05-04 14:30:00 |    2000.00
         5 | 2025-06-15 17:00:00 |    2000.00
```

### Task 2 — GROUP BY + aggregates + HAVING
Per seller in 2025: order count, total, and `round(avg,2)` — only sellers with **≥2 orders**.
```
 seller_id | n |  total  | avg_net
         1 | 2 | 3000.00 | 1500.00
         2 | 3 |  200.00 |   66.67
         3 | 2 | 4000.00 | 2000.00
```

### Task 3 — JOIN + COUNT(DISTINCT) + COALESCE/CASE
Per category (`'n/a'` → `'Other'`): distinct seller count and total 2025 net. (Sellers
with no 2025 orders don't appear — inner join.)
```
  category   | sellers | total_net
 Apparel     |       2 |   4500.00
 Electronics |       2 |   3200.00
 Other       |       1 |   2000.00
```

### Task 4 — LEFT JOIN + CASE buckets + COALESCE
**Every** seller (incl. `ZetaToys`, 0 orders) with their 2025 total and a bucket:
`>=3000 high`, `>=1000 medium`, `>0 low`, else `none`. Order by total desc.
```
    name    | total_2025 | bucket
 GammaWear  |    4000.00 | high
 AlphaTech  |    3000.00 | high
 EpsilonArt |    2000.00 | medium
 DeltaHome  |     500.00 | low
 BetaGoods  |     200.00 | low
 ZetaToys   |          0 | none
```

### Task 5 — Subquery with EXISTS
Sellers who had **at least one refund** (`net_amount < 0`) in 2025.
```
   name
 BetaGoods
```

### Task 6 — Window function: ranking within a group
Rank sellers by their 2025 net **within their category** (`DENSE_RANK() OVER (PARTITION BY ...)`).
```
  category   |    name    | total_2025 | rnk
 Apparel     | GammaWear  |    4000.00 |   1
 Apparel     | DeltaHome  |     500.00 |   2
 Electronics | AlphaTech  |    3000.00 |   1
 Electronics | BetaGoods  |     200.00 |   2
 Other       | EpsilonArt |    2000.00 |   1
```

### Task 7 — Date functions + running total
Monthly 2025 net (`date_trunc('month', ...)`) with a **cumulative running total**
(`sum(...) OVER (ORDER BY month)`). (Tip: `month` is reserved — alias the column something else.)
```
   ym    |   net   | running_total
 2025-02 | 1000.00 |       1000.00
 2025-03 | 3000.00 |       4000.00
 2025-04 | -500.00 |       3500.00
 2025-05 | 2000.00 |       5500.00
 2025-06 | 2000.00 |       7500.00
 2025-07 | 1000.00 |       8500.00
 2025-08 | -800.00 |       7700.00
 2025-11 | 1500.00 |       9200.00
 2025-12 |  500.00 |       9700.00
```

---

## Part 2 — Task 8: the advanced report (the finale)

One query, columns `category | sellers | Q1'25 | Q2'25 | Q3'25 | Q4'25 | revenue_share`:

- **category** — `'n/a'` → `'Other'`; sort ascending.
- **sellers** — comma-join `name (gmv_tier)` (raw tier shown), ordered by **parsed gmv DESC, name ASC**. Parse `$2M`/`$500K`/`$1.5M` (K/M/B) to a number; `n/a` → 0 (sort only).
- **Q1'25…Q4'25** — quarter net of 2025: positive `$1234.56`, negative `($-1234.56)`, none `NULL`.
- **revenue_share** — this category's 2025 net as a **% of the grand total**, 1 decimal + `%`, via a **window function** (`sum(...) OVER ()`), not a subquery.
- Only 2025; only categories with ≥1 2025 order (inner join).

**Verified expected output:**
```
  category   |              sellers              |  Q1'25   |  Q2'25   |   Q3'25    |  Q4'25   | revenue_share
 Apparel     | GammaWear ($1.5M),DeltaHome (n/a) | $3000.00 |          | $1000.00   | $500.00  | 46.4%
 Electronics | AlphaTech ($2M),BetaGoods ($500K) | $1000.00 | $1500.00 | ($-800.00) | $1500.00 | 33.0%
 Other       | EpsilonArt ($300K)                |          | $2000.00 |            |          | 20.6%
```
(Blank cells = `NULL`. 2024 order excluded; Electronics Q3 net-negative; shares sum to 100%.)

---

## Optional stretch (beyond 2h — see `../sql_cheatsheet.md`)
Not required. If you finish early, try these against the same data:
- **Recursive CTE** — generate the 12 months of 2025 so empty months show as 0.
- **GROUPING SETS / ROLLUP** — category subtotals + a grand-total row.
- **LATERAL join** — each seller with their single largest order.
- **Upsert** — `INSERT … ON CONFLICT (id) DO UPDATE` a seller row.
- **View** — wrap Task 8 in `CREATE VIEW seller_report AS …`.

## Milestones (~2 hours)
| Step | Tasks | Est. |
|------|-------|------|
| 1 | Load schema; Tasks 1–2 (basics, GROUP BY/HAVING) | 20 min |
| 2 | Tasks 3–5 (joins, COALESCE/CASE, EXISTS) | 30 min |
| 3 | Tasks 6–7 (window ranking + running total, dates) | 30 min |
| 4 | Task 8 (the report) | 40 min |

## Success criteria
- Tasks 1–7 match the expected blocks above.
- Task 8 matches its table exactly (formatting, NULLs, row order); `revenue_share` uses a window function; the 2024 order never leaks in.

When ready: **"review my SQL capstone"** and I'll assess against this spec.
