-- SQL Capstone — Marketplace Analytics Workbook (PostgreSQL)
-- Load schema.sql first, then write each task's query below. See README.md for
-- the exact spec + expected output of every task. No reference solution in repo.

-- ============================================================================
-- Task 1 — SELECT / WHERE / ORDER BY / LIMIT
-- 3 largest 2025 orders by net_amount (seller_id, ordered_at, net_amount),
-- biggest first, ties broken by ordered_at.
-- TODO:
select null;


-- ============================================================================
-- Task 2 — GROUP BY + aggregates + HAVING
-- Per seller in 2025: count, sum, round(avg,2); only sellers with >= 2 orders.
-- TODO:
select null;


-- ============================================================================
-- Task 3 — JOIN + COUNT(DISTINCT) + COALESCE/CASE
-- Per category ('n/a' -> 'Other'): distinct seller count + total 2025 net.
-- (Inner join orders, so no-order sellers are excluded.)
-- TODO:
select null;


-- ============================================================================
-- Task 4 — LEFT JOIN + CASE buckets + COALESCE
-- EVERY seller with 2025 total (0 if none) and a bucket:
--   >=3000 'high', >=1000 'medium', >0 'low', else 'none'. Order by total desc.
-- hint: LEFT JOIN keeps ZetaToys; COALESCE(sum(...) FILTER (year=2025), 0)
-- TODO:
select null;


-- ============================================================================
-- Task 5 — Subquery with EXISTS
-- Sellers with at least one refund (net_amount < 0) in 2025.
-- hint: WHERE EXISTS (SELECT 1 FROM orders o WHERE o.seller_id = s.id AND ...)
-- TODO:
select null;


-- ============================================================================
-- Task 6 — Window function: ranking within a group
-- Rank sellers by 2025 net WITHIN their category.
-- hint: dense_rank() over (partition by category order by total desc)
--       (compute per-seller 2025 totals in a CTE first)
-- TODO:
select null;


-- ============================================================================
-- Task 7 — Date functions + running total
-- Monthly 2025 net with a cumulative running total.
-- hint: date_trunc('month', ordered_at::timestamp) in a CTE, then
--       sum(net) over (order by month). NB: 'month' is reserved — alias it.
-- TODO:
select null;


-- ============================================================================
-- Task 8 — THE REPORT (finale)
-- category | sellers | Q1'25..Q4'25 | revenue_share   (see README for full spec)
-- Build it as CTEs:
--   sp : per seller -> cat ('n/a'->'Other'), raw gmv, parsed gmvnum (K/M/B; n/a->0)
--   cl : per cat -> string_agg "name (gmv)" ordered by gmvnum desc, name
--   q  : per cat -> SUM FILTER per 2025 quarter (q1..q4) + cat_total
-- final: format quarters ('$x' / '($x)' / NULL); revenue_share via sum() over ();
--        inner join cl+q; order by category.
-- TODO:
select null;
