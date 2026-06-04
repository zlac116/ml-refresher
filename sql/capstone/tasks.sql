-- SQL Capstone — Marketplace Analytics Workbook (PostgreSQL)
-- Load schema.sql first, then write each task's query below. See README.md for
-- the exact spec + expected output of every task. No reference solution in repo.

-- ============================================================================
-- Task 1 — SELECT / WHERE / ORDER BY / LIMIT
-- 3 largest 2025 orders by net_amount (seller_id, ordered_at, net_amount),
-- biggest first, ties broken by ordered_at.
-- TODO:
select seller_id, ordered_at, net_amount
from orders
where extract(year from ordered_at::timestamp) = 2025
order by net_amount desc, ordered_at
limit 3;


-- ============================================================================
-- Task 2 — GROUP BY + aggregates + HAVING
-- Per seller in 2025: count, sum, round(avg,2); only sellers with >= 2 orders.
-- TODO:
select seller_id, count(*) as n, sum(net_amount) as total, round(avg(net_amount),2) as avg_net
from orders
where extract(year from ordered_at::timestamp) = 2025
group by seller_id
having count(*) >= 2
order by seller_id;


-- ============================================================================
-- Task 3 — JOIN + COUNT(DISTINCT) + COALESCE/CASE
-- Per category ('n/a' -> 'Other'): distinct seller count + total 2025 net.
-- (Inner join orders, so no-order sellers are excluded.)
-- TODO:
with a as 
    (
        select category as cat, seller_id, net_amount, ordered_at
        from sellers s
        inner join orders o on o.seller_id = s.id
    ) 
select 
    case when cat = 'n/a' then 'Other' else cat end as category,
    count(distinct seller_id) as sellers, 
    sum(net_amount) as total_net
from a 
where extract(year from ordered_at::timestamp) = 2025 
group by 1
order by 1;


-- ============================================================================
-- Task 4 — LEFT JOIN + CASE buckets + COALESCE
-- EVERY seller with 2025 total (0 if none) and a bucket:
--   >=3000 'high', >=1000 'medium', >0 'low', else 'none'. Order by total desc.
-- hint: LEFT JOIN keeps ZetaToys; COALESCE(sum(...) FILTER (year=2025), 0)
-- TODO:
with a as 
    (
        select s.name, coalesce(sum(o.net_amount) filter (where extract(year from o.ordered_at::timestamp) = 2025), 0) as total_2025
        from sellers s
        left join orders o on o.seller_id = s.id
        group by s.name
    ) 
select 
    name, 
    total_2025,
    case 
        when total_2025 >= 3000 then 'high'
        when total_2025 >= 1000 then 'medium' 
        when total_2025 > 0 then 'low' 
        else 'none' 
    end as bucket 
from a
order by total_2025 desc, name;

-- -- Alternative 1: group after join, then sum() with filter + COALESCE in CASE:
-- with a as 
--     (
--         select s.name, o.ordered_at, o.net_amount
--         from sellers s
--         left join orders o on o.seller_id = s.id
--     ) 
-- select 
--     name, 
--     coalesce(sum(net_amount) filter (where extract(year from ordered_at::timestamp) = 2025), 0) as total_2025,
--     case 
--         when coalesce(sum(net_amount) filter (where extract(year from ordered_at::timestamp) = 2025), 0) >= 3000 then 'high'
--         when coalesce(sum(net_amount) filter (where extract(year from ordered_at::timestamp) = 2025), 0) >= 1000 then 'medium' 
--         when coalesce(sum(net_amount) filter (where extract(year from ordered_at::timestamp) = 2025), 0) > 0 then 'low' 
--         else 'none' 
--     end as bucket 
-- from a
-- group by name
-- order by total_2025 desc, name;

-- -- Alternative 2: group after join, then sum() without filter + COALESCE in CASE:
-- with a as 
--     (
--         select s.name, o.ordered_at, o.net_amount
--         from sellers s
--         left join orders o on o.seller_id = s.id and extract(year from o.ordered_at::timestamp) = 2025
--     ) 
-- select 
--     name, 
--     coalesce(sum(net_amount), 0) as total_2025,
--     case 
--         when coalesce(sum(net_amount), 0) >= 3000 then 'high'
--         when coalesce(sum(net_amount), 0) >= 1000 then 'medium' 
--         when coalesce(sum(net_amount), 0) > 0 then 'low' 
--         else 'none' 
--     end as bucket 
-- from a
-- group by name
-- order by total_2025 desc, name;

-- ============================================================================
-- Task 5 — Subquery with EXISTS
-- Sellers with at least one refund (net_amount < 0) in 2025.
-- hint: WHERE EXISTS (SELECT 1 FROM orders o WHERE o.seller_id = s.id AND ...)
-- TODO:
select s.name
from sellers s
where exists (
    select 1 from orders o
    where o.seller_id = s.id and o.net_amount < 0 and extract(year from o.ordered_at::timestamp) = 2025
)
order by s.name;


-- ============================================================================
-- Task 6 — Window function: ranking within a group
-- Rank sellers by 2025 net WITHIN their category.
-- hint: dense_rank() over (partition by category order by total desc)
--       (compute per-seller 2025 totals in a CTE first)
-- TODO:
with a as(
    select 
        case 
            when s.category = 'n/a' then 'Other'
            else s.category 
        end as category,
        s.name, 
        sum(o.net_amount) filter (where extract(year from o.ordered_at::timestamp) = 2025) as total_2025 
    from sellers s
    inner join orders o on o.seller_id = s.id
    group by 1, 2
)
select 
    category,
    name,
    total_2025,
    dense_rank() over (partition by category order by total_2025 desc) as rnk
from a
order by category, rnk;


-- ============================================================================
-- Task 7 — Date functions + running total
-- Monthly 2025 net with a cumulative running total.
-- hint: date_trunc('month', ordered_at::timestamp) in a CTE, then
--       sum(net) over (order by month). NB: 'month' is reserved — alias it.
-- TODO:
with a as (
    select 
        date_trunc('month', ordered_at::timestamp) as mth,
        sum(net_amount) as net
    from orders 
    where extract(year from ordered_at::timestamp) = 2025 group by mth
)
select to_char(mth, 'YYYY-MM') as ym, net, sum(net) over (order by mth) as running_total
from a
order by ym;


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
with sp as (
    select
        id,
        name,
        case when category = 'n/a' then 'Other' else category end as cat,
        gmv_tier as gmv,
        case
            when gmv_tier = 'n/a' then 0
            when right(gmv_tier, 1) = 'M' then left(replace(gmv_tier, '$', ''), -1)::numeric * 1_000_000
            when right(gmv_tier, 1) = 'K' then left(replace(gmv_tier, '$', ''), -1)::numeric * 1_000
            else 0
        end as gmvnum
    from sellers
), cl as (
    select cat, string_agg(concat(name || ' (' || gmv || ')'), ',' order by gmvnum desc, name) as sellers
    from sp
    group by cat
), q as (
    select
        
        sp.cat,
        sum(o.net_amount) filter (where extract(quarter from o.ordered_at::timestamp) = 1) as q1,
        sum(o.net_amount) filter (where extract(quarter from o.ordered_at::timestamp) = 2) as q2,
        sum(o.net_amount) filter (where extract(quarter from o.ordered_at::timestamp) = 3) as q3,
        sum(o.net_amount) filter (where extract(quarter from o.ordered_at::timestamp) = 4) as q4,
        sum(o.net_amount) as cat_total
    from orders o
    join sp on sp.id = o.seller_id
    where extract(year from o.ordered_at::timestamp) = 2025
    group by sp.cat
)
select
    cl.cat as category,
    cl.sellers,
    case when q.q1 < 0 then '($' || q.q1 || ')' else '$' || q.q1 end as "Q1'25",
    case when q.q2 < 0 then '($' || q.q2 || ')' else '$' || q.q2 end as "Q2'25",
    case when q.q3 < 0 then '($' || q.q3 || ')' else '$' || q.q3 end as "Q3'25",
    case when q.q4 < 0 then '($' || q.q4 || ')' else '$' || q.q4 end as "Q4'25",
    round(100 * cat_total / sum(cat_total) over (), 1)::text || '%' as revenue_share
from cl
join q on q.cat = cl.cat;
