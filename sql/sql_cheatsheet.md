# PostgreSQL Cheatsheet — Beginner → Advanced

A dense, comprehensive reference for the SQL features worth knowing, grouped by
level. Snippets are PostgreSQL dialect.

---

# 1. Basics

## psql / meta-commands
```
psql postgresql://user:pw@host:5432/db     \l   list DBs       \dt  list tables
\d table   describe table   \df functions   \timing on   \x  expanded rows   \q quit
psql -d db -f file.sql        -- run a script
```

## Common data types
```
integer/int  bigint  smallint    numeric(p,s)/decimal   real/double precision
varchar(n)  text  char(n)        boolean                date  time  timestamp  timestamptz
uuid  json  jsonb  bytea         int[]  text[]  (arrays)   interval
```

## DDL — define schema
```sql
create table seller (
  id         serial primary key,            -- auto-increment (or: int generated always as identity)
  email      text unique not null,
  category   text default 'n/a',
  rating     numeric(3,2) check (rating between 0 and 5),
  owner_id   int references users(id) on delete cascade,
  created_at timestamptz default now()
);
alter table seller add column note text;          alter table seller drop column note;
alter table seller rename column note to memo;     drop table if exists seller cascade;
create index idx_seller_cat on seller(category);   truncate seller;
```

## DML — change data
```sql
insert into seller (email, category) values ('a@x.io','Tech'), ('b@x.io','n/a')
  returning id;                                   -- RETURNING gives back inserted rows
update seller set category = 'Other' where category = 'n/a' returning *;
delete from seller where created_at < now() - interval '1 year';
```

## SELECT essentials
```sql
select id, email as e from seller                 -- alias with AS (optional)
where category = 'Tech'                            -- = <> < > <= >=
  and rating between 3 and 5                       -- inclusive range
  and category in ('Tech','Retail')                -- set membership
  and email like '%@x.io'                          -- % any, _ one char ; ILIKE = case-insensitive
  and memo is not null                             -- never  = NULL
order by rating desc nulls last, id                -- multi-key sort
limit 20 offset 40;                                -- paging
select distinct category from seller;              -- dedupe rows
select distinct on (category) category, id         -- first row per category
  from seller order by category, rating desc;
```

---

# 2. Intermediate

## Aggregates + GROUP BY + HAVING
```sql
select category, count(*), count(distinct owner_id),
       sum(rating), avg(rating), min(rating), max(rating)
from seller
where created_at > '2025-01-01'      -- WHERE filters rows BEFORE grouping
group by category
having count(*) >= 2                  -- HAVING filters groups AFTER aggregation
order by count(*) desc;
```

## JOINs
```sql
from a join b        on a.k = b.k     -- INNER: only matching rows
from a left join b   on a.k = b.k     -- keep all A, NULLs for missing B
from a right join b  on a.k = b.k     -- keep all B
from a full join b   on a.k = b.k     -- keep both, NULLs where unmatched
from a cross join b                   -- cartesian product
from emp e join emp m on e.mgr = m.id -- SELF join (table to itself, aliased)
```

## Subqueries
```sql
where rating > (select avg(rating) from seller)            -- scalar subquery
where id in (select seller_id from orders)                 -- IN
where exists (select 1 from orders o where o.seller_id = s.id)   -- EXISTS (correlated)
select s.*, (select count(*) from orders o where o.seller_id=s.id) as n  -- subquery in SELECT
where rating > all (select rating from seller where category='Tech')     -- ALL / ANY
```

## Conditionals & NULLs
```sql
case when x < 0 then 'neg' when x = 0 then 'zero' else 'pos' end   -- searched
case grade when 'A' then 4 when 'B' then 3 else 0 end              -- simple
coalesce(a, b, 0)        -- first non-NULL
nullif(a, 0)             -- NULL if a = 0 (e.g. guard divide:  x / nullif(d,0))
greatest(a,b,c)  least(a,b,c)
```

## Type casts
```sql
'123'::int      x::text      ordered_at::timestamp      cast(x as numeric(10,2))
```

## Set operations
```sql
q1 union q2        -- combine + dedupe        q1 union all q2   -- keep dups (faster)
q1 intersect q2    -- rows in both            q1 except q2      -- in q1 not q2
```

## CTEs (WITH)
```sql
with active as (select * from seller where rating > 3),
     by_cat as (select category, count(*) n from active group by category)
select * from by_cat where n > 1;     -- chain readable steps; great for multi-grain joins
```

---

# 3. Strings, Dates, Numbers

## Strings
```sql
a || b                       length(s)        lower(s) upper(s) initcap(s)   trim(s) ltrim rtrim
left(s,n)  right(s,n)         -- n<0 => all but last/first |n| chars
substring(s from 2 for 3)     substr(s,2,3)    position('x' in s)
replace(s,'a','b')            split_part('a-b-c','-',2)  -- 'b'
lpad(s,5,'0')  rpad(s,5,' ')  concat_ws(',', a, b, c)    -- skips NULLs
s like '%x_'                  s ilike '%X%'    -- case-insensitive
s ~ '^[0-9]+$'                s ~* 'abc'       -- POSIX regex (~ match, ~* case-insensitive)
regexp_replace(s,'\d','#','g')   regexp_match(s,'(\d+)')
```
Parse text→number (`'$1.5M'`): `left(replace(t,'$',''), -1)::numeric * 1e6` (CASE on `right(t,1)`).

## Dates / timestamps
```sql
now()  current_date  current_timestamp
'2025-05-04 10:00'::timestamp
extract(year from ts)  extract(quarter from ts)  extract(month/day/dow/hour from ts)  -- numeric
date_trunc('month', ts)  date_trunc('day', ts)        -- floor
to_char(ts, 'YYYY-MM-DD')  to_char(ts,'YYYY-MM')       -- format -> text
ts + interval '7 days'   ts - interval '1 month'   age(ts1, ts2)
date_part('quarter', ts)                               -- same as extract
generate_series('2025-01-01'::date, '2025-12-01', '1 month')  -- a month per row
```

## Numbers
```sql
round(x,2)  ceil(x)  floor(x)  trunc(x,2)  abs(x)  mod(a,b)  power(a,b)  sqrt(x)
x::numeric / nullif(y,0)        -- safe division
to_char(x,'FM999,990.00')       -- '1,234.50'
```

---

# 4. Advanced

## Window functions (aggregate without collapsing rows)
```sql
sum(x)  over ()                                   -- grand total on every row (share-of-total)
sum(x)  over (partition by cat)                   -- per-group total, rows kept
sum(x)  over (order by dt)                        -- running total
avg(x)  over (order by dt rows between 2 preceding and current row)  -- moving avg (frame)
row_number() over (partition by cat order by x desc)   -- 1,2,3 unique
rank()       over (order by x desc)               -- 1,2,2,4 (gaps)
dense_rank() over (order by x desc)               -- 1,2,2,3 (no gaps)
ntile(4)     over (order by x)                    -- quartile buckets
lag(x)  over (order by dt)   lead(x) over (order by dt)        -- previous / next row
first_value(x) over (...)    last_value(x) over (...)
percent_rank() over (order by x)
-- reuse a window:
select x, sum(x) over w, avg(x) over w from t window w as (partition by cat order by dt);
```
Window funcs run *after* GROUP BY and can't appear in WHERE (filter in an outer query/CTE).

## FILTER + string_agg / array_agg
```sql
sum(amt) filter (where quarter = 1) as q1          -- conditional aggregate / pivot
string_agg(name, ', ' order by name)               -- rows -> delimited string
array_agg(id order by created_at)                  -- rows -> array
jsonb_agg(row_to_json(t))                           -- rows -> JSON array
```

## Recursive CTE
```sql
with recursive months as (
  select date '2025-01-01' as m
  union all
  select m + interval '1 month' from months where m < '2025-12-01'
)
select * from months;            -- also: org charts, graph traversal
```

## GROUPING SETS / ROLLUP / CUBE (subtotals)
```sql
select cat, region, sum(x) from t
group by rollup (cat, region);   -- cat+region rows, per-cat subtotals, grand total
-- grouping sets ((cat),(region),()) for custom combos; cube() for all combos
```

## LATERAL join (correlated join — "for each left row, run this subquery")
```sql
select s.name, o.*
from seller s
cross join lateral (
  select * from orders o where o.seller_id = s.id order by net_amount desc limit 1
) o;                              -- each seller's single biggest order
```

## Upsert
```sql
insert into seller (id, email) values (1, 'a@x.io')
on conflict (id) do update set email = excluded.email;     -- or DO NOTHING
```

## JSON / JSONB
```sql
data->'k'      -- json value      data->>'k'     -- text value
data#>'{a,b}'  -- deep get        data @> '{"k":1}'::jsonb   -- contains
jsonb_build_object('id', id, 'name', name)     jsonb_set(data,'{k}','1')
jsonb_array_elements(data->'items')            -- expand array to rows
```

## Arrays
```sql
'{1,2,3}'::int[]   array[1,2,3]   arr[1]   array_length(arr,1)
x = any(arr)   unnest(arr)        -- array -> rows      array_agg(x)  -- rows -> array
```

---

# 5. Ops & performance

## Views
```sql
create view seller_report as select ... ;            -- virtual, always fresh
create materialized view mv as select ... ;          -- stored; refresh materialized view mv;
```

## Indexes
```sql
create index on orders(seller_id);                   -- btree (default): =, <, >, ORDER BY, joins
create index on seller(lower(email));                -- expression index
create index on orders(seller_id) where net_amount<0;-- partial index
create index on doc using gin(data);                 -- GIN: jsonb @>, arrays, full-text
```
Index columns used in `WHERE`/`JOIN`/`ORDER BY` on big tables; don't over-index writes.

## Transactions
```sql
begin;  update ...;  savepoint s1;  delete ...;  rollback to s1;  commit;   -- or rollback;
```

## Inspect a query plan
```sql
explain select ...;            -- estimated plan
explain (analyze, buffers) select ...;   -- actually runs it; shows real time + rows
```
Watch for `Seq Scan` on large tables (missing index) and bad row estimates.

---

# 6. Gotchas
- `WHERE` (pre-aggregate) vs `HAVING` (post-aggregate). Window funcs → filter in an outer query.
- `NULL` semantics: `= NULL` is never true (use `IS NULL`); `anything || NULL = NULL`;
  `count(col)` skips NULLs, `count(*)` doesn't; `NULL` sorts last with `DESC` (use `NULLS FIRST/LAST`).
- `left(s,-1)` drops the **last** char (handy to strip a unit suffix).
- Reserved words (`month`, `order`, `user`, …) need an alias/quoting: `... as ym`, or `"user"`.
- Integer division: `5/2 = 2`; force `5/2.0` or `5::numeric/2`.
- `DISTINCT ON (x)` needs `ORDER BY x, …` to be deterministic.
- `UNION` dedupes (sorts); use `UNION ALL` if you don't need it — much faster.
