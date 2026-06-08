# PostgreSQL Cheatsheet — Beginner → Advanced

A dense, comprehensive reference for PostgreSQL ≥ 14 (most patterns work
back to PG 12; version notes are added where features are newer).
Targets the *current* recommended idioms, not legacy ones.

**Modern conventions worth knowing upfront** — these are what Postgres docs
themselves recommend, and what the rest of this cheatsheet defaults to:

- **`int generated always as identity`** over `serial` — SQL-standard, recommended since PG 10.
- **`MERGE` (PG 15+)** over `ON CONFLICT` for multi-action upserts; `ON CONFLICT` is fine for simple upsert.
- **`CREATE INDEX CONCURRENTLY`** in production — avoids the exclusive table lock.
- **`IS DISTINCT FROM`** for NULL-safe equality — `a = b` returns NULL when either is NULL.
- **`EXPLAIN (analyze, buffers, verbose)`** for plans — more useful than the bare flags.
- **CTE `MATERIALIZED` / `NOT MATERIALIZED`** hints (PG 12+) — pre-PG 12, CTEs were always an optimization barrier; that changed.

---

# 1. Basics

## psql / meta-commands
```
psql postgresql://user:pw@host:5432/db     \l   list DBs       \dt  list tables
\d table   describe table   \df functions   \timing on   \x  expanded rows   \q quit
psql -d db -f file.sql        -- run a script
\copy table from 'file.csv' csv header   -- bulk import from client-side file
```

## Common data types
```
integer/int  bigint  smallint    numeric(p,s)/decimal   real/double precision
varchar(n)  text  char(n)        boolean                date  time  timestamp  timestamptz
uuid  json  jsonb  bytea         int[]  text[]  (arrays)   interval
int4range  tstzrange  daterange  (range types)    tsvector  tsquery  (full-text)
```

## DDL — define schema (modern style)
```sql
create table seller (
  id         int generated always as identity primary key,   -- prefer over `serial`
  email      text unique not null,
  category   text default 'n/a',
  rating     numeric(3,2) check (rating between 0 and 5),
  owner_id   int references users(id) on delete cascade,
  -- Generated (computed) column — recomputed on read for STORED:
  display    text generated always as (upper(email)) stored,
  created_at timestamptz default now()
);
comment on column seller.rating is '0-5 stars from customer reviews';
alter table seller add column note text;
alter table seller drop column note;
alter table seller rename column note to memo;
drop table if exists seller cascade;
truncate seller restart identity;            -- restart identity sequences
```

## DML — change data (use RETURNING liberally)
```sql
insert into seller (email, category) values ('a@x.io','Tech'), ('b@x.io','n/a')
  returning id, created_at;                  -- INSERT/UPDATE/DELETE all support RETURNING
update seller set category = 'Other' where category = 'n/a' returning id, email;
delete from seller where created_at < now() - interval '1 year' returning id;

-- Bulk load from server-side file (admin only) or via client-side \copy:
copy seller (email, category) from '/path/file.csv' csv header;
```

## SELECT essentials
```sql
select id, email as e from seller                 -- alias with AS (optional)
where category = 'Tech'                            -- = <> < > <= >=
  and rating between 3 and 5                       -- inclusive range
  and category in ('Tech','Retail')                -- set membership
  and email like '%@x.io'                          -- % any, _ one char ; ILIKE = case-insensitive
  and memo is not null                             -- never  = NULL
  and a is distinct from b                         -- NULL-safe `<>` (treats NULL as a value)
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
from a join b using (k)               -- shorthand when join columns share a name
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
coalesce(a, b, 0)             -- first non-NULL
nullif(a, 0)                  -- NULL if a = 0 (e.g. guard divide:  x / nullif(d,0))
greatest(a,b,c)  least(a,b,c)
a is distinct from b          -- NULL-safe inequality
a is not distinct from b      -- NULL-safe equality (both NULL → true)
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

## CTEs (WITH) + materialization hints (PG 12+)
```sql
with active as (select * from seller where rating > 3),
     by_cat as (select category, count(*) n from active group by category)
select * from by_cat where n > 1;

-- Materialization hints — PG 12+
with active as materialized       (select * from seller where rating > 3) ... -- force materialize
with active as not materialized   (select * from seller where rating > 3) ... -- force inline (allow planner to push predicates)
```
Pre-PG 12 CTEs were ALWAYS materialized (a "fence" the planner couldn't see through).
PG 12+ inlines simple CTEs by default but materializes if the CTE is referenced
multiple times or has side effects. Use the hint when the default guesses wrong.

## CTE with INSERT/UPDATE/DELETE (data pipelines in one statement)
```sql
with deleted as (
  delete from staging where processed_at < now() - interval '7 days' returning *
)
insert into archive select * from deleted;
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
extract(year from ts)  extract(quarter from ts)  extract(month/day/dow/hour from ts)
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
avg(x)  over (order by dt rows between 2 preceding and current row)  -- moving avg
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

### Frame clauses — ROWS vs RANGE vs GROUPS (PG 11+)
```sql
-- ROWS: by physical row count
sum(x) over (order by dt rows between 7 preceding and current row)        -- last-8-row sum

-- RANGE: by VALUE distance (logical) — pairs naturally with timestamps
sum(x) over (order by dt range between interval '7 days' preceding and current row)

-- GROUPS (PG 11+): by peer groups (groups of rows with equal ORDER BY value)
sum(x) over (order by dt groups between 1 preceding and current row)

-- Exclusions
sum(x) over (... rows between 1 preceding and 1 following exclude current row)
sum(x) over (... rows between unbounded preceding and current row exclude ties)
```

## FILTER, WITHIN GROUP, string_agg / array_agg
```sql
sum(amt) filter (where quarter = 1) as q1          -- conditional aggregate / pivot
count(*) filter (where status = 'error') as errors

-- WITHIN GROUP — ordered-set aggregates (percentiles, mode):
percentile_cont(0.5)  within group (order by x) as median
percentile_disc(0.9)  within group (order by x) as p90
mode()                within group (order by x) as most_common

string_agg(name, ', ' order by name)               -- rows -> delimited string
array_agg(id order by created_at)                  -- rows -> array
jsonb_agg(row_to_json(t) order by id)              -- rows -> JSON array
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
-- grouping(cat) returns 1 if this is a subtotal row over cat, 0 otherwise
```

## LATERAL join (correlated join — "for each left row, run this subquery")
```sql
select s.name, o.*
from seller s
cross join lateral (
  select * from orders o where o.seller_id = s.id order by net_amount desc limit 1
) o;                              -- each seller's single biggest order
```

## Upsert — `ON CONFLICT` (simple) vs `MERGE` (PG 15+, SQL-standard)
```sql
-- ON CONFLICT — simple, single-row, INSERT-with-fallback
insert into seller (id, email) values (1, 'a@x.io')
on conflict (id) do update set email = excluded.email;

-- ON CONFLICT with conditional update (only update if better)
insert into seller (id, email, rating) values (1, 'a@x.io', 5.0)
on conflict (id) do update set rating = excluded.rating
  where excluded.rating > seller.rating;                 -- skip update if worse

-- MERGE (PG 15+) — SQL-standard, multi-action, can DELETE in same statement
merge into seller s
using staging t on s.id = t.id
when matched and t.deleted then delete
when matched then update set email = t.email, rating = t.rating
when not matched then insert (id, email, rating) values (t.id, t.email, t.rating);
```

## SELECT ... FOR UPDATE SKIP LOCKED (queue pattern)
```sql
-- Worker pulls one row to process, skipping rows other workers have locked
begin;
  select * from jobs where status = 'pending'
    order by created_at limit 1
    for update skip locked;
  -- ... do the work, then:
  update jobs set status = 'done' where id = $1;
commit;
```
Lets N workers process a queue concurrently without blocking each other.

## JSON / JSONB
```sql
data->'k'      -- json value      data->>'k'      -- text value
data#>'{a,b}'  -- deep get json    data#>>'{a,b}'  -- deep get text
data @> '{"k":1}'::jsonb           -- contains
data ?  'k'                         -- key exists
jsonb_build_object('id', id, 'name', name)
jsonb_set(data,'{k}','1')
jsonb_array_elements(data->'items')               -- expand JSON array to rows
jsonb_path_query(data, '$.items[*] ? (@.price > 10)')   -- PG 12+ JSONPath
jsonb_path_exists(data, '$.items[*].price')
```
**Indexing**: `create index on doc using gin (data jsonb_path_ops);` is faster
+ smaller for `@>` containment queries than the default `gin(data)`.

## Arrays
```sql
'{1,2,3}'::int[]   array[1,2,3]   arr[1]   array_length(arr,1)
x = any(arr)   unnest(arr)        -- array -> rows
array_agg(x order by id)          -- rows -> array
arr || arr2                       -- concat
unnest(arr) with ordinality as t(val, pos)        -- preserve position (PG 9.4+)
```

## Range types + EXCLUDE constraints (no-overlap guarantees)
```sql
create extension if not exists btree_gist;        -- needed for non-range cols in EXCLUDE
create table booking (
  room_id   int,
  during    tstzrange not null,
  exclude using gist (room_id with =, during with &&)   -- &&: range overlaps
);
-- Postgres now refuses inserts that would overlap. Range operators:
-- &&  overlaps,  @>  contains,  <@  contained-by,  -|-  adjacent
```

## Full-text search (basics)
```sql
to_tsvector('english', 'The quick brown fox') @@ to_tsquery('english', 'fox & brown');
-- Indexed full-text column:
alter table doc add column search tsvector
  generated always as (to_tsvector('english', title || ' ' || body)) stored;
create index on doc using gin (search);
-- Query:
select * from doc where search @@ websearch_to_tsquery('english', 'fox AND brown');
```
`websearch_to_tsquery` accepts Google-like syntax (`quoted phrases`, `OR`, `-exclude`).

---

# 5. Ops & performance

## Views
```sql
create view seller_report as select ... ;            -- virtual, always fresh
create materialized view mv as select ... ;          -- stored; refresh materialized view mv;
refresh materialized view concurrently mv;           -- no read-lock (needs unique index)
```

## Indexes — defaults + production-safe creation
```sql
create index on orders(seller_id);                          -- btree (default): =, <, >, ORDER BY, joins
create index on seller(lower(email));                       -- expression index
create index on orders(seller_id) where net_amount<0;       -- partial index
create unique index on user_ (email) where deleted_at is null;  -- partial UNIQUE — common pattern
create index on doc using gin (data jsonb_path_ops);        -- GIN: jsonb @>, arrays, full-text
create index on events using brin (created_at);             -- BRIN: huge append-only tables (cheap, lossy)

-- PRODUCTION: never block writers — use CONCURRENTLY
create index concurrently on orders(seller_id, ordered_at);
reindex concurrently index idx_orders_seller;
drop index concurrently if exists idx_old;
```
Index columns used in `WHERE` / `JOIN` / `ORDER BY` on big tables; don't over-index writes.
**`CONCURRENTLY` is roughly 2× slower but never holds the exclusive lock** —
required for online DDL. Caveat: it can't run inside a transaction.

## Transactions + isolation
```sql
begin;  update ...;  savepoint s1;  delete ...;  rollback to s1;  commit;

-- Isolation levels (default is READ COMMITTED):
begin isolation level repeatable read;     -- snapshot consistency for the whole txn
begin isolation level serializable;        -- strictest; may retry on conflicts (catch 40001)
```

## Inspect a query plan
```sql
explain select ...;                                    -- estimated plan, no execution
explain (analyze, buffers, verbose) select ...;        -- runs it; shows real time + I/O + col info
explain (analyze, buffers, format json) select ...;    -- machine-readable for tooling
```
Watch for `Seq Scan` on large tables (missing index), `Rows Removed by Filter`
(predicate not pushed down), and bad row estimates (planner stats stale →
run `analyze`).

## Maintenance (the things that bite you in prod)
```sql
analyze;                              -- update planner statistics (run after big inserts)
analyze table_name;
vacuum;                               -- reclaim dead row space (autovacuum usually handles)
vacuum (verbose, analyze) table_name;
vacuum full table_name;               -- rewrite table; takes EXCLUSIVE lock — emergency only
```

---

# 6. Gotchas

- **`WHERE`** (pre-aggregate) vs **`HAVING`** (post-aggregate). Window funcs → filter in an outer query/CTE.
- **`NULL` semantics**: `= NULL` is never true (use `IS NULL` or `IS DISTINCT FROM`); `anything || NULL = NULL`; `count(col)` skips NULLs, `count(*)` doesn't; `NULL` sorts last with `DESC` (use `NULLS FIRST/LAST`).
- **`left(s,-1)`** drops the **last** char (handy to strip a unit suffix).
- **Reserved words** (`month`, `order`, `user`, …) need an alias/quoting: `... as ym`, or `"user"`.
- **Integer division**: `5/2 = 2`; force `5/2.0` or `5::numeric/2`.
- **`DISTINCT ON (x)`** needs `ORDER BY x, …` to be deterministic.
- **`UNION`** dedupes (sorts); use `UNION ALL` if you don't need it — much faster.
- **CTEs pre-PG 12** were always materialized (optimization barrier). PG 12+ inlines by default. Use `MATERIALIZED` / `NOT MATERIALIZED` to override.
- **`CREATE INDEX CONCURRENTLY`** can't run inside a transaction block; if it fails midway you get an **invalid index** — drop and retry.
- **`MERGE`** doesn't lock conflicting rows the way `INSERT ON CONFLICT` does; under concurrent writes you may need `SERIALIZABLE` isolation.
- **`jsonb_path_query`** returns multiple rows (it's a set-returning function) — wrap in a subquery if you want one row per input row.
- **`GENERATED ... STORED`** can't be updated directly; the generation expression is re-evaluated on row modification.
- **`EXCLUDE` constraints** need the `btree_gist` extension for non-range columns (`room_id WITH =`).
- **Window frame default** is `RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW` when `ORDER BY` is specified — surprising for moving averages. Always be explicit.
