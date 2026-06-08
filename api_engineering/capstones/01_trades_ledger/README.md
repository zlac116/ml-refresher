# Capstone — Production Trades Ledger API

> **This is YOUR exercise.** Every module under `app/` is a skeleton: signatures
> and docstrings are provided, but the bodies raise `NotImplementedError` with
> `# TODO`s describing what to build. Implement them yourself. When you're done,
> ask me to **review** it — I won't hand you the solution.

Build a **production-grade REST API** for a personal trades/positions ledger,
applying current (May 2026) best practices. Authenticated users can record
trades, query their positions and realised/unrealised P&L, and close positions —
and must **never** be able to see or touch another user's data.

This is deliberately a CRUD service (not ML inference, which the
`../tutorial/rest_api.ipynb` already covers) so you exercise the full backend
stack: async DB, auth, ownership/authorization, observability, containerisation,
and tests.

> ### ⏱️ Time budget: ~2 hours (hard cap)
> The **CORE** (the requirements marked _Core_ below, and the Build order section) is
> sized to fit in about **2 hours**. Everything marked **Stretch** — close/patch/delete
> trades, structured logging, portfolio, OpenTelemetry, rate limiting, Alembic,
> Docker, CI — is **optional** and beyond the budget. The skeleton ships the
> stretch files too (clearly labelled) so the structure is complete; do **not**
> attempt them inside the 2 hours. Stop when the core is green.

---

## Functional requirements

A user owns **trades**. A trade has: symbol, side (buy/sell), quantity, entry
price, optional exit price, status (open/closed), timestamps, and an owner.

Endpoints (all under `/api/v1`, all JSON). The **Scope** column marks the
2-hour core vs optional stretch:

| Method & path | Auth | Scope | Purpose |
|---|---|---|---|
| `POST /auth/register` | public | **Core** | Create a user (email + password). |
| `POST /auth/login` | public | **Core** | Return a JWT access token (OAuth2 password flow). |
| `GET  /auth/me` | required | **Core** | Current user profile. |
| `POST /trades` | required | **Core** | Open a new trade (owner = caller). |
| `GET  /trades` | required | **Core** | List **caller's** trades; pagination + filter by status/symbol. |
| `GET  /trades/{id}` | required | **Core** | Fetch one trade — **only if caller owns it** (else 404). |
| `GET  /health` | public | **Core** | Liveness probe. |
| `PATCH /trades/{id}` | required | Stretch | Update an owned, open trade. |
| `POST /trades/{id}/close` | required | Stretch | Close an owned trade at an exit price; compute realised P&L. |
| `DELETE /trades/{id}` | required | Stretch | Delete an owned trade. |
| `GET  /portfolio/summary` | required | Stretch | Aggregate caller's open exposure + realised P&L. |
| `GET  /ready` | public | Stretch | Readiness probe (503 until DB reachable). |
| `GET  /metrics` | public | Stretch | Prometheus metrics. |

---

## Non-functional requirements (the "production" part)

These are the best practices you're being graded on.

### Core (required — fits the ~2-hour budget)

1. **Layered architecture** — keep concerns separated:
   `api` (HTTP) → `services` (business logic) → `repositories` (DB queries) →
   `models` (ORM). Schemas (`schemas.py`) validate I/O. No SQL in routers, no
   HTTP objects in services.
2. **Async all the way** — `async def` endpoints, SQLAlchemy 2.0 async engine.
   (Core may run on **SQLite via aiosqlite**; switch to Postgres/`asyncpg` for
   the stretch deployment.) No blocking I/O inside async routes.
3. **Validation** — Pydantic v2 models for every request/response. Reject bad
   input at the boundary (positive quantity, valid side enum, price > 0).
4. **Auth** — OAuth2 password flow issuing **JWTs**. Hash passwords with Argon2.
   Verify token signature **and** expiry on every protected request.
5. **Authorization / ownership (BOLA)** — the single most important security
   control here. Every trade access must check `trade.owner_id == current_user`.
   A user requesting someone else's trade id must get **404** (not 403 — don't
   leak existence).
6. **Error handling** — custom exceptions mapped to clean HTTP responses via
   exception handlers. Never leak stack traces or SQL.
7. **Config** — all settings from env via `pydantic-settings` (12-factor). No
   secrets in code. See `.env.example`.
8. **Tests** — pytest + httpx async client against a SQLite test DB: an
   auth-required check and an explicit **BOLA test** proving user A cannot read
   user B's trade.
9. **Liveness** — `GET /health`.

### Stretch (optional — beyond the 2-hour budget)

- **More trade ops** — `PATCH`, `POST /{id}/close` (realised P&L), `DELETE`.
- **Structured logging** — JSON logs with a per-request correlation id
  (middleware-generated, echoed in a response header).
- **Observability** — OpenTelemetry tracing on FastAPI + SQLAlchemy; Prometheus
  `/metrics`.
- **Rate limiting** — global limit + a stricter limit on `/auth/login`.
- **Migrations** — Alembic instead of dev-time `create_all`. Run on Postgres.
- **Containerisation** — multi-stage `Dockerfile` + `docker-compose.yml`.
- **CI** — GitHub Actions running lint + tests.
- **Readiness + portfolio aggregation** — `GET /ready` (DB probe) and
  `GET /portfolio/summary`.

---

## OWASP API Security Top 10 (2023) — your checklist

The 2023 list is still current in May 2026. Satisfy at least these:

- **API1 Broken Object-Level Authorization (BOLA)** → ownership checks (req. 5).
- **API2 Broken Authentication** → JWT signature + expiry validation, Argon2.
- **API3 Broken Object Property-Level Auth** → response schemas never expose
  `password_hash` or other users' fields.
- **API4 Unrestricted Resource Consumption** → pagination caps + rate limiting.
- **API5 Broken Function-Level Auth** → protected routes require a valid user.
- **API8 Security Misconfiguration** → no debug mode in prod, no secrets in code,
  CORS locked down.

---

## Setup (uv)

```bash
cd api_engineering/capstone
cp .env.example .env          # set SECRET_KEY; for the CORE use a SQLite URL:
                              #   DATABASE_URL=sqlite+aiosqlite:///./dev.db
uv sync                       # install all deps (+ dev group)
```

Run the API (core — SQLite, tables auto-created on startup):

```bash
uv run uvicorn app.main:app --reload     # docs at http://localhost:8000/docs
```

Run the tests:

```bash
uv run pytest
```

> **Stretch deployment only** (after the core works): switch `DATABASE_URL` to
> Postgres and either `uv run alembic upgrade head` or `docker compose up --build`.
> Don't do this for the 2-hour core.

---

## How to approach this build

**Mental model — a request flows down through layers, and each file is one layer:**

```
HTTP request
  → api/v1/*.py        route: receive the request, call a service
    → services.py      business logic + the ownership (BOLA) check
      → repositories.py    the actual DB queries
        → models.py          the ORM tables
  schemas.py     validates the JSON in/out at the route boundary
  core/          config (settings) + security (JWT + password hashing)
  db/session.py  the shared async DB connection
  main.py        wires everything together — assembled LAST
```

Once that clicks, each file has **one job** and you fill them **bottom-up**.

**Ignore these for the core (they're stretch — don't open them yet):**
`core/observability.py`, `Dockerfile`, `docker-compose.yml`, `migrations/`,
`.github/`, and the `/ready`, `/metrics`, `PATCH`, `/close`, `DELETE`,
`/portfolio/summary` routes.

## Build order (~2 hours)

Build in this sequence — each step unblocks the next, and **`main.py` comes last**
because it imports everything else. (This is the one canonical order; ignore any
other sequencing.)

| # | File — its one job | Est. |
|---|--------------------|------|
| 1 | `core/config.py` — settings + `get_settings()` | ✅ done |
| 2 | `db/session.py` — async engine + `SessionLocal` + `get_session` (import `get_settings`; SQLite URL) | 15 min |
| 3 | `models.py` — `User` + `Trade` ORM columns | 10 min |
| 4 | `schemas.py` — Pydantic request/response models | 10 min |
| 5 | `core/security.py` — hash/verify password + create/decode JWT | 15 min |
| 6 | `exceptions.py` — `register_exception_handlers` (map each error → HTTP status) | 10 min |
| 7 | `repositories.py` — the user + trade DB queries | 10 min |
| 8 | `services.py` — `AuthService` + `TradeService` (`_get_owned` BOLA check, open/get/list) | 20 min |
| 9 | `api/deps.py` — `get_current_user` (decode token → load user) | 10 min |
| 10 | `api/v1/auth.py` + `trades.py` + `ops.py` (`/health`) — wire routes to services | 15 min |
| 11 | `main.py` — `create_app`: FastAPI + exception handlers + routers + `lifespan` (`create_all`) | 10 min |
| 12 | `tests/` — auth-required + the **BOLA** test; `uv run pytest` | 15 min |

**Stretch (do NOT start within the 2 hours):** more trade ops (`PATCH` / `close`+P&L /
`DELETE`) → structured logging + correlation ids → rate limiting → `/ready` + `/metrics`
+ OpenTelemetry → Alembic on Postgres → Dockerfile + compose → CI → `/portfolio/summary`.

## Success criteria (core)

- The **Core** endpoints behave per the table; `uv run pytest` green.
- A request for another user's trade id returns **404** (the BOLA test passes).
- Auth is required on protected routes; bad input returns 422.
- `/docs` shows fully-typed request/response schemas.

(Stretch adds: JSON logs with correlation ids, `/metrics`, and `docker compose up`.)

When ready: **"review my capstone"** and I'll assess it against this spec.
