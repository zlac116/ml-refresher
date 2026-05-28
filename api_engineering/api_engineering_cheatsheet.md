# API Engineering Cheatsheet (Production FastAPI, 2026)

Condensed reference for building production REST APIs. See `tutorial/` for the
worked walkthrough and `capstone/` for the exercise.

## Layered architecture (keep concerns separate)
```
api (HTTP)  →  services (business logic + authz)  →  repositories (DB queries)  →  models (ORM)
                         ↑ schemas.py validates I/O at the boundary
```
No SQL in routers. No HTTP objects in services. No business logic in repositories.

```
app/{main, exceptions}.py
app/core/{config, security, observability}.py
app/db/session.py · app/{models, schemas, repositories, services}.py
app/api/{deps}.py · app/api/v1/{router, ...routes}.py
```

## Async DB (SQLAlchemy 2.0)
```python
engine = create_async_engine(settings.database_url, pool_pre_ping=True)   # ...+asyncpg
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)

async def get_session() -> AsyncIterator[AsyncSession]:
    async with SessionLocal() as s:
        yield s
```
`async def` routes must do **non-blocking** I/O only. Sync route → threadpool.

## Validation (Pydantic v2)
```python
class TradeCreate(BaseModel):
    symbol: str
    quantity: float = Field(gt=0)          # reject bad input at the edge -> 422
class TradeRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)   # never expose password_hash
```

## Auth: JWT + Argon2
```python
token = jwt.encode({"sub": uid, "iat": now, "exp": now+ttl}, key, algorithm="HS256")
jwt.decode(token, key, algorithms=["HS256"])          # verifies signature + exp
hash_ = PasswordHash.recommended().hash(pw)           # Argon2 (pwdlib)
```
Verify signature **and** expiry every protected request. Hash, never store, passwords.

## Authorization / ownership — close BOLA (OWASP API1)
```python
async def _get_owned(self, id, owner_id):
    t = await self.repo.get(id)
    if t is None or t.owner_id != owner_id:
        raise NotFoundError          # 404, NOT 403 — don't leak existence
    return t
```
The #1 API risk. Route every single-object op through an ownership check.

## Error handling (no leaks)
Raise domain exceptions in services; map to HTTP in one place:
```python
app.add_exception_handler(NotFoundError, lambda r, e: JSONResponse(404, {"error": ...}))
```
Never return stack traces or SQL.

## Config (12-factor)
```python
class Settings(BaseSettings):          # pydantic-settings, reads env/.env
    secret_key: str
    database_url: str
```
No secrets in code. `.env` is git-ignored.

## Pagination & rate limiting (OWASP API4)
```python
limit: int = Query(50, ge=1, le=100)   # cap page size
limiter = Limiter(key_func=get_remote_address)   # slowapi; stricter on /auth/login
```

## Observability
- **Structured logs**: structlog → JSON + per-request correlation id (middleware,
  echoed as `X-Request-ID`).
- **Tracing/metrics**: OpenTelemetry (FastAPI + SQLAlchemy) + Prometheus `/metrics`.
- **Probes**: `/health` (liveness, no DB) · `/ready` (503 until `SELECT 1` works).

## Testing
```python
AsyncClient(transport=ASGITransport(app=app), base_url="http://test")
app.dependency_overrides[get_session] = test_session   # SQLite test DB
```
Cover: happy path · validation (422) · auth required (401) · **BOLA (404)**.

## Deploy
- `gunicorn app.main:app -k uvicorn.workers.UvicornWorker -w <CPUs>`
- Multi-stage Dockerfile, non-root user, HEALTHCHECK. Migrations via Alembic.

## OWASP API Top 10 (2023, still current) — hit these
API1 BOLA (ownership) · API2 broken auth (JWT/Argon2) · API3 property-level (response
schemas) · API4 resource consumption (pagination + rate limit) · API5 function-level
auth · API8 misconfig (no debug/secrets, CORS locked).

## Async do/don't
- ✅ `await` async DB/HTTP libs (asyncpg, httpx).
- ❌ blocking calls (requests, time.sleep, heavy CPU) inside `async def` — they
  freeze the event loop. Offload to a threadpool or a worker.
