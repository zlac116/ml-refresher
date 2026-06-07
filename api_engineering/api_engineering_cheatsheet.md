# API Engineering Cheatsheet (Production FastAPI, 2026)

Condensed reference for production REST APIs. FastAPI ≥ 0.115, Pydantic v2,
SQLAlchemy 2.0. See `tutorial/` for the walkthrough and `capstone/` for
the exercise. Each section: paste-able code → why → trap.

---

## 1. Layered architecture (separate concerns, hard)

```
api (HTTP)  →  services (business logic + authz)  →  repositories (DB queries)  →  models (ORM)
                         ↑ schemas.py validates I/O at the boundary
```

**Rules**: no SQL in routers · no HTTP objects (Request, HTTPException) in services · no business logic in repositories.

```
app/main.py                        ← FastAPI app + lifespan
app/exceptions.py                  ← domain exception types
app/core/{config, security, observability}.py
app/db/session.py                  ← async engine + session factory
app/{models, schemas, repositories, services}.py
app/api/deps.py                    ← Annotated dependency aliases
app/api/v1/{router, ...routes}.py  ← versioned routes
```

**Why versioned (`v1/`)**: breaking changes ship as `v2/`. Old clients keep working.

---

## 2. Lifespan (modern startup/shutdown)

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    app.state.engine = create_async_engine(settings.database_url, pool_pre_ping=True)
    yield
    # Shutdown
    await app.state.engine.dispose()

app = FastAPI(lifespan=lifespan)
```

**Why**: `on_event("startup")` / `on_event("shutdown")` are **deprecated**. Lifespan also runs in tests under `TestClient(app) as c:`.
**Trap**: a `raise` before `yield` makes the app never start; FastAPI logs cryptically. Always yield.

---

## 3. Dependency injection — modern `Annotated[..., Depends(...)]`

```python
from typing import Annotated
from fastapi import Depends

# app/api/deps.py — type aliases collected in one place
SessionDep  = Annotated[AsyncSession, Depends(get_session)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
CurrentUser = Annotated[User, Depends(get_current_user)]

# routes use the alias directly
@router.get("/me")
async def me(user: CurrentUser): ...
```

**Why**: official FastAPI ≥ 0.95 recommendation. Tests override via `app.dependency_overrides[get_session] = test_session`.
**Trap**: `def endpoint(user: User = Depends(get_current_user))` still works but the type-alias form composes better and reads cleanly.

---

## 4. Async DB (SQLAlchemy 2.0)

```python
engine = create_async_engine(settings.database_url, pool_pre_ping=True, pool_size=10, max_overflow=20)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)

async def get_session() -> AsyncIterator[AsyncSession]:
    async with SessionLocal() as session:
        yield session
```

`async def` routes must do **non-blocking** I/O only. Sync route → FastAPI auto-runs it in a threadpool.

**N+1 prevention** — when a relationship is needed in the response:
```python
stmt = select(Order).options(selectinload(Order.items)).where(Order.user_id == uid)
```
Without `selectinload`, accessing `order.items` lazy-loads once **per row** → death by a thousand queries.
**Trap**: `expire_on_commit=False` is essential for async — ORM objects stay valid after `commit()`.

---

## 5. Validation (Pydantic v2)

```python
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator
from typing import Self

class TradeCreate(BaseModel):
    symbol:   str   = Field(..., min_length=1, max_length=12, examples=["AAPL"])
    quantity: float = Field(..., gt=0)
    price:    float = Field(..., gt=0)

    @field_validator("symbol")
    @classmethod
    def _uppercase(cls, v: str) -> str:
        return v.upper()

    @model_validator(mode="after")
    def _check_notional(self) -> Self:
        if self.quantity * self.price > 1_000_000:
            raise ValueError("notional > $1M requires approval")
        return self

class TradeRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)    # read from ORM attrs (was orm_mode in v1)
    id:       int
    symbol:   str
    quantity: float
    # NB: NO password_hash, NO email — response model controls what leaks
```

**Why**: validation at the edge → 422 with structured error detail. Read schemas explicitly enumerate fields → prevents data leak (OWASP API3 — property-level authz).
**Trap**: `from __future__ import annotations` in Pydantic model files has subtle edge cases with forward refs. Drop it unless needed.

---

## 6. Routes — status codes + response_model

```python
@router.post(
    "/trades",
    response_model=TradeRead,
    status_code=201,                                   # POST → 201 Created
    responses={409: {"description": "Duplicate trade"}},
)
async def create_trade(
    payload: TradeCreate,
    session: SessionDep,
    user: CurrentUser,
) -> TradeRead:
    return await TradeService(session).create(payload, owner_id=user.id)

@router.delete("/trades/{trade_id}", status_code=204)  # DELETE → 204 No Content
async def delete_trade(trade_id: int, session: SessionDep, user: CurrentUser) -> None:
    await TradeService(session).delete_owned(trade_id, owner_id=user.id)
```

| Verb | Default status | Notes |
|---|---|---|
| GET | 200 | 404 if not found |
| POST | **201** | not 200 |
| PUT / PATCH | 200 | full vs partial update |
| DELETE | **204** | no body |

**Trap**: returning a dict instead of the `response_model` type → no schema enforcement, anything could leak. Always declare `response_model`.

---

## 7. Auth — JWT + Argon2

```python
from pwdlib import PasswordHash
import jwt
from datetime import datetime, timezone, timedelta

pw_hasher = PasswordHash.recommended()                  # Argon2id

def hash_password(pw: str) -> str:
    return pw_hasher.hash(pw)

def verify_password(pw: str, h: str) -> bool:
    return pw_hasher.verify(pw, h)

def issue_token(uid: int, key: str, ttl_min: int = 60) -> str:
    now = datetime.now(timezone.utc)
    return jwt.encode(
        {"sub": str(uid), "iat": now, "exp": now + timedelta(minutes=ttl_min)},
        key, algorithm="HS256",
    )

def decode_token(token: str, key: str) -> dict:
    return jwt.decode(token, key, algorithms=["HS256"])  # raises on bad sig / expired
```

Wire it as a dependency:
```python
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")

async def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)],
    session: SessionDep,
    settings: SettingsDep,
) -> User:
    try:
        payload = decode_token(token, settings.secret_key)
    except jwt.PyJWTError:
        raise HTTPException(401, "Invalid token")
    user = await session.get(User, int(payload["sub"]))
    if user is None:
        raise HTTPException(401, "User not found")
    return user
```

**Why**: signature **and** expiry verified every protected request. Hashed passwords only — never store raw or sha256.
**Trap**: `bcrypt` has a 72-char limit and is slower than Argon2id. Default to Argon2 via `pwdlib`.

---

## 8. Authorization / ownership — close BOLA (OWASP API1)

```python
async def _get_owned(self, id: int, owner_id: int) -> Trade:
    t = await self.repo.get(id)
    if t is None or t.owner_id != owner_id:
        raise NotFoundError()                           # 404, NOT 403 — don't leak existence
    return t
```

The #1 API risk. Route **every single-object operation** through an ownership check. Returning 403 for "exists but not yours" tells attackers which IDs exist.

---

## 9. Error handling — no leaks, one map

```python
# app/exceptions.py
class DomainError(Exception): ...
class NotFoundError(DomainError): ...
class ConflictError(DomainError): ...

# app/main.py
@app.exception_handler(NotFoundError)
async def _not_found(req: Request, exc: NotFoundError):
    return JSONResponse(status_code=404, content={"error": str(exc) or "not found"})

@app.exception_handler(ConflictError)
async def _conflict(req: Request, exc: ConflictError):
    return JSONResponse(status_code=409, content={"error": str(exc)})
```

**Why**: services raise *domain* exceptions; HTTP mapping happens in **one place**. Services stay HTTP-agnostic + reusable.
**Trap**: never return stack traces, raw SQL, or framework internals in error bodies. Production = generic messages + structured logs for the real detail.

---

## 10. Config — pydantic-settings + lru_cache

```python
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    secret_key:    str
    database_url:  str
    log_level:     str = "INFO"
    cors_origins:  list[str] = ["http://localhost:3000"]

@lru_cache
def get_settings() -> Settings:                          # cached → parsed once per process
    return Settings()
```

**Why**: 12-factor — config via env, `.env` for local dev only. `@lru_cache` is the canonical FastAPI pattern (parsing env vars on every request is wasteful).
**Trap**: secrets in code → leaks via git history. `.env` is `.gitignore`d. CI/prod sets env directly.

---

## 11. CORS + security headers

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,                # NOT ["*"] — pin to your frontend(s)
    allow_credentials=True,                              # only if you actually need cookies
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)

# Security headers via middleware
from starlette.middleware.base import BaseHTTPMiddleware

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"]        = "DENY"
        response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains"
        return response

app.add_middleware(SecurityHeadersMiddleware)
```

**Trap**: `allow_origins=["*"]` + `allow_credentials=True` is silently blocked by browsers — pin origins.

---

## 12. Pagination + rate limiting (OWASP API4)

```python
@router.get("/trades", response_model=list[TradeRead])
async def list_trades(
    session: SessionDep, user: CurrentUser,
    limit: int = Query(50, ge=1, le=100),                # CAPPED page size
    cursor: int | None = None,
):
    return await TradeService(session).list_owned(user.id, limit, cursor)

# Rate limiting (slowapi)
from slowapi import Limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@router.post("/auth/login")
@limiter.limit("5/minute")                                # stricter on auth
async def login(...): ...
```

**Why**: an uncapped `limit` is a DoS surface (single request → million-row scan). Auth endpoints get tighter limits to throttle credential stuffing.
**Trap**: cursor pagination scales; `offset` pagination is OK for small data but slow at high offsets.

---

## 13. Background tasks (post-response fire-and-forget)

```python
from fastapi import BackgroundTasks

@router.post("/orders", status_code=201)
async def create_order(
    payload: OrderCreate, session: SessionDep, user: CurrentUser,
    background: BackgroundTasks,
):
    order = await OrderService(session).create(payload, owner_id=user.id)
    background.add_task(send_confirmation_email, order.id, user.email)
    return order                                          # response goes out BEFORE the task runs
```

**Why**: simple "log this / send this" work that shouldn't block the response. For heavy/durable work, use Celery / Arq / RQ + a Redis broker.
**Trap**: background tasks run in the same process — a crash mid-task is lost. Don't use this for "must run" work; queue it.

---

## 14. Streaming responses (LLM tokens, large CSVs)

```python
from fastapi.responses import StreamingResponse

@router.post("/chat")
async def chat(req: ChatRequest):
    async def token_stream():
        async for token in llm.stream(req.prompt):
            yield f"data: {token}\n\n"                    # SSE format
    return StreamingResponse(token_stream(), media_type="text/event-stream")
```

**Why**: caller sees tokens as they're generated, not 30s of nothing. Same pattern for paginated DB exports (`yield row.json() + "\n"`).
**Trap**: middleware that buffers (some reverse proxies) breaks streaming. Set `X-Accel-Buffering: no` for nginx.

---

## 15. Observability

- **Structured logs**: `structlog` → JSON to stdout · per-request correlation id (middleware sets `X-Request-ID`, echoed in response + every log line).
- **Tracing + metrics**: OpenTelemetry FastAPI + SQLAlchemy instrumentors → Tempo/Jaeger · Prometheus `/metrics` endpoint via `prometheus-fastapi-instrumentator`.
- **Probes**:
  - `GET /health` — liveness, NO DB call (returns 200 always)
  - `GET /ready`  — readiness, runs `SELECT 1` against DB (503 until ready)

**Trap**: putting a DB query in `/health` couples liveness to DB availability — k8s kills the pod every time the DB hiccups.

---

## 16. Testing

```python
import pytest
from httpx import AsyncClient, ASGITransport

@pytest.fixture
async def client(test_session_factory):
    app.dependency_overrides[get_session] = lambda: test_session_factory()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()

# In-memory SQLite with StaticPool keeps the same connection across the test
engine = create_async_engine("sqlite+aiosqlite:///:memory:", poolclass=StaticPool, connect_args={"check_same_thread": False})
```

**Cover**:
| Test | Asserts |
|---|---|
| happy path | 200/201, response schema, side effects |
| validation | 422 on bad payload + error detail mentions the field |
| auth required | 401 without token, 200 with valid token |
| **BOLA** | 404 when User A tries to access User B's resource |
| rate limit | 429 after the configured limit |

**Trap**: forget `app.dependency_overrides.clear()` → next test sees the override. Use a fixture with `yield`.

---

## 17. Deploy

```dockerfile
# multi-stage Dockerfile
FROM python:3.12-slim AS build
RUN pip install uv && uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

FROM python:3.12-slim
RUN useradd -m app
COPY --from=build /opt/venv /opt/venv
COPY --chown=app:app app/ /app/app/
USER app
HEALTHCHECK CMD curl -f http://localhost:8000/health || exit 1
CMD ["gunicorn", "app.main:app", "-k", "uvicorn.workers.UvicornWorker", "-w", "4", "--bind", "0.0.0.0:8000"]
```

- `gunicorn -w <CPU count>` for process-level parallelism.
- Non-root user (`USER app`).
- Migrations via Alembic — run as an init container / pre-deploy step, **not** in the app's lifespan.

---

## 18. OWASP API Top 10 (2023) — coverage checklist

| ID | Risk | This doc's answer |
|---|---|---|
| API1 | BOLA | §8 — every single-object op routed through `_get_owned` |
| API2 | Broken auth | §7 — JWT signature + expiry, Argon2 passwords |
| API3 | Broken property-level authz | §5/§6 — explicit `response_model` (no `**user.dict()`) |
| API4 | Resource consumption | §12 — capped pagination + rate limit |
| API5 | Broken function-level authz | role check via dep (`RequireAdmin`) |
| API6 | Server-side request forgery | validate URLs in payloads if you fetch them |
| API7 | Misconfiguration | §10/§11 — env config, locked CORS |
| API8 | Lack of inventory | use OpenAPI tags + version dirs (`v1/`, `v2/`) |
| API9 | Improper inventory | log retired endpoints, deprecate gracefully |
| API10 | Unsafe consumption of APIs | validate upstream API responses with Pydantic too |

---

## 19. Async do/don't

| ✅ Do | ❌ Don't |
|---|---|
| `await asyncpg`, `await httpx.AsyncClient` | `requests.get(...)` inside `async def` |
| `await asyncio.sleep(...)` | `time.sleep(...)` inside `async def` |
| Long CPU work in a threadpool: `await run_in_threadpool(fn, ...)` | Heavy CPU directly in `async def` |
| Sync route + threadpool (FastAPI does it for you) | "Make everything async because async is faster" |

**Rule**: any blocking call inside `async def` freezes the event loop → no concurrency. Either go non-blocking or use `run_in_threadpool` / a worker process.

---

## 20. Anti-patterns to recognise instantly

| Smell | Fix |
|---|---|
| `@app.on_event("startup")` | Use `lifespan` context manager (§2) |
| `dep: T = Depends(...)` in every signature | Type alias `TDep = Annotated[T, Depends(...)]` once (§3) |
| `from sqlalchemy.orm import Session` (sync) in async routes | `AsyncSession` + `async_sessionmaker` (§4) |
| `class Config: orm_mode = True` | `model_config = ConfigDict(from_attributes=True)` (§5) |
| `dict(user)` or `**user.__dict__` as response | Declare `response_model=` (§6) |
| `bcrypt.hashpw(...)` | `pwdlib.PasswordHash.recommended()` (Argon2, §7) |
| Raise `HTTPException(403)` on "not yours" | Raise `NotFoundError` → 404 (§8) |
| `try/except Exception:` returning the message | Domain exception + global handler (§9) |
| `os.getenv("SECRET_KEY")` scattered in routes | One `Settings` class + `@lru_cache` (§10) |
| `allow_origins=["*"]` | Pin origins (§11) |
| `limit: int = Query(50)` with no `le` cap | Cap with `le=100` (§12) |
| Long-running work in route handler | `BackgroundTasks` (§13) or external queue |
| `response: Response` to build a streamed body | `StreamingResponse` (§14) |
| DB query in `/health` | `/health` no-DB, `/ready` runs `SELECT 1` (§15) |
| `from __future__ import annotations` in Pydantic model files | Drop it — Pydantic v2 prefers concrete annotations |
| Alembic migration in lifespan | Run as init container / pre-deploy step |
| Logging passwords / tokens / PII | Redact in formatter; structlog has processors for this |

---

## 21. Cross-references

- Worked walkthrough: `tutorial/`
- Capstone (skeleton + TODOs): `capstone/`
- ML pipeline patterns: `../toolkit/ml_project_methodology.md`
- MLflow patterns for ML serving: `../toolkit/mlflow_cheatsheet.md`
- Production wrapping (FastAPI + MLflow): `../quant_finance/capstones/lmm_nn_surrogate/api_extension/`
