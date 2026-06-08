# API Engineering Cheatsheet (Production FastAPI, 2026)

Condensed reference for production REST APIs. **FastAPI ≥ 0.115, Pydantic v2,
SQLAlchemy 2.0, Python 3.12+**. See `tutorial/` for the walkthrough and
`capstone/` for the exercise. Each section: paste-able code → why → trap.

**Modern conventions worth knowing upfront** — what the current docs recommend:
- **Lifespan** context manager, not `on_event("startup")`.
- **`Annotated[T, Depends(...)]`** type-alias DI, not `T = Depends(...)`.
- **SQLAlchemy 2.0 declarative** (`Mapped[X]` + `mapped_column`), not legacy `Column()`.
- **`select(...) + session.execute()`**, not `session.query()`.
- **Pydantic v2 typed fields** (`SecretStr`, `EmailStr`, `HttpUrl`), not bare `str`.
- **`fastapi.status` constants** (`status.HTTP_201_CREATED`), not raw ints.
- **Argon2id via `pwdlib`**, not `bcrypt` directly.

---

## 0. The general pattern — every endpoint, every API

Internalise this skeleton; everything else in this doc is a variation on
it. The pattern is the same regardless of the resource you're exposing.

### Per-endpoint pattern (5 steps, always the same)

```python
@router.post(                              # 1. DECLARE — verb, path, response shape, status
    "/trades",
    response_model=TradeRead,
    status_code=status.HTTP_201_CREATED,
)
async def create_trade(
    payload: TradeCreate,                  # 2. VALIDATE — Pydantic request body @ the edge
    session: SessionDep,                   # 3. INJECT — Annotated deps (DB, settings, user)
    user:    CurrentUser,                  #    auth + identity resolved in the dep
) -> TradeRead:
    return await TradeService(             # 4. DELEGATE — service does the actual work
        session
    ).create(payload, owner_id=user.id)    # 5. RETURN — value that conforms to response_model
```

### Project-level wiring (once at startup, in `main.py`)

```
1. settings   → pydantic-settings + @lru_cache (read env, .env in dev)
2. lifespan   → create_async_engine + sessionmaker on startup; dispose on shutdown
3. middleware → CORS, TrustedHost, GZip, security headers (order = onion, last in = first hit)
4. routers    → include versioned modules (app/api/v1/...) with prefix + tags
5. handlers   → register exception handlers (NotFoundError → 404, ConflictError → 409, ...)
```

### The request flow (mental model)

```
REQUEST IN
  ↓
middleware       (CORS, rate limit, security headers, request-id)
  ↓
route function   (FastAPI resolves Depends graph, validates payload via Pydantic)
  ↓
service          (business logic, authorization, calls repo)
  ↓
repository       (SQL via select() + execute, returns ORM objects)
  ↓
service returns  → route returns → response_model validates → response_class serialises
  ↓
RESPONSE OUT
```

If anything **raises** along this chain, the exception propagates UP
until it's caught by an exception handler (§10) or becomes a generic 500.
Services raise **domain** exceptions (`NotFoundError`, `ConflictError`),
never `HTTPException` — the HTTP mapping is the routes' job.

### Layer responsibility — the only rules that matter

| Layer | Knows | Doesn't know |
|---|---|---|
| **routes** (`app/api/v1/`) | HTTP (status, response_model, query/path/header params) | SQL, business rules |
| **services** (`app/services.py`) | business logic, authorization, transactions | HTTP, raw SQL |
| **repositories** (`app/repositories.py`) | SQL (`select(...) + execute`) | business rules, HTTP |
| **schemas** (`app/schemas.py`) | I/O shapes + validators (Pydantic) | anything stateful |
| **models** (`app/models.py`) | ORM mapping (`Mapped[]` + `mapped_column`) | application logic |

If a layer touches something in the wrong column → refactor. **One smell to
recognise instantly**: a route that imports `select` from SQLAlchemy → SQL
leaked out of the repository.

### The Pydantic-at-the-boundary contract

```
inbound  JSON  →  request schema  (TradeCreate)   →  validated dict   →  service
                       ↑ 422 on bad data
                                                                    ↓
outbound JSON  ←  response_model  (TradeRead)     ←  ORM / dict      ←  service
                       ↑ filters to declared fields only (no password_hash leak)
```

Pydantic does double duty: **enforces input shape** (catches malformed
requests before they reach your code) AND **constrains output shape**
(prevents accidental data leaks via `**user.__dict__`). Every endpoint
should declare both.

### Variations covered below

- Different verbs / status codes / OpenAPI docs → §7
- Auth + login + refresh tokens → §8
- Role-based gating beyond ownership → §9
- Streaming / SSE / WebSocket responses → §15
- Background work after the response → §14
- Pagination & rate limiting → §13

The rest of this doc fills in the details for each of those choice points.

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
from fastapi import Depends, Path, Query, Header, Cookie

# app/api/deps.py — type aliases collected in one place
SessionDep  = Annotated[AsyncSession, Depends(get_session)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
CurrentUser = Annotated[User, Depends(get_current_user)]

# Path / Query / Header / Cookie parameters — also via Annotated
@router.get("/trades/{trade_id}")
async def get_trade(
    trade_id: Annotated[int, Path(ge=1, description="Trade ID")],
    include_meta: Annotated[bool, Query(description="Include audit metadata")] = False,
    user_agent:   Annotated[str | None, Header()] = None,
    session_id:   Annotated[str | None, Cookie()] = None,
    user:         CurrentUser,
    session:      SessionDep,
): ...
```

**Why**: official FastAPI ≥ 0.95 recommendation. Tests override via `app.dependency_overrides[get_session] = test_session`. The `Annotated` form composes (chain validators, swap impl, share across routes).
**Trap**: `def endpoint(user: User = Depends(get_current_user))` still works but Annotated reads better and tooling (mypy, pyright) understands it more cleanly.

---

## 4. ORM models — SQLAlchemy 2.0 declarative

```python
from datetime import datetime
from sqlalchemy import ForeignKey, String, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

class Base(DeclarativeBase): ...

class User(Base):
    __tablename__ = "users"
    id:            Mapped[int]      = mapped_column(primary_key=True)
    email:         Mapped[str]      = mapped_column(String(255), unique=True, index=True)
    password_hash: Mapped[str]
    created_at:    Mapped[datetime] = mapped_column(server_default=func.now())
    trades:        Mapped[list["Trade"]] = relationship(back_populates="owner")

class Trade(Base):
    __tablename__ = "trades"
    id:        Mapped[int]   = mapped_column(primary_key=True)
    symbol:    Mapped[str]   = mapped_column(String(12), index=True)
    quantity:  Mapped[float]
    price:     Mapped[float]
    owner_id:  Mapped[int]   = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    owner:     Mapped["User"] = relationship(back_populates="trades")
```

**Why**: `Mapped[X]` + `mapped_column` is the **2.0 declarative style** — type-hint-first, IDE/mypy-aware, replaces the legacy `Column(Integer, primary_key=True)` signature. `relationship` returns typed `Mapped[list[...]]`.
**Trap**: missing `back_populates` on one side of a relationship → silent unidirectional behaviour. Always set both sides.

---

## 5. Async DB sessions + 2.0-style queries

```python
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy import select
from sqlalchemy.orm import selectinload

engine = create_async_engine(
    settings.database_url, pool_pre_ping=True, pool_size=10, max_overflow=20,
)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)

async def get_session() -> AsyncIterator[AsyncSession]:
    async with SessionLocal() as session:
        yield session

# 2.0-STYLE QUERIES — select() + session.execute()
async def get_user_trades(session: AsyncSession, uid: int) -> list[Trade]:
    stmt = (
        select(Trade)
        .where(Trade.owner_id == uid)
        .options(selectinload(Trade.owner))     # N+1 prevention (eager load)
        .order_by(Trade.id.desc())
        .limit(50)
    )
    result = await session.execute(stmt)
    return list(result.scalars().all())          # .scalars() unwraps the row tuples

# Single result variants
await session.execute(stmt).scalar_one()         # exactly one row, else raises
await session.execute(stmt).scalar_one_or_none() # at most one, else None
```

`async def` routes must do **non-blocking** I/O only. Sync route → FastAPI auto-runs it in a threadpool.

**Why `selectinload`**: without it, `for t in trades: print(t.owner.email)` lazy-loads once per trade → N+1.
**Trap**: `expire_on_commit=False` is essential for async — without it, ORM objects expire after `commit()` and accessing attributes triggers re-fetch (often fails outside a session).
**Trap**: `session.query(...)` is legacy 1.x. Modern is `session.execute(select(...))`. The query-builder syntax is otherwise the same.

---

## 6. Validation (Pydantic v2 — modern types + validators)

```python
from typing import Self
from pydantic import (
    BaseModel, ConfigDict, EmailStr, Field, HttpUrl, SecretStr,
    computed_field, field_validator, model_validator,
)

class UserCreate(BaseModel):
    email:    EmailStr                                 # auto-validates RFC 5322
    password: SecretStr   = Field(..., min_length=12)  # SecretStr hides in __repr__/__str__
    website:  HttpUrl | None = None                    # auto-validates URL

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
    model_config = ConfigDict(from_attributes=True)    # was orm_mode = True in v1
    id:       int
    symbol:   str
    quantity: float
    price:    float

    @computed_field                                    # serialised but not a model field
    @property
    def notional(self) -> float:
        return self.quantity * self.price
```

**Useful Pydantic v2 idioms**:
- `model_dump()` — dict (was `.dict()`).
- `model_dump_json()` — JSON string (was `.json()`).
- `model_validate(obj)` — parse from dict / ORM object (was `parse_obj()`).
- `model_validate_json(s)` — parse from JSON string directly (skips dict middle step).
- `Field(..., alias="userId")` — accept `userId` on input; `serialization_alias="userId"` — emit `userId` on output.

**Why**: validation at the edge → 422 with structured detail. Read schemas explicitly enumerate fields → prevents data leak (OWASP API3 — property-level authz). `SecretStr` keeps secrets out of logs/repr.
**Trap**: `from __future__ import annotations` in Pydantic model files has subtle edge cases with forward refs. Drop it unless needed.

---

## 7. Routes — status codes, response_model, OpenAPI customization

```python
from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/trades", tags=["trades"])

@router.post(
    "",
    response_model=TradeRead,
    status_code=status.HTTP_201_CREATED,                # prefer constants over raw ints
    summary="Create a new trade",
    description="Validates notional cap of $1M and stamps the owner.",
    responses={
        status.HTTP_409_CONFLICT: {"description": "Duplicate trade"},
    },
)
async def create_trade(
    payload: TradeCreate,
    session: SessionDep,
    user: CurrentUser,
) -> TradeRead:
    return await TradeService(session).create(payload, owner_id=user.id)

@router.patch(
    "/{trade_id}",
    response_model=TradeRead,
    response_model_exclude_unset=True,                  # don't emit fields the client didn't update
)
async def update_trade(trade_id: int, payload: TradePartialUpdate, session: SessionDep, user: CurrentUser):
    return await TradeService(session).update_owned(trade_id, payload, owner_id=user.id)

@router.delete(
    "/{trade_id}",
    status_code=status.HTTP_204_NO_CONTENT,             # DELETE → 204
)
async def delete_trade(trade_id: int, session: SessionDep, user: CurrentUser) -> None:
    await TradeService(session).delete_owned(trade_id, owner_id=user.id)

@router.get("/internal", include_in_schema=False)       # hide from /docs
async def internal_metrics(): ...

@router.get("/legacy", deprecated=True)                 # mark in /docs, still callable
async def legacy(): ...
```

| Verb | Default status | Notes |
|---|---|---|
| GET | `HTTP_200_OK` | 404 if not found |
| POST | **`HTTP_201_CREATED`** | not 200 |
| PUT / PATCH | `HTTP_200_OK` | full vs partial update |
| DELETE | **`HTTP_204_NO_CONTENT`** | no body |

**Trap**: returning a dict instead of the `response_model` type → no schema enforcement, anything could leak. Always declare `response_model`.

---

## 8. Auth — JWT + Argon2 + full login flow

```python
from typing import Annotated
from datetime import datetime, timezone, timedelta

import jwt                                              # PyJWT package (not python-jose)
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pwdlib import PasswordHash

pw_hasher = PasswordHash.recommended()                  # Argon2id

def hash_password(pw: str) -> str:        return pw_hasher.hash(pw)
def verify_password(pw: str, h: str)->bool: return pw_hasher.verify(pw, h)

def issue_token(uid: int, key: str, ttl_min: int = 60, token_type: str = "access") -> str:
    now = datetime.now(timezone.utc)
    return jwt.encode(
        {"sub": str(uid), "iat": now, "exp": now + timedelta(minutes=ttl_min), "type": token_type},
        key, algorithm="HS256",
    )

# ── Dependency that pulls the current user from the bearer token ──
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")

async def get_current_user(
    token:    Annotated[str, Depends(oauth2_scheme)],
    session:  SessionDep,
    settings: SettingsDep,
) -> User:
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=["HS256"])
        if payload.get("type") != "access":             # refuse refresh-token use on protected routes
            raise jwt.PyJWTError("wrong token type")
    except jwt.PyJWTError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid token")
    user = await session.get(User, int(payload["sub"]))
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "User not found")
    return user

# ── Login endpoint (issues access + refresh) ──
auth_router = APIRouter(prefix="/auth", tags=["auth"])

@auth_router.post("/login")
async def login(
    form: Annotated[OAuth2PasswordRequestForm, Depends()],    # username + password from form data
    session: SessionDep, settings: SettingsDep,
) -> dict:
    user = await UserRepo(session).get_by_email(form.username)
    if user is None or not verify_password(form.password, user.password_hash):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Bad credentials")
    return {
        "access_token":  issue_token(user.id, settings.secret_key, ttl_min=15, token_type="access"),
        "refresh_token": issue_token(user.id, settings.secret_key, ttl_min=60*24*7, token_type="refresh"),
        "token_type":    "bearer",
    }

@auth_router.post("/refresh")
async def refresh(refresh_token: str, settings: SettingsDep) -> dict:
    try:
        payload = jwt.decode(refresh_token, settings.secret_key, algorithms=["HS256"])
        if payload.get("type") != "refresh":
            raise jwt.PyJWTError("wrong token type")
    except jwt.PyJWTError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid refresh token")
    return {
        "access_token": issue_token(int(payload["sub"]), settings.secret_key, ttl_min=15, token_type="access"),
        "token_type":   "bearer",
    }
```

**Why**: signature **and** expiry verified every protected request. Hashed passwords only — never store raw or sha256. Access token TTL short (15 min); refresh token longer (days/weeks); refresh endpoint trades a valid refresh token for a fresh access token. The `type` claim prevents callers using a refresh token on a protected route.
**Trap**: `bcrypt` has a 72-char limit and is slower than Argon2id. **`pwdlib`** is the FastAPI docs' recommended package.
**Trap**: `import jwt` resolves to **PyJWT**, not `python-jose`. The latter is somewhat unmaintained; PyJWT is the active choice.

---

## 9. Authorization / ownership — close BOLA (OWASP API1)

```python
async def _get_owned(self, id: int, owner_id: int) -> Trade:
    t = await self.repo.get(id)
    if t is None or t.owner_id != owner_id:
        raise NotFoundError()                           # 404, NOT 403 — don't leak existence
    return t
```

The #1 API risk. Route **every single-object operation** through an ownership check. Returning 403 for "exists but not yours" tells attackers which IDs exist (timing differences also leak — fast 404 = doesn't exist; slow 404 = exists but not yours; equalise where possible).

For role-based gates, layer a separate dep on top:
```python
async def require_admin(user: CurrentUser) -> User:
    if user.role != "admin":
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Admin only")
    return user

AdminUser = Annotated[User, Depends(require_admin)]

@router.delete("/users/{uid}", status_code=status.HTTP_204_NO_CONTENT)
async def hard_delete(uid: int, admin: AdminUser, session: SessionDep): ...
```

---

## 10. Error handling — no leaks, one map

```python
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

# app/exceptions.py
class DomainError(Exception): ...
class NotFoundError(DomainError): ...
class ConflictError(DomainError): ...
class ForbiddenError(DomainError): ...

# app/main.py — map domain → HTTP in ONE place
@app.exception_handler(NotFoundError)
async def _not_found(req: Request, exc: NotFoundError):
    return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"error": str(exc) or "not found"})

@app.exception_handler(ConflictError)
async def _conflict(req: Request, exc: ConflictError):
    return JSONResponse(status_code=status.HTTP_409_CONFLICT, content={"error": str(exc)})
```

**Why**: services raise *domain* exceptions; HTTP mapping happens in **one place**. Services stay HTTP-agnostic + reusable (a CLI / scheduled job can call them without dragging FastAPI in).
**Trap**: never return stack traces, raw SQL, or framework internals in error bodies. Production = generic messages + structured logs for the real detail.

---

## 11. Config — pydantic-settings + lru_cache

```python
from functools import lru_cache
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", env_prefix="APP_")
    secret_key:   SecretStr                              # SecretStr keeps it out of logs/repr
    database_url: str
    log_level:    str = "INFO"
    cors_origins: list[str] = ["http://localhost:3000"]

@lru_cache
def get_settings() -> Settings:                          # cached → parsed once per process
    return Settings()

# Use:
settings = get_settings()
jwt.encode(payload, settings.secret_key.get_secret_value(), algorithm="HS256")  # .get_secret_value() to unwrap
```

**Why**: 12-factor — config via env, `.env` for local dev only. `@lru_cache` is the canonical FastAPI pattern (parsing env vars on every request is wasteful).
**Trap**: secrets in code → leaks via git history. `.env` is `.gitignore`d. CI/prod sets env directly. Use `SecretStr` so a stray `print(settings)` doesn't dump the secret.

---

## 12. Middleware stack — CORS + security + GZip + HTTPS

```python
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

# CORS (pin origins, never "*" with credentials)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,                 # NOT ["*"] — pin to your frontend(s)
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)

# Reject Host header spoofing (prevents reverse-proxy bypass tricks)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["api.example.com", "*.example.com"])

# Force HTTPS (only behind a reverse proxy that terminates TLS for you)
if settings.env == "prod":
    app.add_middleware(HTTPSRedirectMiddleware)

# Compress responses (negligible CPU, big bandwidth saving for JSON)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Custom security headers
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"]    = "nosniff"
        response.headers["X-Frame-Options"]           = "DENY"
        response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains; preload"
        response.headers["Referrer-Policy"]           = "strict-origin-when-cross-origin"
        return response
app.add_middleware(SecurityHeadersMiddleware)
```

**Middleware order matters** — added LAST runs FIRST on requests, LAST on responses (think onion).
**Trap**: `allow_origins=["*"]` + `allow_credentials=True` is silently blocked by browsers — pin origins.

---

## 13. Pagination + rate limiting (OWASP API4)

```python
from fastapi import Query
from slowapi import Limiter
from slowapi.util import get_remote_address

@router.get("/trades", response_model=list[TradeRead])
async def list_trades(
    session: SessionDep, user: CurrentUser,
    limit: int = Query(50, ge=1, le=100),                # CAPPED page size
    cursor: int | None = None,
):
    return await TradeService(session).list_owned(user.id, limit, cursor)

# Rate limiting (slowapi — in-process; for distributed use fastapi-limiter + Redis)
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@router.post("/auth/login")
@limiter.limit("5/minute")                                # stricter on auth
async def login(...): ...
```

**Why**: an uncapped `limit` is a DoS surface (single request → million-row scan). Auth endpoints get tighter limits to throttle credential stuffing.
**Trap**: cursor pagination scales; `offset` pagination is OK for small data but slow at high offsets.

---

## 14. Background tasks (post-response fire-and-forget)

```python
from fastapi import BackgroundTasks

@router.post("/orders", status_code=status.HTTP_201_CREATED)
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

## 15. Streaming, SSE, WebSockets (real-time)

### Streaming response (NDJSON / large CSV)
```python
from fastapi.responses import StreamingResponse

@router.get("/export")
async def export_trades(session: SessionDep):
    async def rows():
        async for row in session.stream(select(Trade)):
            yield row.Trade.model_dump_json() + "\n"     # NDJSON
    return StreamingResponse(rows(), media_type="application/x-ndjson")
```

### Server-Sent Events (LLM tokens) — use `sse-starlette` for correct framing
```python
from sse_starlette.sse import EventSourceResponse

@router.post("/chat")
async def chat(req: ChatRequest):
    async def gen():
        async for token in llm.stream(req.prompt):
            yield {"data": token}                         # auto-formats `data: ...\n\n`
    return EventSourceResponse(gen())
```

### WebSockets (bidirectional, persistent connection)
```python
from fastapi import WebSocket, WebSocketDisconnect

@app.websocket("/ws/{room}")
async def ws_room(ws: WebSocket, room: str):
    await ws.accept()
    try:
        while True:
            msg = await ws.receive_text()
            await ws.send_text(f"[{room}] {msg}")
    except WebSocketDisconnect:
        await on_disconnect(room)
```

**Why WS over SSE**: WS is bidirectional; SSE is server→client only. For LLM streaming, SSE is enough and simpler.
**Trap**: middleware that buffers (some reverse proxies) breaks streaming and SSE. For nginx: `proxy_buffering off;` or send `X-Accel-Buffering: no`.
**Trap**: WebSocket auth — FastAPI doesn't run `Depends(get_current_user)` automatically; pass the token via query string or first-message handshake.

---

## 16. Observability

- **Structured logs**: `structlog` → JSON to stdout · per-request correlation id (middleware sets `X-Request-ID`, echoed in response + every log line).
- **Tracing + metrics**: OpenTelemetry FastAPI + SQLAlchemy instrumentors → Tempo/Jaeger · Prometheus `/metrics` endpoint via `prometheus-fastapi-instrumentator`.
- **Probes**:
  - `GET /health` — liveness, NO DB call (returns 200 always)
  - `GET /ready`  — readiness, runs `SELECT 1` against DB (503 until ready)

**Trap**: putting a DB query in `/health` couples liveness to DB availability — k8s kills the pod every time the DB hiccups.

---

## 17. Testing

```python
import pytest
from httpx import AsyncClient, ASGITransport
from sqlalchemy.pool import StaticPool

# In-memory SQLite with StaticPool keeps the same connection across the test
test_engine = create_async_engine(
    "sqlite+aiosqlite:///:memory:",
    poolclass=StaticPool, connect_args={"check_same_thread": False},
)

@pytest.fixture
async def client(test_session_factory):
    app.dependency_overrides[get_session] = lambda: test_session_factory()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()                    # CRUCIAL — leaks across tests otherwise
```

**Cover**:
| Test | Asserts |
|---|---|
| happy path | 200/201, response schema, side effects |
| validation | 422 on bad payload + error detail mentions the field |
| auth required | 401 without token, 200 with valid token |
| **BOLA** | 404 when User A tries to access User B's resource |
| rate limit | 429 after the configured limit |
| OpenAPI | `/docs` loads, expected paths present |

**Trap**: forget `app.dependency_overrides.clear()` → next test sees the override.

---

## 18. Deploy

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

- `gunicorn -w <CPU count>` for process-level parallelism; alternative: `uvicorn --workers N` directly (FastAPI docs now mention both).
- Non-root user (`USER app`).
- Migrations via Alembic — run as an init container / pre-deploy step, **not** in the app's lifespan.

---

## 19. OWASP API Top 10 (2023) — coverage checklist

| ID | Risk | This doc's answer |
|---|---|---|
| API1 | BOLA | §9 — every single-object op routed through `_get_owned` |
| API2 | Broken auth | §8 — JWT signature + expiry, Argon2 passwords, access+refresh split |
| API3 | Broken property-level authz | §6/§7 — explicit `response_model` (no `**user.dict()`) |
| API4 | Resource consumption | §13 — capped pagination + rate limit |
| API5 | Broken function-level authz | role check via dep (`AdminUser`) |
| API6 | Server-side request forgery | validate URLs in payloads (Pydantic `HttpUrl` + allowlist before fetching) |
| API7 | Misconfiguration | §11/§12 — env config, locked CORS + TrustedHost + HTTPS |
| API8 | Lack of inventory | use OpenAPI tags + version dirs (`v1/`, `v2/`) |
| API9 | Improper inventory | log retired endpoints, `deprecated=True` flag, sunset header |
| API10 | Unsafe consumption of APIs | validate upstream API responses with Pydantic too |

---

## 20. Async do/don't

| ✅ Do | ❌ Don't |
|---|---|
| `await asyncpg`, `await httpx.AsyncClient` | `requests.get(...)` inside `async def` |
| `await asyncio.sleep(...)` | `time.sleep(...)` inside `async def` |
| Long CPU work in a threadpool: `await run_in_threadpool(fn, ...)` | Heavy CPU directly in `async def` |
| Sync route + threadpool (FastAPI does it for you) | "Make everything async because async is faster" |

**Rule**: any blocking call inside `async def` freezes the event loop → no concurrency. Either go non-blocking or use `run_in_threadpool` / a worker process.

---

## 21. Anti-patterns to recognise instantly

| Smell | Fix |
|---|---|
| `@app.on_event("startup")` | Use `lifespan` context manager (§2) |
| `dep: T = Depends(...)` in every signature | Type alias `TDep = Annotated[T, Depends(...)]` once (§3) |
| `Column(Integer, primary_key=True)` | `Mapped[int] = mapped_column(primary_key=True)` (§4) |
| `session.query(Model).filter(...)` | `select(Model).where(...)` + `await session.execute(...)` (§5) |
| `from sqlalchemy.orm import Session` (sync) in async routes | `AsyncSession` + `async_sessionmaker` (§5) |
| `class Config: orm_mode = True` | `model_config = ConfigDict(from_attributes=True)` (§6) |
| `password: str` in a model | `password: SecretStr` (§6) — keeps out of logs/repr |
| `email: str` with manual regex | `email: EmailStr` (§6) |
| `dict(user)` / `**user.__dict__` as response | Declare `response_model=` (§7) |
| `status_code=201` (raw int) | `status_code=status.HTTP_201_CREATED` (§7) |
| `bcrypt.hashpw(...)` | `pwdlib.PasswordHash.recommended()` (Argon2, §8) |
| `from jose import jwt` | `import jwt` (PyJWT — active package) (§8) |
| One token type for everything | Access (short) + refresh (long), distinguished by `type` claim (§8) |
| Raise `HTTPException(403)` on "not yours" | Raise `NotFoundError` → 404 (§9) |
| `try/except Exception:` returning the message | Domain exception + global handler (§10) |
| `os.getenv("SECRET_KEY")` scattered in routes | One `Settings` class + `@lru_cache` (§11) |
| Plain `str` for secrets in Settings | `SecretStr` + `.get_secret_value()` (§11) |
| `allow_origins=["*"]` with `allow_credentials=True` | Pin origins (§12) |
| No `TrustedHostMiddleware` in prod | Add it — prevents Host header tricks (§12) |
| `limit: int = Query(50)` with no `le` cap | Cap with `le=100` (§13) |
| Long-running work in route handler | `BackgroundTasks` (§14) or external queue |
| Manual `data: ...\n\n` SSE string formatting | `EventSourceResponse` from `sse-starlette` (§15) |
| `Depends(get_current_user)` on a WebSocket route | Manual token extraction from query/first msg (§15) |
| DB query in `/health` | `/health` no-DB, `/ready` runs `SELECT 1` (§16) |
| `from __future__ import annotations` in Pydantic model files | Drop it — Pydantic v2 prefers concrete annotations |
| Alembic migration in lifespan | Run as init container / pre-deploy step |
| Logging passwords / tokens / PII | Redact in formatter; structlog has processors for this |

---

## 22. Cross-references

- Worked walkthrough: `tutorial/`
- Capstone (skeleton + TODOs): `capstone/`
- ML pipeline patterns: `../toolkit/ml_project_methodology.md`
- MLflow patterns for ML serving: `../toolkit/mlflow_cheatsheet.md`
- SQL reference (Postgres ≥ 14): `../sql/sql_cheatsheet.md`
- Production wrapping (FastAPI + MLflow): `../quant_finance/capstones/lmm_nn_surrogate/api_extension/`
