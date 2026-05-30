"""Application entry point: builds and configures the FastAPI app.

WHAT THIS FILE IS — the assembler, built LAST because it imports everything else.
`app` at module scope is what uvicorn runs: `uvicorn app.main:app`. `create_app()`
puts the pieces together; `lifespan` runs startup/shutdown work.

DOCS:
  app factory + include_router: https://fastapi.tiangolo.com/tutorial/bigger-applications/
  lifespan events: https://fastapi.tiangolo.com/advanced/events/

RUN IT (after every file is implemented):  uv run uvicorn app.main:app --reload
  then open http://localhost:8000/docs
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup runs before the first request; shutdown runs on exit.

    TODO (core) — create the tables in dev (SQLite), then dispose the engine:
        from app.db.session import Base, engine
        import app.models            # IMPORTANT: import models so Base.metadata knows the tables
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        yield
        await engine.dispose()
    (Stretch: use Alembic migrations instead of create_all.)
    """
    from app.db.session import Base, engine
    import app.models            # IMPORTANT: import models so Base.metadata knows the tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    await engine.dispose()


def create_app() -> FastAPI:
    """Assemble and return the app.

    TODO (core) — in this order:
        from app.api.v1.router import api_router
        from app.api.v1 import ops
        from app.exceptions import register_exception_handlers

        app = FastAPI(title="Trades Ledger API", lifespan=lifespan)
        register_exception_handlers(app)     # domain errors -> clean HTTP responses
        app.include_router(api_router)        # mounts /api/v1/auth/* and /api/v1/trades/*
        app.include_router(ops.router)        # mounts /health at the root
        return app

    (Stretch, after core works: CORSMiddleware, the slowapi limiter, observability.)
    """
    from app.api.v1.router import api_router
    from app.api.v1 import ops
    from app.exceptions import register_exception_handlers

    app = FastAPI(title="Trades Ledger API", lifespan=lifespan)
    register_exception_handlers(app)     # domain errors -> clean HTTP responses
    app.include_router(api_router)        # mounts /api/v1/auth/* and /api/v1/trades/*
    app.include_router(ops.router)        # mounts /health at the root
    return app


app = create_app()
