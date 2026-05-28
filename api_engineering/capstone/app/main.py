"""Application entry point: builds and configures the FastAPI app.

`app` at module scope is what uvicorn/gunicorn import (`app.main:app`).
Implement create_app() to assemble the pieces in the right order.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown hooks.

    Startup: configure_logging(settings.log_level). For the 4-hour CORE you may
    create tables here in dev (Base.metadata.create_all via the async engine);
    for the stretch version use Alembic migrations instead.
    Shutdown: dispose the async engine.
    """
    # TODO (startup): configure logging, optionally create tables (dev only).
    yield
    # TODO (shutdown): await engine.dispose()


def create_app() -> FastAPI:
    """Assemble the application.

    Suggested order:
      1. app = FastAPI(title="Trades Ledger API", lifespan=lifespan)
      2. CORSMiddleware (settings.cors_origin_list) + CorrelationIdMiddleware
      3. register_exception_handlers(app)        # from app.exceptions
      4. include api_router (/api/v1) and the ops router (root)
      --- stretch ---
      5. attach deps.limiter to app.state + slowapi's exception handler
      6. setup_observability(app)                 # OTel tracing + /metrics
    """
    # TODO: implement per the docstring and return the app.
    raise NotImplementedError


app = create_app()
