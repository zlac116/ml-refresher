"""Structured logging, request correlation ids, and OpenTelemetry wiring.

Three pieces, all called from main.py:
  1. configure_logging()        -- structlog -> JSON logs.
  2. CorrelationIdMiddleware    -- one X-Request-ID per request, bound into logs.
  3. setup_observability(app)   -- OpenTelemetry (FastAPI + SQLAlchemy) + /metrics.
"""
from starlette.middleware.base import BaseHTTPMiddleware

# Suggested imports as you implement:
#   import structlog
#   from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
#   from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
#   from prometheus_fastapi_instrumentator import Instrumentator


def configure_logging(log_level: str) -> None:
    """Configure structlog to emit JSON, merging contextvars (the correlation id)."""
    # TODO: set up structlog processors (merge_contextvars, timestamper, JSONRenderer)
    #       and wire the stdlib logging level.
    raise NotImplementedError


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Attach a correlation id to every request.

    - Read `X-Request-ID` from the request headers, or generate a uuid4 if absent.
    - Bind it via structlog.contextvars so every log line in this request carries it.
    - Echo it back in the `X-Request-ID` response header.
    - Clear the contextvars after the response.
    """

    async def dispatch(self, request, call_next):
        # TODO: implement per the docstring.
        raise NotImplementedError


def setup_observability(app) -> None:
    """Instrument the app for tracing + metrics.

    - FastAPIInstrumentor.instrument_app(app)
    - SQLAlchemyInstrumentor().instrument(engine=...)  (pass the async engine's sync_engine)
    - Instrumentator().instrument(app).expose(app, endpoint="/metrics")
    Only configure an OTLP exporter if settings.otel_exporter_otlp_endpoint is set.
    """
    # TODO: implement per the docstring.
    raise NotImplementedError
