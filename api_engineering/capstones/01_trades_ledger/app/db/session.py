"""Async database engine, session factory, and the declarative Base.

WHAT THIS FILE IS — the shared plumbing every other layer uses to talk to the
database. You build it ONCE here; routes/repositories then get a ready-to-use
session by depending on get_session. Nothing here is business logic.

WHAT YOU'RE PROVIDING:
  - Base         -- the class every ORM model (models.py) will subclass.
  - engine       -- the connection pool to the DB, built from settings.database_url.
  - SessionLocal -- a factory that produces AsyncSession objects.
  - get_session  -- a FastAPI dependency that yields one AsyncSession per request.

DOCS (canonical pattern — read the "Synopsis - ORM" section):
  https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html
"""
from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.core.config import get_settings


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""

# TODO 1 — the engine (one per app): a pool of connections to the DB.
#   engine = create_async_engine(get_settings().database_url, pool_pre_ping=True)
# engine = ...  # <-- replace this
engine = create_async_engine(get_settings().database_url, pool_pre_ping=True)

# TODO 2 — the session factory, bound to the engine. expire_on_commit=False keeps
#          loaded objects usable after commit (matters with async).
#   SessionLocal = async_sessionmaker(engine, expire_on_commit=False)
# SessionLocal = ...  # <-- replace this
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency: open a session, hand it to the caller, then close it.

    TODO 3 — implement exactly this body:
        async with SessionLocal() as session:
            yield session
    The `async with` guarantees the session is closed (and rolled back on error)
    when the request finishes, so routes/repositories never manage that.
    """
    # raise NotImplementedError

    async with SessionLocal() as session:
        yield session


# Test
import asyncio
from sqlalchemy import text

async def main():
    async for s in get_session():
        breakpoint()
        print('get_session OK ->', (await s.execute(text('SELECT 1'))).scalar())
        await engine.dispose()

if __name__ == "__main__":
    asyncio.run(main())
