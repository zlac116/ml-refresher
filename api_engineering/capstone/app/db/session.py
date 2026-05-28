"""Async database engine, session factory, and the declarative Base.

Provides:
  - Base         -- DeclarativeBase that all ORM models subclass.
  - engine       -- AsyncEngine from settings.database_url.
  - SessionLocal -- async_sessionmaker bound to the engine.
  - get_session  -- FastAPI dependency yielding an AsyncSession.
"""
from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


# TODO: build the async engine + sessionmaker from settings.database_url, e.g.
#   from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
#   engine = create_async_engine(get_settings().database_url, pool_pre_ping=True)
#   SessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency: yield an AsyncSession and guarantee it is closed.

    Pattern:  async with SessionLocal() as session: yield session
    """
    # TODO: implement the async-generator dependency.
    raise NotImplementedError
