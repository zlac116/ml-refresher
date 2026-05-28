"""Pytest fixtures: an async HTTP client wired to a fresh, isolated test database.

Use SQLite (aiosqlite) so tests are fast and never touch your real Postgres.
Override the get_session dependency to hand routes a test session.
"""
import pytest


@pytest.fixture
async def client():
    """Yield an httpx.AsyncClient bound to the app via ASGITransport.

    Setup:
      - build an async sqlite engine (sqlite+aiosqlite:///:memory: or a temp file)
      - Base.metadata.create_all on it
      - app.dependency_overrides[get_session] = <test session factory>
      - yield AsyncClient(transport=ASGITransport(app=app), base_url="http://test")
    Teardown: drop tables / clear dependency_overrides.
    """
    # TODO: implement per the docstring.
    raise NotImplementedError


@pytest.fixture
async def auth_headers(client):
    """Register + login a test user; return {'Authorization': 'Bearer <token>'}."""
    # TODO: POST /api/v1/auth/register then /login; build the header dict.
    raise NotImplementedError
