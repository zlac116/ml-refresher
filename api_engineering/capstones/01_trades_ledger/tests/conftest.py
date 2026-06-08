"""Pytest fixtures: an async HTTP client wired to a fresh, isolated test database.

Use SQLite (aiosqlite) so tests are fast and never touch your real Postgres.
Override the get_session dependency to hand routes a test session.
"""
import pytest
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.db.session import Base, get_session
from app import models

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
    # Build in-memory engine per-test
    test_engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False}
    )
    Testsession = async_sessionmaker(test_engine, expire_on_commit=False)
    
    # Create empty tables in the test DB (like lifespan in main.py)
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    # Override the real get_session so that every toute talks to test DB
    async def _override_get_session():
        async with Testsession() as s:
            yield s
    app.dependency_overrides[get_session] = _override_get_session
    
    # In-process HTTP client - no real server
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    
    # Teardown: undo override, dispose the engine
    app.dependency_overrides.clear()
    await test_engine.dispose()
    

@pytest.fixture
async def auth_headers(client):
    """Register + login a test user; return {'Authorization': 'Bearer <token>'}."""
    # TODO: POST /api/v1/auth/register then /login; build the header dict.
    body = {"email": "test@example.com", "password": "supersecret123"}
    r = await client.post("/api/v1/auth/register", json=body)
    assert r.status_code == 201, r.text
    
    r = await client.post(
        "/api/v1/auth/login",
        data={"username": body["email"], "password": body["password"]},
    )
    assert r.status_code == 200, r.text
    token = r.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}

