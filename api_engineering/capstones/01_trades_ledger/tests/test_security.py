"""Security tests. The headline one proves BOLA (OWASP API1) is closed."""

async def register_and_login(client, email: str, password: str) -> dict:
    """Return the Bearer header for a freshly registered user."""
    body = {"email": email, "password": password}
    r = await client.post("api/v1/auth/register", json=body)
    
    assert r.status_code == 201, r.text
      
    r = await client.post(
        "/api/v1/auth/login",
        data={"username": email, "password": password},
    )   
    assert r.status_code == 200, r.text
    token = r.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


async def test_bola_cannot_read_others_trade(client):
    """User A's trade must be invisible to user B.

    Steps:
      - register + login user A; POST a trade; capture its id
      - register + login user B (different credentials)
      - GET /api/v1/trades/{id} as user B
      - assert the response is 404 (NOT 200, and NOT 403 — do not leak existence)
    """
    # TODO
    headers_a = await register_and_login(client, "a@x.io", "supersecret123")
    headers_b = await register_and_login(client, "b@x.io", "supersecret123")

    # A opens a trade
    r = await client.post(
        "/api/v1/trades",
        headers=headers_a,
        json={"symbol": "AAPL", "side": "buy", "quantity": 10, "entry_price": 100})
    assert r.status_code == 201, r.text
    trade_id = r.json()["id"]

    # B tries to read it — THE KEY ASSERTION
    r = await client.get(f"/api/v1/trades/{trade_id}", headers=headers_b)
    assert r.status_code == 404, r.text                    # ← the proof

    # (positive control — proves the trade really exists)
    r = await client.get(f"/api/v1/trades/{trade_id}", headers=headers_a)
    assert r.status_code == 200, r.text
    

async def test_invalid_token_rejected(client):
    """A malformed/expired Bearer token -> 401."""
    # TODO
    r = await client.post("/api/v1/trades", headers={"Authorization": "Bearer malformed"})
    
    assert r.status_code == 401, r.text
    