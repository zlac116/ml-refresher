"""Happy-path + validation + auth tests for the trades API."""


async def test_requires_auth(client):
    """GET /api/v1/trades with no token -> 401."""
    # TODO
    r = await client.get("/api/v1/trades")
    assert r.status_code == 401


async def test_open_and_get_trade(client, auth_headers):
    """POST a trade, GET it back: fields round-trip and status == 'open'."""
    # TODO
    r = await client.post(
        "/api/v1/trades",
        headers=auth_headers,
        json={"symbol": "AAPL", "side": "buy", "quantity": 10, "entry_price": 100}
    )
    assert r.status_code == 201, r.text
    
    trade_id = r.json()["id"]
    r = await client.get(f"/api/v1/trades/{trade_id}", headers=auth_headers)
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "open"

async def test_open_trade_validation(client, auth_headers):
    """POST with quantity <= 0 (or an invalid side) -> 422."""
    # TODO
    r = await client.post(
        "/api/v1/trades",
        headers=auth_headers,
        json={"symbol": "AAPL", "side": "buy", "quantity": -1, "entry_price": 100}
    )
    assert r.status_code == 422, r.text

async def test_close_trade_computes_pnl(client, auth_headers):
    """Open a buy, close above entry -> realised_pnl > 0 and status == 'closed'."""
    # TODO
    # Buy 10 AAPL at 100
    r = await client.post(
        "/api/v1/trades",
        headers=auth_headers,
        json={"symbol": "AAPL", "side": "buy", "quantity": 10, "entry_price": 100}
    )
    trade_id = r.json()["id"]
    
    # Close at 110
    r = await client.post(
        f"/api/v1/trades/{trade_id}/close",
        headers=auth_headers,
        json={"exit_price": 110}
    )
    
    body = r.json()
    
    assert body["realised_pnl"] > 0, body
    assert body["status"] == "closed", body