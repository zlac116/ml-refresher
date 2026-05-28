"""Happy-path + validation + auth tests for the trades API."""


async def test_requires_auth(client):
    """GET /api/v1/trades with no token -> 401."""
    # TODO
    raise NotImplementedError


async def test_open_and_get_trade(client, auth_headers):
    """POST a trade, GET it back: fields round-trip and status == 'open'."""
    # TODO
    raise NotImplementedError


async def test_open_trade_validation(client, auth_headers):
    """POST with quantity <= 0 (or an invalid side) -> 422."""
    # TODO
    raise NotImplementedError


async def test_close_trade_computes_pnl(client, auth_headers):
    """Open a buy, close above entry -> realised_pnl > 0 and status == 'closed'."""
    # TODO
    raise NotImplementedError
