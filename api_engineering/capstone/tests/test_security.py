"""Security tests. The headline one proves BOLA (OWASP API1) is closed."""


async def test_bola_cannot_read_others_trade(client):
    """User A's trade must be invisible to user B.

    Steps:
      - register + login user A; POST a trade; capture its id
      - register + login user B (different credentials)
      - GET /api/v1/trades/{id} as user B
      - assert the response is 404 (NOT 200, and NOT 403 — do not leak existence)
    """
    # TODO
    raise NotImplementedError


async def test_invalid_token_rejected(client):
    """A malformed/expired Bearer token -> 401."""
    # TODO
    raise NotImplementedError
