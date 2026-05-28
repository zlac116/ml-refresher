"""Service layer: business logic + authorization. No SQL, no HTTP objects.

This is where ownership (BOLA) checks live and where realised P&L is computed.
Services raise domain exceptions from app.exceptions; they never raise HTTP errors.
"""
from app import schemas
from app.repositories import TradeRepository, UserRepository


class AuthService:
    def __init__(self, users: UserRepository):
        self.users = users

    async def register(self, data: schemas.RegisterRequest) -> "object":
        """Create a user; raise ConflictError if the email already exists.

        Hash the password with core.security.hash_password before storing — never
        persist the plaintext. Return the created user.
        """
        # TODO
        raise NotImplementedError

    async def authenticate(self, email: str, password: str) -> str:
        """Verify credentials and return a signed JWT access token.

        Raise AuthError on failure. Use verify_password (constant-time). Do NOT
        reveal whether the email or the password was the wrong part.
        """
        # TODO
        raise NotImplementedError


class TradeService:
    def __init__(self, trades: TradeRepository):
        self.trades = trades

    async def _get_owned(self, trade_id: int, owner_id: int):
        """Fetch a trade and assert the caller owns it — YOUR CORE BOLA DEFENCE.

        If the trade doesn't exist OR trade.owner_id != owner_id, raise
        NotFoundError (404, NOT 403 — don't leak existence). Route every
        single-trade operation (get/update/close/delete) through this helper.
        """
        # TODO
        raise NotImplementedError

    async def open_trade(self, owner_id: int, data: schemas.TradeCreate):
        """Create an open trade owned by owner_id."""
        # TODO
        raise NotImplementedError

    async def get_trade(self, trade_id: int, owner_id: int):
        """Return an owned trade (via _get_owned)."""
        # TODO
        raise NotImplementedError

    async def list_trades(self, owner_id: int, *, status=None, symbol=None,
                          limit: int = 50, offset: int = 0):
        """Return a page of the caller's trades."""
        # TODO
        raise NotImplementedError

    async def update_trade(self, trade_id: int, owner_id: int, data: schemas.TradeUpdate):
        """Patch an owned, still-open trade. Raise BusinessRuleError if it's closed."""
        # TODO
        raise NotImplementedError

    async def close_trade(self, trade_id: int, owner_id: int, data: schemas.TradeClose):
        """Close an owned trade at data.exit_price; set status/closed_at and
        compute realised P&L.

        Realised P&L convention: buy  -> (exit - entry) * quantity
                                 sell -> (entry - exit) * quantity
        Raise BusinessRuleError if the trade is already closed.
        """
        # TODO
        raise NotImplementedError

    async def delete_trade(self, trade_id: int, owner_id: int) -> None:
        """Delete an owned trade."""
        # TODO
        raise NotImplementedError

    async def portfolio_summary(self, owner_id: int) -> schemas.PortfolioSummary:
        """Aggregate the caller's open exposure and realised P&L."""
        # TODO
        raise NotImplementedError
