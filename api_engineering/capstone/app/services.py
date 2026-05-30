"""Service layer: business logic + authorization. No SQL, no HTTP objects.

WHAT THIS FILE IS — the brain. It calls repositories for data, calls core.security
for hashing/JWTs, applies the rules (ownership/BOLA, P&L, "can't close twice"),
and raises domain errors from app.exceptions. It never touches the DB directly and
never raises HTTPException — the route + the exception handlers deal with HTTP.

This is the part that is genuinely YOUR exercise — the hints below give the steps
and which helpers to call, but you write and understand the logic (especially the
`_get_owned` ownership check, which is the whole security point of the capstone).
"""
from datetime import datetime, timezone

from app import schemas
from app.core import security
from app.exceptions import AuthError, BusinessRuleError, ConflictError, NotFoundError
from app.models import Trade
from app.repositories import TradeRepository, UserRepository


class AuthService:
    def __init__(self, users: UserRepository):
        self.users = users

    async def register(self, data: schemas.RegisterRequest):
        """Create a user; ConflictError if the email already exists. Returns the User.

        Steps:
          1. if await self.users.get_by_email(data.email): raise ConflictError(...)
          2. hash the password: security.hash_password(data.password)
          3. return await self.users.create(data.email, <hash>)   # never store plaintext
        """
        if await self.users.get_by_email(data.email):
            raise ConflictError(f"email already exists: {data.email}")
        pw_h = security.hash_password(data.password)
        return await self.users.create(data.email, pw_h)

    async def authenticate(self, email: str, password: str) -> str:
        """Verify credentials, return a signed JWT. AuthError on failure.

        Steps:
          1. user = await self.users.get_by_email(email)
          2. if user is None OR not security.verify_password(password, user.password_hash):
                 raise AuthError(...)        # same error either way — don't reveal which
          3. return security.create_access_token(subject=user.email)
             (use email as the token subject — get_current_user loads the user back
              via UserRepository.get_by_email, so the subject must be the email)
        """
        user = await self.users.get_by_email(email)
        if user is None or not security.verify_password(password, user.password_hash):
            raise AuthError("authentication error")
        return security.create_access_token(subject=user.email)


class TradeService:
    def __init__(self, trades: TradeRepository):
        self.trades = trades

    async def _get_owned(self, trade_id: int, owner_id: int) -> Trade:
        """Fetch a trade and assert the caller owns it — YOUR CORE BOLA DEFENCE.

        trade = await self.trades.get(trade_id)
        if trade is None or trade.owner_id != owner_id:
            raise NotFoundError("trade not found")   # 404 even if it exists but isn't yours
        return trade

        Route every single-trade op (get/update/close/delete) through this.
        """
        trade = await self.trades.get(trade_id)
        if trade is None or trade.owner_id != owner_id:
            raise NotFoundError("trade not found")
        return trade

    async def open_trade(self, owner_id: int, data: schemas.TradeCreate) -> Trade:
        """Build a Trade from `data` with owner_id + status 'open', then persist it.

        Steps: construct Trade(owner_id=owner_id, symbol=data.symbol, side=data.side.value,
        quantity=data.quantity, entry_price=data.entry_price, status='open');
        return await self.trades.add(trade).
        """
        trade = Trade(
            owner_id=owner_id, 
            symbol=data.symbol, 
            side=data.side.value, 
            quantity=data.quantity,
            entry_price=data.entry_price,
            status="open"
        )
        return await self.trades.add(trade)

    async def get_trade(self, trade_id: int, owner_id: int) -> Trade:
        """return await self._get_owned(trade_id, owner_id)"""
        return await self._get_owned(trade_id, owner_id)

    async def list_trades(self, owner_id: int, *, status=None, symbol=None,
                          limit: int = 50, offset: int = 0) -> list[Trade]:
        """return await self.trades.list_for_owner(owner_id, status=..., symbol=..., limit=..., offset=...)"""
        return await self.trades.list_for_owner(
            owner_id=owner_id,
            status=status,
            symbol=symbol,
            limit=limit,
            offset=offset,
        )

    async def update_trade(self, trade_id: int, owner_id: int, data: schemas.TradeUpdate) -> Trade:
        """Patch an owned, still-open trade. (Stretch.)

        Steps: trade = await self._get_owned(...); if trade.status != 'open':
        raise BusinessRuleError; apply the non-None fields from data; return await self.trades.save(trade).
        """
        trade = await self._get_owned(trade_id, owner_id)
        if trade.status != "open":
            raise BusinessRuleError("Trade is not open")
        
        patch = data.model_dump(exclude_unset=True)
        for k, v in patch.items():
            setattr(trade, k, v)
            
        return await self.trades.save(trade)

    async def close_trade(self, trade_id: int, owner_id: int, data: schemas.TradeClose) -> Trade:
        """Close an owned trade at data.exit_price; compute realised P&L. (Stretch.)

        Steps:
          1. trade = await self._get_owned(trade_id, owner_id)
          2. if trade.status == 'closed': raise BusinessRuleError("already closed")
          3. set trade.exit_price = data.exit_price; trade.status = 'closed';
             trade.closed_at = datetime.now(timezone.utc)
          4. P&L:  buy  -> (exit - entry) * quantity ;  sell -> (entry - exit) * quantity
             (store it if you added a column, or compute it for the response)
          5. return await self.trades.save(trade)
        """
        trade = await self._get_owned(trade_id, owner_id)
        if trade.status == "closed":
            raise BusinessRuleError("already closed")
        trade.exit_price = data.exit_price
        trade.status = "closed"
        trade.closed_at = datetime.now(timezone.utc)
        if trade.side == "buy":
            realised_pnl = (trade.exit_price - trade.entry_price) * trade.quantity
        else:
            realised_pnl = (trade.entry_price - trade.exit_price) * trade.quantity
        
        trade.realised_pnl = realised_pnl
        
        return await self.trades.save(trade)
        

    async def delete_trade(self, trade_id: int, owner_id: int) -> None:
        """trade = await self._get_owned(...); await self.trades.delete(trade)   # (stretch)"""
        trade = await self._get_owned(trade_id, owner_id)
        await self.trades.delete(trade)

    async def portfolio_summary(self, owner_id: int) -> schemas.PortfolioSummary:
        """Aggregate the caller's open exposure + realised P&L. (Stretch.)"""
        trades = await self.list_trades(owner_id, limit =10_000)
        open_positions = sum(1 for t in trades if t.status == "open")
        gross_exposure = sum(t.quantity for t in trades if t.status == "open")
        realised_pnl = sum((t.realised_pnl or 0.0) for t in trades if t.status == "closed")
        return schemas.PortfolioSummary(
            open_positions=open_positions,
            gross_exposure=gross_exposure,
            realised_pnl=realised_pnl
        )
