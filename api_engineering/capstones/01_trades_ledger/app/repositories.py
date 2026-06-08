"""Repository layer: ALL database queries live here.

WHAT THIS FILE IS — the only place that talks SQL/ORM. No business logic, no HTTP,
no authorization. A repository takes an AsyncSession and returns ORM objects; the
service layer calls these and owns the rules.

KEY ASYNC PATTERNS (SQLAlchemy 2.0):
  read one : await session.scalar(select(Model).where(...))      # -> Model | None
  read many: (await session.scalars(stmt)).all()                 # -> list[Model]
  get by pk: await session.get(Model, pk)                        # -> Model | None
  write    : session.add(obj); await session.commit(); await session.refresh(obj)
  delete   : await session.delete(obj); await session.commit()

NOTE ON COMMIT: the get_session dependency does NOT auto-commit, so writes must
commit themselves (done in the write methods below). refresh() reloads server
defaults like the auto id / created_at.

DOCS: https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html#synopsis-orm

TEST IN ISOLATION: easier once models + a DB exist — see the tests in tests/.
"""
from sqlalchemy import select

from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Trade, User


class UserRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_email(self, email: str) -> User | None:
        """Return the user with this email, or None."""
    
        return await self.session.scalar(select(User).where(User.email == email))
    

    async def create(self, email: str, password_hash: str) -> User:
        """Insert and return a new user."""
        user = User(email=email, password_hash=password_hash)
        self.session.add(user)
        await self.session.commit()
        await self.session.refresh(user) # populates id + created_at
        return user


class TradeRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get(self, trade_id: int) -> Trade | None:
        """Return the trade by id, or None. (Ownership is checked in the service.)"""
        return await self.session.get(Trade, trade_id)

    async def list_for_owner(
        self,
        owner_id: int,
        *,
        status: str | None = None,
        symbol: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Trade]:
        """A page of trades belonging to owner_id (always filter at the query level)."""
        stmt = select(Trade).where(Trade.owner_id == owner_id)
        if status: stmt = stmt.where(Trade.status == status)
        if symbol: stmt = stmt.where(Trade.symbol == symbol)
        stmt = stmt.limit(limit).offset(offset)
        return list(await self.session.scalars(stmt))

    async def add(self, trade: Trade) -> Trade:
        """Persist a new trade and return it (refreshed)."""
        self.session.add(trade)
        await self.session.commit()
        await self.session.refresh(trade)
        return trade
        

    async def save(self, trade: Trade) -> Trade:
        """Commit updates to an already-tracked trade and return it."""
        await self.session.commit()
        await self.session.refresh(trade)
        return trade

    async def delete(self, trade: Trade) -> None:
        """Remove a trade."""
        await self.session.delete(trade)
        await self.session.commit()
