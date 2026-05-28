"""Repository layer: ALL database queries live here.

No business logic, no HTTP, no authorization decisions — just data access.
Repositories take an AsyncSession and return ORM models. The service layer
calls these and owns the business/authorization rules.
"""
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Trade, User


class UserRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_email(self, email: str) -> User | None:
        """Return the user with this email, or None."""
        # TODO: await self.session.scalar(select(User).where(User.email == email))
        raise NotImplementedError

    async def create(self, email: str, password_hash: str) -> User:
        """Insert and return a new user."""
        # TODO: build User, session.add, flush, refresh, return.
        raise NotImplementedError


class TradeRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get(self, trade_id: int) -> Trade | None:
        """Return the trade by id, or None. (Ownership is enforced in the service.)"""
        # TODO
        raise NotImplementedError

    async def list_for_owner(
        self,
        owner_id: int,
        *,
        status: str | None = None,
        symbol: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Trade]:
        """Return a page of trades belonging to owner_id, with optional filters.

        Always filter by owner_id at the query level — never fetch-all-then-filter.
        """
        # TODO: build select(Trade).where(Trade.owner_id == owner_id) (+ filters),
        #       apply .limit(limit).offset(offset), return list.
        raise NotImplementedError

    async def add(self, trade: Trade) -> Trade:
        """Persist a new trade and return it (refreshed)."""
        # TODO
        raise NotImplementedError

    async def save(self, trade: Trade) -> Trade:
        """Flush updates to an existing (already-tracked) trade."""
        # TODO
        raise NotImplementedError

    async def delete(self, trade: Trade) -> None:
        """Remove a trade."""
        # TODO
        raise NotImplementedError
