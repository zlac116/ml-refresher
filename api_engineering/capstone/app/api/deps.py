"""Shared FastAPI dependencies — the seam between HTTP and the service layer.

Provides the OAuth2 scheme, the rate limiter, service providers (DI), and the
get_current_user guard that protects authenticated routes.
"""
from fastapi import Depends
from fastapi.security import OAuth2PasswordBearer
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_session
from app.repositories import TradeRepository, UserRepository
from app.services import AuthService, TradeService

# Points /docs' "Authorize" button + extracts the Bearer token.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")

# IP-keyed limiter; attach to the app in main.py and apply limits per route.
limiter = Limiter(key_func=get_remote_address)


def get_auth_service(session: AsyncSession = Depends(get_session)) -> AuthService:
    """Build an AuthService wired to this request's session (DI pattern)."""
    return AuthService(UserRepository(session))


def get_trade_service(session: AsyncSession = Depends(get_session)) -> TradeService:
    """Build a TradeService wired to this request's session (DI pattern)."""
    return TradeService(TradeRepository(session))


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    session: AsyncSession = Depends(get_session),
):
    """Resolve the authenticated user from the Bearer token, or raise AuthError.

    Steps: decode_access_token(token) -> subject; load the user via
    UserRepository; raise AuthError if the token is invalid/expired or the user
    no longer exists. Return the ORM User.
    """
    # TODO: implement per the docstring.
    raise NotImplementedError
