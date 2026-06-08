"""Shared FastAPI dependencies — the seam between HTTP and the service layer.

WHAT THIS FILE IS — small functions FastAPI runs *before* your route, via `Depends`.
They supply a DB session, build the service objects, and (for protected routes)
turn the Bearer token into the current User. The pieces marked "provided" are done;
you implement `get_current_user`.

DOCS: https://fastapi.tiangolo.com/tutorial/dependencies/  (security: .../security/)
"""
from fastapi import Depends
from fastapi.security import OAuth2PasswordBearer
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import decode_access_token
from app.db.session import get_session
from app.exceptions import AuthError
from app.repositories import TradeRepository, UserRepository
from app.services import AuthService, TradeService

# (provided) Tells /docs where to log in, and extracts the Bearer token from requests.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")

# (provided) IP-keyed rate limiter; wired to the app in main.py (stretch).
limiter = Limiter(key_func=get_remote_address)


# (provided) DI: build a service bound to this request's session.
def get_auth_service(session: AsyncSession = Depends(get_session)) -> AuthService:
    return AuthService(UserRepository(session))


def get_trade_service(session: AsyncSession = Depends(get_session)) -> TradeService:
    return TradeService(TradeRepository(session))


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    session: AsyncSession = Depends(get_session),
):
    """Resolve the authenticated user from the Bearer token, or raise AuthError.

    TODO:
        email = decode_access_token(token)                 # raises AuthError if bad/expired
        user = await UserRepository(session).get_by_email(email)
        if user is None:
            raise AuthError("user no longer exists")
        return user

    Any route with `user = Depends(get_current_user)` is now protected — an
    unauthenticated request never reaches your handler.
    """
    email = decode_access_token(token)
    user = await UserRepository(session).get_by_email(email)
    if user is None:
        raise AuthError("user no longer exists")
    return user
