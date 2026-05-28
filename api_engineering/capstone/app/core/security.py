"""Password hashing (Argon2) and JWT creation/verification.

Pure functions — no DB, no FastAPI imports. Used by the auth service and by the
get_current_user dependency.
"""
import jwt  # PyJWT
from pwdlib import PasswordHash

from app.core.config import get_settings

# pwdlib's recommended hasher is Argon2 (the 2026 default over bcrypt/passlib).
_password_hash = PasswordHash.recommended()


def hash_password(plain: str) -> str:
    """Return an Argon2 hash of the plaintext password."""
    # TODO: _password_hash.hash(plain)
    raise NotImplementedError


def verify_password(plain: str, hashed: str) -> bool:
    """Return True iff the plaintext matches the stored hash."""
    # TODO: _password_hash.verify(plain, hashed)
    raise NotImplementedError


def create_access_token(subject: str, expires_minutes: int | None = None) -> str:
    """Create a signed JWT whose `sub` claim identifies the user.

    Claims to include: sub (subject), iat (issued-at), exp (now + expiry).
    Sign with settings.secret_key and settings.jwt_algorithm.
    """
    # TODO: build the claims dict (use timezone-aware UTC) and jwt.encode(...).
    raise NotImplementedError


def decode_access_token(token: str) -> str:
    """Verify signature + expiry and return the `sub` claim.

    On any invalid/expired token, raise app.exceptions.AuthError — do NOT let
    raw jwt exceptions propagate.
    """
    # TODO: jwt.decode(token, key, algorithms=[...]); catch jwt.PyJWTError -> AuthError.
    raise NotImplementedError
