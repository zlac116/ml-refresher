"""Password hashing (Argon2) and JWT creation/verification.

WHAT THIS FILE IS — pure security helpers, no DB and no FastAPI. The auth service
uses hash/verify when registering & logging in; get_current_user uses decode on
every protected request.

  - hash_password / verify_password: turn a password into an Argon2 hash and check it.
  - create_access_token: make a signed JWT (a tamper-proof token) carrying the user id.
  - decode_access_token: verify a token's signature + expiry and read the user id back.

DOCS:
  PyJWT: https://pyjwt.readthedocs.io/en/stable/usage.html
  pwdlib: https://frankie567.github.io/pwdlib/

TEST IN ISOLATION (after implementing):
  uv run python -c "from app.core.security import hash_password, verify_password; \
    h = hash_password('secret123'); print(verify_password('secret123', h), verify_password('wrong', h))"
"""
from datetime import datetime, timedelta, timezone

import jwt  # PyJWT
from pwdlib import PasswordHash

from app.core.config import get_settings
from app.exceptions import AuthError

# pwdlib's recommended hasher is Argon2 (the 2026 default over bcrypt/passlib).
_password_hash = PasswordHash.recommended()


def hash_password(plain: str) -> str:
    """Return an Argon2 hash of the plaintext password."""
    return _password_hash.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    """Return True iff the plaintext matches the stored hash."""
    return _password_hash.verify(plain, hashed)


def create_access_token(subject: str, expires_minutes: int | None = None) -> str:
    """Create a signed JWT whose `sub` claim identifies the user."""

    s = get_settings()
    minutes = expires_minutes or s.access_token_expire_minutes
    now = datetime.now(timezone.utc)
    claims = {"sub": subject, "iat": now, "exp": now + timedelta(minutes=minutes)}
    return jwt.encode(claims, s.secret_key, algorithm=s.jwt_algorithm)



def decode_access_token(token: str) -> str:
    """Verify signature + expiry and return the `sub` claim; raise AuthError otherwise."""

    s = get_settings()
    try:
        payload = jwt.decode(token, s.secret_key, algorithms=[s.jwt_algorithm])
    except jwt.PyJWTError:
        raise AuthError("invalid or expired token")
    return payload["sub"]
    # Never let a raw jwt exception escape — always convert to AuthError.

# if __name__ == "__main__":
  
#     h = hash_password("password1234")
#     print("Password verified: ", verify_password("password124", h))
#     token = create_access_token("user1", 5)
#     print("token: ", decode_access_token(token+"&"))