"""Domain exceptions and their HTTP mappings.

Business code raises these (never raw HTTPException), keeping HTTP concerns out
of the service layer. main.py registers handlers that translate them to clean
JSON responses with the right status codes.
"""


class AppError(Exception):
    """Base class for all domain errors."""


class NotFoundError(AppError):
    """Resource does not exist OR is not owned by the caller -> 404.

    BOLA safety: 'exists but owned by someone else' MUST map to 404, never 403,
    so you don't leak the existence of other users' resources.
    """


class AuthError(AppError):
    """Authentication failed: bad credentials or invalid/expired token -> 401."""


class ConflictError(AppError):
    """Resource conflict, e.g. registering an email that already exists -> 409."""


class BusinessRuleError(AppError):
    """A business-rule violation not expressible in Pydantic -> 422.

    e.g. closing an already-closed trade, exit price on a non-existent position.
    """


def register_exception_handlers(app) -> None:
    """Map each AppError subclass to a JSON HTTP response via app.add_exception_handler.

    Return a consistent envelope, e.g. {"error": {"type": ..., "detail": ...}}.
    Never include tracebacks or SQL in the response body.
    """
    # TODO: register one handler per error type (NotFoundError->404, AuthError->401,
    #       ConflictError->409, BusinessRuleError->422). Optionally a catch-all 500.
    raise NotImplementedError
