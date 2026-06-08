"""Domain exceptions and their HTTP mappings.

WHAT THIS FILE IS — your own error types + the one place that turns them into HTTP
responses. Services `raise NotFoundError(...)` (never a raw HTTPException), so the
service layer stays free of HTTP concerns; the handlers registered here translate
each error into the right status code + a clean JSON body.

DOCS: https://fastapi.tiangolo.com/tutorial/handling-errors/#install-custom-exception-handlers
"""
from fastapi import Request
from fastapi.responses import JSONResponse


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
    """Attach a handler per error type so each maps to a JSON HTTP response.

    Status map: NotFoundError->404, AuthError->401, ConflictError->409,
                BusinessRuleError->422.

    TODO — a small factory avoids repetition; register one per (error, status):
    """
    # raise NotImplementedError

    def _make(status_code: int):
        async def handler(request: Request, exc: AppError):
            return JSONResponse(
                status_code=status_code,
                content={"error": {"type": type(exc).__name__, "detail": str(exc)}}
            )
        return handler
    
    app.add_exception_handler(NotFoundError, _make(404))
    app.add_exception_handler(AuthError, _make(401))
    app.add_exception_handler(ConflictError, _make(409))
    app.add_exception_handler(BusinessRuleError, _make(422))
    
    # Never put a traceback or SQL in the response body.
    
# if __name__ == "__main__":
    
#     from fastapi import FastAPI
#     from fastapi.testclient import TestClient
    
#     app = FastAPI()
#     register_exception_handlers(app)
    
#     @app.get("/nf")
#     def _nf(): raise NotFoundError("trade not found")
#     @app.get("/auth")
#     def _au(): raise AuthError("bad credentials")
#     @app.get("/cf")
#     def _cf(): raise ConflictError("email taken")
#     @app.get("/biz")
#     def _bz(): raise BusinessRuleError("already closed")
    
#     c = TestClient(app)
#     for path, expected in [("/nf", 404), ("/auth", 401), ("/cf", 409), ("/biz", 422)]:
#         r = c.get(path)
#         print(f"{path:6s} -> status={r.status_code} body={r.json()}")
#         assert r.status_code == expected, f"expected {expected}, got {r.status_code}"
#         breakpoint()
#         assert r.json()["error"]["type"]
        
#     print("ALL OK")
    