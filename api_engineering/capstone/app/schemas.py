"""Pydantic v2 schemas — validate JSON IN (requests) and shape JSON OUT (responses).

WHAT THIS FILE IS — the contract at the HTTP boundary. A *request* model auto-rejects
bad input with a 422 (e.g. negative quantity). A *response* model controls exactly
which fields go OUT — never expose password_hash (OWASP API3).

FIELD PATTERNS:
    field: type                       # required
    field: type = Field(gt=0)         # required + constraint
    field: type | None = None         # optional
    # response models read straight off ORM objects with:
    model_config = ConfigDict(from_attributes=True)

DOCS: https://docs.pydantic.dev/latest/concepts/models/   (Field: .../fields/)

TEST IN ISOLATION (after filling a model):
  uv run python -c "from app.schemas import TradeCreate; \
    print(TradeCreate(symbol='AAPL', side='buy', quantity=10, entry_price=190.5))"
"""
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class Side(str, Enum):
    buy = "buy"
    sell = "sell"


class TradeStatus(str, Enum):
    open = "open"
    closed = "closed"


# ----- auth -----
class RegisterRequest(BaseModel):
    # TODO:
    email: EmailStr
    password: str = Field(min_length=8)


class Token(BaseModel):
    # TODO:
    access_token: str
    token_type: str = "bearer"


class UserRead(BaseModel):
    # TODO:
    model_config = ConfigDict(from_attributes=True)
    id: int
    email: EmailStr
    created_at: datetime


# ----- trades -----
class TradeCreate(BaseModel):
    # TODO:
    symbol: str
    side: Side
    quantity: float = Field(gt=0)
    entry_price: float = Field(gt=0)

class TradeUpdate(BaseModel):
    # TODO — all optional (only the provided fields get patched on an OPEN trade):
    symbol: str | None = None
    quantity: float | None = Field(default=None, gt=0)
    entry_price: float | None = Field(default=None, gt=0)


class TradeClose(BaseModel):
    # TODO:
    exit_price: float = Field(gt=0)


class TradeRead(BaseModel):
    # TODO:
    model_config = ConfigDict(from_attributes=True)
    id: int ; symbol: str ; side: Side ; quantity: float ; entry_price: float
    exit_price: float | None ; status: TradeStatus
    opened_at: datetime ; closed_at: datetime | None
    realised_pnl: float | None = None     # computed on close; None while open


class PortfolioSummary(BaseModel):
    # TODO (stretch — only needed for GET /portfolio/summary):
    open_positions: int
    gross_exposure: float
    realised_pnl: float

# Test
# if __name__ == "__main__":
    
#     print(TradeClose(exit_price=0.89))
#     print(TradeCreate(symbol="MON", side="buy", quantity=20000, entry_price=0.0234))
