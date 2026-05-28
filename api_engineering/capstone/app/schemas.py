"""Pydantic v2 schemas for request/response validation.

Rules to follow:
  - Request models validate input with constraints (quantity > 0, price > 0,
    side is a valid enum). Bad input is rejected at the boundary -> 422.
  - Response models NEVER expose password_hash or any other user's data
    (OWASP API3). Use model_config = ConfigDict(from_attributes=True) to read
    from ORM objects.
  - Use distinct Create / Update / Read models; do not reuse the ORM model.
"""
from enum import Enum

from pydantic import BaseModel  # also: ConfigDict, EmailStr, Field, field_validator


class Side(str, Enum):
    buy = "buy"
    sell = "sell"


class TradeStatus(str, Enum):
    open = "open"
    closed = "closed"


# ----- auth -----
class RegisterRequest(BaseModel):
    # TODO: email: EmailStr ; password: str = Field(min_length=8)
    pass


class Token(BaseModel):
    # TODO: access_token: str ; token_type: str = "bearer"
    pass


class UserRead(BaseModel):
    # TODO: id, email, created_at. NO password_hash. ConfigDict(from_attributes=True).
    pass


# ----- trades -----
class TradeCreate(BaseModel):
    # TODO: symbol: str ; side: Side ; quantity: float = Field(gt=0) ;
    #       entry_price: float = Field(gt=0)
    pass


class TradeUpdate(BaseModel):
    # TODO: optional fields patchable on an OPEN trade (all Optional / default None).
    pass


class TradeClose(BaseModel):
    # TODO: exit_price: float = Field(gt=0)
    pass


class TradeRead(BaseModel):
    # TODO: id, symbol, side, quantity, entry_price, exit_price, status,
    #       opened_at, closed_at, realised_pnl. ConfigDict(from_attributes=True).
    pass


class PortfolioSummary(BaseModel):
    # TODO: open_positions: int ; gross_exposure: float ; realised_pnl: float ; ...
    pass
