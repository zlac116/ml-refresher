"""SQLAlchemy 2.0 ORM models — the database TABLES, typed with Mapped[...].

WHAT THIS FILE IS — the structure of the rows your API will create. `User` is the
`users` table; `Trade` is the `trades` table. They start EMPTY; the API fills them
(register → insert a user; open a trade → insert a trade). `Trade.owner_id` links
a trade to its user and is the column every ownership (BOLA) check depends on.

HOW A COLUMN IS DECLARED:
    name: Mapped[<python type>] = mapped_column(<options>)

DOCS: https://docs.sqlalchemy.org/en/20/orm/declarative_tables.html

TEST IN ISOLATION (after you've filled both classes):
  uv run python -c "from app.models import User, Trade; \
    print('users:', list(User.__table__.columns.keys())); \
    print('trades:', list(Trade.__table__.columns.keys()))"
"""
from datetime import datetime

from sqlalchemy import ForeignKey, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


class User(Base):
    __tablename__ = "users"
    # TODO — add one `Mapped[...] = mapped_column(...)` line per column:
    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(unique=True)
    password_hash: Mapped[str] = mapped_column()
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

class Trade(Base):
    __tablename__ = "trades"
    # TODO — add one line per column:
    id: Mapped[int] = mapped_column(primary_key=True)
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    symbol: Mapped[str] = mapped_column()
    side: Mapped[str] = mapped_column() # "buy" | "sell"
    quantity: Mapped[float] = mapped_column() # >0 (enforced in schema)
    entry_price: Mapped[float] = mapped_column() # >0 (enforced in schema)
    exit_price: Mapped[float | None] = mapped_column(default=None) # null until closed
    status: Mapped[str] = mapped_column(default="open") # "open" | "closed"
    opened_at: Mapped[datetime] = mapped_column(server_default=func.now())
    closed_at: Mapped[datetime | None] = mapped_column(default=None)
    realised_pnl: Mapped[float | None] = mapped_column(default=None)

# if __name__ == "__main__":
#     print('users: ', list(User.__table__.columns.keys()))
#     print('trades: ', list(Trade.__table__.columns.keys()))