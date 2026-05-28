"""SQLAlchemy 2.0 ORM models (typed with Mapped[...], async-compatible).

Two tables: users and trades. One user owns many trades; trades.owner_id is the
column every ownership (BOLA) check hinges on.
"""
from app.db.session import Base


class User(Base):
    __tablename__ = "users"
    # TODO: typed columns via mapped_column:
    #   id: Mapped[int]            -- primary key
    #   email: Mapped[str]         -- unique, indexed
    #   password_hash: Mapped[str]
    #   created_at: Mapped[datetime]  -- server_default=func.now()
    # Optional: trades relationship (relationship(back_populates=...)).


class Trade(Base):
    __tablename__ = "trades"
    # TODO: typed columns via mapped_column:
    #   id: Mapped[int]                          -- primary key
    #   owner_id: Mapped[int]                    -- ForeignKey("users.id"), indexed, NOT NULL
    #   symbol: Mapped[str]
    #   side: Mapped[str]                        -- "buy" | "sell" (or SQLAlchemy Enum)
    #   quantity: Mapped[float]                  -- > 0 (enforce in schema)
    #   entry_price: Mapped[float]               -- > 0
    #   exit_price: Mapped[float | None]         -- nullable until closed
    #   status: Mapped[str]                      -- "open" | "closed", default "open"
    #   opened_at: Mapped[datetime]              -- server_default=func.now()
    #   closed_at: Mapped[datetime | None]       -- nullable
