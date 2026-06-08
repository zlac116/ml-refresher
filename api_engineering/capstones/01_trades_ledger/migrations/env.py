"""Alembic migration environment (STRETCH GOAL).

The simplest way to start this file correctly is to scaffold the async template
and adapt it:

    uv run alembic init -t async migrations   # generates a working async env.py

Then wire it to this project:
  - import Base from app.db.session AND import app.models so Base.metadata is
    fully populated, then set: target_metadata = Base.metadata
  - read the DB URL from app settings (get_settings().database_url), not alembic.ini
  - keep the async run_migrations_online() that uses create_async_engine +
    connection.run_sync(do_run_migrations)

For the 4-hour CORE you can SKIP Alembic entirely and create tables in the app
lifespan (dev only). Implement this file only if you tackle the stretch goal.
"""
raise NotImplementedError("Stretch goal: scaffold with `alembic init -t async` and wire to app settings.")
