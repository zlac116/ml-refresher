# STRETCH GOAL — multi-stage build. Fill in the TODOs.

# Stage 1: build the dependency venv with uv.
FROM python:3.12-slim AS builder
# TODO:
#   - install uv (COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv)
#   - WORKDIR /app ; COPY pyproject.toml uv.lock ./
#   - uv sync --frozen --no-dev   (creates /app/.venv)

# Stage 2: slim runtime as a non-root user.
FROM python:3.12-slim
# TODO:
#   - create a non-root user and USER it
#   - COPY --from=builder /app/.venv /app/.venv ; COPY ./app /app/app
#   - ENV PATH="/app/.venv/bin:$PATH" ; WORKDIR /app ; EXPOSE 8000
#   - HEALTHCHECK CMD curl -f http://localhost:8000/health || exit 1
#   - CMD ["gunicorn", "app.main:app", "-k", "uvicorn.workers.UvicornWorker",
#          "-w", "4", "-b", "0.0.0.0:8000"]
