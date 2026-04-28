# api — Building & Deploying a REST API for ML Inference

`rest_api.ipynb` is a 13-stage pipeline-stage walkthrough that takes a freshly
trained model and ships it as a production-ready FastAPI service.

## What you build

By the end of the notebook you have:

- A LightGBM model trained on BTC hourly data (sign of next 4h return), bundled
  with `feature_names`, `threshold`, and `trained_through`.
- A FastAPI service exposing:
  - `POST /predict` — single-row prediction
  - `POST /predict/batch` — vectorised, capped at 1000 rows
  - `GET /data/recent` — paginated bar data with query-param validation
  - `GET /health` — liveness probe (unauthenticated)
  - `GET /ready` — readiness probe (503 until model warm)
  - `GET /metrics` — Prometheus-format counters
  - `GET /info` — model version + feature names
- Pydantic schemas for every request and response.
- API-key authentication (header-based, constant-time comparison).
- Structured JSON logging with per-request correlation IDs.
- A multi-stage `Dockerfile` (non-root user, healthcheck) and `docker-compose.yml`.
- A pytest integration suite covering happy paths, validation, auth, and
  property-based order-invariance.

## Running it

The notebook runs every example **inline** via FastAPI's `TestClient` — you don't
need to launch a real server to follow along. To deploy as a real service, copy
the relevant cells into `app/main.py`, drop in the Dockerfile + compose from
Stage 11, and `docker-compose up`.

```bash
make lab        # opens Jupyter Lab; navigate to api/rest_api.ipynb
```

## Stages

1. **Stage 0** — Problem framing for an inference API
2. **Stage 1** — Train + persist a model with metadata
3. **Stage 2** — A minimal FastAPI app
4. **Stage 3** — Validated I/O with Pydantic
5. **Stage 4** — Error handling and HTTP status codes
6. **Stage 5** — API key authentication
7. **Stage 6** — Batch prediction endpoint
8. **Stage 7** — Data endpoints with pagination
9. **Stage 8** — Ops endpoints: liveness, readiness, metrics
10. **Stage 9** — Structured logging with correlation IDs
11. **Stage 10** — Async vs sync — when to use which
12. **Stage 11** — Containerise with Docker
13. **Stage 12** — Testing the API
