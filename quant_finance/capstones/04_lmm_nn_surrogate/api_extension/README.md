# LMM NN Surrogate — MLflow Registry + REST API (extension)

A **3-hour** extension to the LMM NN surrogate capstone. Wraps the trained
surrogate in a production-shaped workflow: train + register via MLflow, serve
inference + calibration via FastAPI, manage versions through a small REST API.

```
┌──────────────────────────────────────────────────────────────┐
│ CLI:  train_and_register.py                                  │
│   ├─ generate_data() ────────┐                               │
│   ├─ Surrogate + train       │                               │
│   ├─ mlflow.log_params/      │ (reused from parent capstone) │
│   │     log_metrics          │                               │
│   └─ mlflow.pytorch.log_model(registered_model_name=...)     │
│        + client.set_registered_model_alias(... "candidate")  │
└────────────────────┬─────────────────────────────────────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │   MLflow registry          │
        │   lmm-surrogate            │
        │     v1   @production       │      ← what the API loads
        │     v2   (no alias)        │
        │     v3   @candidate        │      ← waiting for promotion
        └──────────┬─────────────────┘
                   │  load("models:/lmm-surrogate@production")
                   ▼
        ┌────────────────────────────┐
        │   FastAPI service          │
        │   POST /calibrate          │   ← broker quotes → θ*
        │   POST /price              │   ← θ + instrument → IV
        │   GET  /models             │   ← list versions + aliases
        │   POST /models/.../promote │   ← move version to alias
        └────────────────────────────┘
```

---

## Mental model

### Why MLflow

In the parent capstone you saved artifacts to a timestamped directory
(`model_outputs/20260607T143212/`). That works for one developer on one
machine. The moment you have:

- multiple training runs to compare ("which `--hidden 64 64` checkpoint
  was best?"),
- a deployed service that needs to load *the current best* model,
- an auditor asking *"what model priced this trade on 7 June?"*,

the timestamped directory pattern breaks. MLflow is the standard answer:

| Concept | What it is |
|---|---|
| **Run** | One execution of training. Logs params, metrics, artifacts. Cheap, many per day. |
| **Registered model** | A named entry (here: `lmm-surrogate`) with a version history. |
| **Version** | Each `mlflow.pytorch.log_model(..., registered_model_name=...)` creates a new version. |
| **Alias** (modern API since MLflow 2.9) | A movable pointer like `@production` or `@candidate`. The API loads `models:/lmm-surrogate@production` — that resolves to whichever version currently carries the `production` alias. Promotion = re-aliasing. |

The old `Stage` API (`Staging`/`Production`/`Archived`) is **deprecated** —
use aliases everywhere.

### Why split CLI training from API inference

Real prod systems separate **offline training** from **online inference**:

- training runs on a dedicated GPU box overnight (slow, batchable)
- inference runs on a stateless web service (fast, autoscaling)

Mixing them in one FastAPI process is wrong for three reasons:
1. Training blocks request handlers for minutes — health checks fail.
2. Training memory footprint dominates — autoscaling becomes pointless.
3. Failure modes mix — a bad training run can crash a running inference replica.

This extension mirrors that split: `train_and_register.py` is the offline
job; `app/` is the online service.

### Why bounds validation matters

The surrogate is **only valid inside its training region** (the
`LMM_PARAM_LO/HI` and `T_LO/HI` etc. ranges). A REST caller can send any
JSON they like. Without validation, the NN happily extrapolates and returns
garbage IVs.

The API enforces the training region at the schema boundary (Pydantic
validators on `Params` and `Instrument`). Out-of-bounds inputs → `422
Unprocessable Entity`. This is non-negotiable in real risk systems.

---

## Repo layout

```
api_extension/
├── README.md                 ← this file (build guide)
├── pyproject.toml            ← uv project: mlflow + fastapi + torch (CPU)
├── train_and_register.py     ← CLI: train → log → register (~50 lines)
├── app/
│   ├── __init__.py
│   ├── main.py               ← FastAPI app + lifespan that loads from registry
│   ├── config.py             ← Pydantic Settings (registry URI, model name)
│   ├── schemas.py            ← request/response models + bounds validators
│   ├── registry.py           ← MLflow client wrappers
│   ├── services.py           ← calibrate + price using app.state.model
│   └── routes/
│       ├── __init__.py
│       ├── calibrate.py      ← POST /calibrate
│       ├── price.py          ← POST /price
│       └── models.py         ← GET /models, POST /models/{name}/promote
└── tests/
    └── test_e2e.py           ← one integration test module
```

A short `sys.path` shim at the top of `train_and_register.py` lets it
`from surrogate import generate_data, Surrogate, ...` — the parent
capstone stays untouched.

---

## Each file's job

### `pyproject.toml`
uv project. Pin torch-cpu via `[tool.uv.sources]`. Dependencies: `mlflow`,
`fastapi`, `uvicorn[standard]`, `pydantic`, `pydantic-settings`, `numpy`,
`scipy`, `pandas`, `joblib`, `torch`. Dev: `pytest`, `httpx`.

### `train_and_register.py` (CLI)
The "offline training" job. Imports `generate_data`, `Surrogate`,
`train_surrogate` from the parent capstone. Wraps everything in
`with mlflow.start_run():`. Logs hyperparams (n_data, hidden, lr, epochs),
final train/val MSE, the training-history CSV, and the model itself via
`mlflow.pytorch.log_model(..., registered_model_name="lmm-surrogate")`.
After the run, `client.set_registered_model_alias("lmm-surrogate",
"candidate", new_version)` — so the new version is discoverable but does
not auto-promote.

### `app/config.py`
Pydantic `Settings` reading env vars (with sensible defaults):
`MLFLOW_TRACKING_URI` (default `./mlruns`), `MODEL_NAME` (default
`lmm-surrogate`), `MODEL_ALIAS` (default `production`). Loaded once,
injected via FastAPI dependency.

### `app/schemas.py`
Pydantic models with **field validators that enforce training bounds**:

- `Instrument(T, K, F)` — `T` in `[T_LO, T_HI]`, `F` in `[F_LO, F_HI]`,
  `log(K/F)` in `[LOG_M_LO, LOG_M_HI]`.
- `Params(sig_a, sig_c, sabr_alpha, rho_inf)` — each in its
  `LMM_PARAM_LO/HI` slot.
- `CalibrateRequest{instruments, market_ivs}` /
  `CalibrateResponse{theta_star, cost, success, model_version, verify}`.
- `PriceRequest{params, instruments}` / `PriceResponse{ivs, model_version}`.
- `ModelVersion{version, aliases, created_at, run_id}` /
  `ModelsListResponse{name, versions}`.
- `PromoteRequest{version, alias}`.

The bound constants live in `app/config.py` (mirror the parent capstone
ranges) so schemas can import them.

### `app/registry.py`
Thin wrapper around `mlflow.tracking.MlflowClient`:
- `load_model_by_alias(name, alias) -> (model, version_int)`
- `list_versions(name) -> list[ModelVersion]`
- `set_alias(name, version, alias) -> None`

Nothing leaks `mlflow.*` types out of this module — the routes only see
the pydantic schemas.

### `app/services.py`
Two pure functions over `model` + request:
- `run_calibration(model, instruments, market_ivs)` — wraps the parent's
  `calibrate()` + `verify()`. Returns a dict matching `CalibrateResponse`
  (minus `model_version`, which the route adds).
- `run_pricing(model, params, instruments)` — wraps the parent's `nn_iv()`.

Both are sync. PyTorch inference is CPU-bound and short; async would add
nothing.

### `app/routes/calibrate.py`, `price.py`, `models.py`
One `APIRouter` each. Routes pull `model` and `model_version` from
`app.state` (set by `main.py` at startup). `/models` and `/promote` use
the `registry.py` wrappers. `/promote` does **not** reload the in-process
model — that requires a restart, which is the production-correct
behaviour (atomic, no partial state).

### `app/main.py`
The lifespan event:
1. Reads `Settings`.
2. `mlflow.set_tracking_uri(settings.tracking_uri)`.
3. Calls `registry.load_model_by_alias(model_name, alias)`.
4. Stores `model` and `model_version` on `app.state`.
5. Includes the three routers.

If the alias doesn't resolve (no registered model yet), the app should
fail fast with a clear error — not boot in a degraded state.

### `tests/test_e2e.py`
One pytest module. Fixture: smoke-train a tiny surrogate (`n_data=200`,
`epochs=20`) into a `tmp_path / "mlruns"`, set `production` alias, boot a
`TestClient` against that tracking URI. Tests:

1. `POST /calibrate` → success, `theta_star` within bounds.
2. `POST /price` → response shape correct, IV in plausible range.
3. `GET /models` → returns one entry with `production` alias.
4. `POST /models/lmm-surrogate/promote` → 200 OK.
5. `POST /price` with out-of-bounds `T=20.0` → 422.

---

## Ordered build steps (the 3-hour path)

Work top-to-bottom. After each step, run the verification command. If
green, move on. If red, fix before adding more code.

### Step 1 — scaffold (5 min)
- Init the uv project: `uv init --no-readme` inside `api_extension/`.
- Edit `pyproject.toml` to pin deps + CPU torch source.
- `uv sync`.

✅ `uv run python -c "import mlflow, fastapi, torch; print('ok')"`.

### Step 2 — train + register (45 min)
- Open `train_and_register.py`. Read the TODOs in order.
- Wire imports from the parent capstone with the `sys.path` shim.
- Implement the `with mlflow.start_run():` block.
- Add `mlflow.pytorch.log_model(..., registered_model_name=...)`.
- Set `@candidate` alias on the new version.

✅ `uv run python train_and_register.py --n-data 1000 --epochs 100`
   → `mlruns/` exists, prints version number.
✅ `uv run mlflow ui` (separate terminal) → see the run + the registered
   model.
✅ Manually promote v1 to `production` via the UI or with a one-liner:
   ```bash
   uv run python -c "from mlflow.tracking import MlflowClient; \
     MlflowClient().set_registered_model_alias('lmm-surrogate', 'production', 1)"
   ```

### Step 3 — schemas + config (30 min)
- `app/config.py`: Pydantic Settings + the LMM bound constants.
- `app/schemas.py`: all pydantic models + the field validators.

✅ `uv run python -c "from app.schemas import Instrument; \
   Instrument(T=1.0, K=0.03, F=0.035); \
   try: Instrument(T=20.0, K=0.03, F=0.035) \
   except Exception as e: print('correctly rejected:', e)"`.

### Step 4 — registry wrapper (15 min)
- `app/registry.py`: three functions. Use `mlflow.pytorch.load_model` for
  the load and `MlflowClient` for the rest.

✅ `uv run python -c "from app.registry import load_model_by_alias; \
   m, v = load_model_by_alias('lmm-surrogate', 'production'); print(v)"`.

### Step 5 — services + routes + main (60 min)
- `app/services.py`: two functions wrapping the parent capstone.
- `app/routes/*.py`: thin route handlers, one router per file.
- `app/main.py`: lifespan + include_router.

✅ `uv run uvicorn app.main:app --reload`. In another terminal:
   ```bash
   curl http://localhost:8000/models | jq
   curl -X POST http://localhost:8000/calibrate \
     -H 'content-type: application/json' \
     -d '{"instruments":[{"T":1.0,"K":0.03,"F":0.035}], "market_ivs":[0.36]}' | jq
   ```

### Step 6 — tests (30 min)
- `tests/test_e2e.py`. Fixture trains tiny model into `tmp_path`.
- Five tests as listed above.

✅ `uv run pytest -v`.

### Step 7 — README polish (5 min)
- Update the top of this file with anything you learnt.
- Add example `curl` invocations.

---

## What to ignore (STRETCH — not in the 3-hour budget)

These are real-world extras. Skip them. If you finish early and have a
clear head, pick **one**.

- **Log every `/calibrate` call as a nested MLflow run.** Trivially extends
  the audit trail to per-request granularity. Adds a `mlflow.start_run(nested=True)`
  inside the calibrate route. Watch out for tracking-uri contention.
- **JWT auth on `/promote`.** Reuse the API capstone's Argon2 + JWT
  pattern. Only `/promote` needs it; the read endpoints stay public.
- **Sqlite-backed tracking server.** `mlflow server --backend-store-uri
  sqlite:///mlflow.db --default-artifact-root ./mlartifacts`. Switch
  `MLFLOW_TRACKING_URI` to `http://localhost:5000`.
- **Hot reload on promote.** Have `/promote` re-load the model into
  `app.state`. Tempting but production-incorrect (atomicity, partial state
  during reload). Skip.
- **ONNX export + batched inference.** `mlflow.onnx.log_model` then serve
  via ONNXRuntime for ~10x lower latency. Useful real-world skill, but
  costs you the whole budget.
- **Drift detection.** Compare each new training run's val MSE to the
  median of the last N runs; fail registration if regression > 2σ.
  Conceptually clean, mechanically fiddly with MLflow search.

---

## Verification — what "done" looks like

End-to-end:

1. `uv run python train_and_register.py` registers v1, aliased `@candidate`.
2. Manually promote v1 → `@production`.
3. `uv run uvicorn app.main:app` boots without errors.
4. `curl POST /calibrate` returns `theta_star` close to the parent
   capstone's `true_params = [0.18, 0.40, 0.015, 0.30]`, calib RMSE
   under 25 bp.
5. `uv run python train_and_register.py --hidden 128 128` registers v2,
   aliased `@candidate`.
6. `curl POST /models/lmm-surrogate/promote` with `version=2, alias=production`.
7. Restart the API. `curl POST /calibrate` again → same answer (since
   both versions trained on the same toy), but `model_version` field in
   the response now shows `2`.
8. `uv run pytest` is green.

Time check: ~2h 50min if you don't yak-shave the schemas. Stop at 3h hard
cap — if you're not done, skip the tests and document what's missing.
