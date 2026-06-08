# NAVIGATION.md — How to read this capstone in VSCode

A guided tour through the **api_extension** project: how the pieces fit
together, the order to read them, and how to trace a real request end-to-end
with the debugger. Designed so you can sit down with VSCode and the project
and build a complete mental model in ~60 minutes.

For the *production* of the model (the OFFLINE side), see also
[`train_and_register.py`](train_and_register.py). For the LANGUAGE
patterns this project demonstrates (LCEL-style FastAPI, MLflow registry,
PyTorch surrogate), see:

- [`README.md`](README.md) — what this project does + how to run it
- [`../../../../api_engineering/api_engineering_cheatsheet.md`](../../../../api_engineering/api_engineering_cheatsheet.md) — generic FastAPI patterns
- [`../../../../toolkit/mlflow_cheatsheet.md`](../../../../toolkit/mlflow_cheatsheet.md) — MLflow registry patterns

---

## 0. One-time VSCode setup (2 min)

```
Ctrl+Shift+P  →  Python: Select Interpreter
              →  pick .venv/bin/python in this capstone (api_extension/)
```

This gives you:
- F12 / Ctrl+Click jumps across the whole codebase
- The integrated terminal auto-activates the right venv (no `source activate`)
- Pylance shows types + parameter hints inline

Also install the **Python Debugger** extension (`ms-python.debugpy`) if you haven't —
required for Phase 3 below.

Drop a `.vscode/launch.json` so debug runs are one-click (`F5`):

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Uvicorn (debug)",
      "type": "debugpy",
      "request": "launch",
      "module": "uvicorn",
      "args": ["app.main:app", "--reload", "--port", "8003"],
      "justMyCode": false
    }
  ]
}
```

`justMyCode: false` lets you step INTO library code (FastAPI, MLflow, PyTorch) —
critical when "why does this magic line work?" needs an answer.

---

## 1. The mental model — read this BEFORE the code

### Two distinct workflows that share one codebase

```
┌────────────────────────────────────────────────────────────────────┐
│ OFFLINE — runs occasionally (when you train a new model)           │
│                                                                    │
│   train_and_register.py                                            │
│     ↓                                                              │
│     [trains surrogate.Surrogate on synthetic data]                 │
│     ↓                                                              │
│     [logs model + signature + metadata to MLflow registry]         │
│     ↓                                                              │
│     [tags new version @candidate, awaiting human review]           │
└────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────┐
│ ONLINE — runs continuously (FastAPI server)                        │
│                                                                    │
│   uv run uvicorn app.main:app --port 8003                          │
│     ↓                                                              │
│     [app/main.py lifespan: load model from registry @ startup]     │
│     ↓                                                              │
│     [serves POST /calibrate, POST /price, GET /models,             │
│      POST /models/.../promote]                                     │
└────────────────────────────────────────────────────────────────────┘
```

These two workflows share `surrogate.py` (the parent capstone — the
class definition + numerical methods). Both worlds import from it.

### Layered architecture inside `app/`

```
┌───────────────────────────────────────────────────────────────┐
│  app/routes/       ← OUTERMOST — HTTP layer                   │
│  app/deps.py       ← injection helpers (ModelDep, SettingsDep)│
│  app/schemas.py    ← BOUNDARY types (Pydantic in/out shapes)  │
│  app/services.py   ← BUSINESS LOGIC (no HTTP, no SQL/MLflow)  │
│  app/registry.py   ← REGISTRY LAYER (MLflow client wrappers)  │
│  surrogate.py      ← MATHEMATICS (parent capstone, imported)  │
└───────────────────────────────────────────────────────────────┘
```

Strict directionality:
- Routes call services, never the other way around
- Services call registry/surrogate, never routes
- Schemas are imported by everyone, depend on no one

Whenever you read code that violates this (e.g., a route directly calling
`mlflow.pytorch.load_model`), it's a smell.

### The Pydantic-at-the-boundary contract

```
inbound  JSON  →  CalibrateRequest (validates)  →  dict   →  service
                      ↑ 422 on bad data
                                                              ↓
outbound JSON  ←  CalibrateResponse (filters)   ←  ORM/dict ←  service
```

Pydantic does two jobs: **validate inputs** at the request boundary, AND
**constrain outputs** (response_model filters what gets serialised — prevents
accidental data leaks).

---

## 2. The file map — what each file does

Spend 3 minutes confirming this from the file explorer:

```
api_extension/
├── README.md                ← spec + how to run; ALWAYS READ FIRST
├── NAVIGATION.md            ← this guide
├── pyproject.toml           ← dependencies (mlflow, fastapi, pydantic, torch)
├── train_and_register.py    ← OFFLINE CLI (not part of the running server)
│
├── app/
│   ├── main.py              ← ENTRY POINT — FastAPI app + lifespan
│   ├── config.py            ← Settings (env-backed config + bounds constants)
│   ├── deps.py              ← Annotated[..., Depends(...)] aliases
│   ├── schemas.py           ← Pydantic in/out models (the data contract)
│   ├── registry.py          ← MLflow client wrappers (load, list, set_alias)
│   ├── services.py          ← run_calibration, run_pricing (the business logic)
│   └── routes/
│       ├── calibrate.py     ← POST /calibrate
│       ├── price.py         ← POST /price
│       └── models.py        ← GET /models, POST /models/{name}/promote
│
├── tests/
│   └── test_e2e.py          ← integration tests (one of the best docs)
│
└── mlflow.db, mlruns/       ← runtime artifacts (gitignored, regen on train)
```

---

## 3. Phase 1 — Static reading (20 min, no execution yet)

### 3.1 Read `README.md` (5 min)

Already done in step 1, but confirm you understand:
- What the model is (NN surrogate for LMM swaption calibration)
- The two workflows (offline / online)
- How to run it locally

### 3.2 Skim `pyproject.toml` (1 min)

Look at `[project.dependencies]` — gives you the technology surface:

```toml
"mlflow>=2.12"            # ← registry
"fastapi>=0.115"          # ← HTTP framework
"pydantic-settings>=2.2"  # ← env-driven Settings
"torch>=2.2"              # ← the model
"numpy", "scipy"          # ← numerical methods
```

You need a working knowledge of FastAPI, Pydantic, and MLflow's registry API
to read this code fluently.

### 3.3 Read `app/main.py` end to end (5 min)

```
Ctrl+P → main.py
Ctrl+Shift+O → outline (shows lifespan, create_app, root, app)
```

Read top-down:
1. **Imports** — `mlflow`, `FastAPI`, then the local modules (`config`, `registry`, `routes`)
2. **`lifespan` function** — runs at startup. The four lines that matter:
   ```python
   settings = get_settings()
   mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
   model, version = load_model_by_alias(settings.model_name, settings.model_alias)
   app.state.model, app.state.model_version = model, version
   ```
3. **`create_app`** — instantiates FastAPI, includes routers, returns it
4. **`app = create_app()`** at module bottom — the importable symbol uvicorn looks for

Use `Ctrl+Click` to navigate:
- Click `get_settings` → opens `app/config.py`
- Click `load_model_by_alias` → opens `app/registry.py`
- `Ctrl+-` (or `Alt+Left`) to navigate BACK in your reading history

### 3.4 Skim `app/schemas.py` — the data contract (5 min)

```
Ctrl+P → schemas.py
Ctrl+Shift+O → outline (Instrument, Params, CalibrateRequest, ...)
```

You don't need to read implementations — just learn the SHAPES:

| Schema | Means |
|---|---|
| `Instrument(T, K, F)` | One swaption-like coordinate |
| `Params(sig_a, sig_c, sabr_alpha, rho_inf)` | The 4 LMM parameters |
| `CalibrateRequest{instruments, market_ivs}` | What you send to /calibrate |
| `CalibrateResponse{theta_star, cost, success, model_version, verify}` | What you get back |
| `VerifyReport{rows, rmse_calib_bp, rmse_surrogate_bp}` | The audit table |
| `PriceRequest{params, instruments}` / `PriceResponse{ivs, model_version}` | The /price contract |
| `ModelVersion`, `ModelsListResponse`, `PromoteRequest` | Registry-management types |

Notice the bounds validators inside `Instrument` and `Params` — they reject
out-of-training-region inputs at the boundary. This is the only place
validation lives; routes and services trust the inputs.

### 3.5 Skim `app/config.py` + `app/deps.py` (3 min)

- **`config.py`** — one `Settings` class (`mlflow_tracking_uri`, `model_name`, `model_alias`) + bound constants (`LMM_PARAM_LO/HI`, `T_LO/HI`, …). All overridable via env vars (the docstring shows examples).
- **`deps.py`** — three Annotated type aliases (`SessionDep`, `SettingsDep`, `ModelDep`). These are the modern `Annotated[T, Depends(...)]` form. Routes use them as parameter types.

You've now seen the data structures + how they're injected. Next is the
behavior.

### 3.6 Skim `app/registry.py` + `app/services.py` (3 min)

- **`registry.py`** — three functions: `load_model_by_alias`, `list_versions`, `set_alias`. Wrappers around `mlflow.MlflowClient` so the rest of the app never imports MLflow directly.
- **`services.py`** — `run_pricing(model, params, instruments)` and `run_calibration(model, instruments, market_ivs)`. They convert Pydantic → numpy, call the parent capstone's `nn_iv` / `calibrate`, return dicts/lists ready for Pydantic to serialize.

Don't read every line — just confirm what each function takes in and what it returns.

---

## 4. Phase 2 — Trace ONE endpoint end-to-end (15 min)

Pick `POST /calibrate` (the richest endpoint — covers optimizer, verify report, registry).

### 4.1 Start at the route

```
Ctrl+P → routes/calibrate.py
```

The handler is ~8 lines. Read them. Notice:
- `req: CalibrateRequest` — Pydantic validates the body before this function runs.
- `model_and_version: ModelDep` — FastAPI resolves the dep (looks up `app.state.model`).
- Calls `run_calibration(...)`, builds a `CalibrateResponse`, returns it.

### 4.2 Follow into `run_calibration`

`F12` on `run_calibration` → jumps to `services.py`.

This is where the work happens:

```python
def run_calibration(model, instruments, market_ivs):
    device = _device(model)
    tuples = _instruments_to_tuples(instruments)   # Pydantic → list of (T,K,F)
    ivs_np = np.asarray(market_ivs, dtype=np.float64)
    x0     = (LMM_PARAM_LO + LMM_PARAM_HI) / 2     # midpoint of the box
    bounds = (LMM_PARAM_LO, LMM_PARAM_HI)

    res = calibrate(model, tuples, ivs_np, x0, bounds, device)   # ← PARENT
    theta_star = res.x

    verify_rep = _build_verify_report(...)         # three-way IV comparison

    return { "theta_star": ..., "cost": ..., "success": ..., "verify": verify_rep }
```

### 4.3 Follow into the parent capstone's `calibrate`

`F12` on `calibrate` (or any of `nn_iv`, `mock_lmm_price`, etc.) → opens
`surrogate.py` — this is the PARENT capstone, the math.

You'll see:
```python
def calibrate(model, market_instruments, market_ivs, x0, bounds, device):
    res = least_squares(
        fun = lambda p: nn_iv(model, p, market_instruments, device) - market_ivs,
        x0  = x0,
        bounds = bounds,
    )
    return res
```

Scipy's `least_squares` is the optimizer; the lambda is the residual. Every
candidate `p` the optimizer tries triggers a call to `nn_iv` (a forward pass
through the surrogate). This is the **headline pattern**: surrogate replaces
the slow MC inside the inner loop.

### 4.4 Read `nn_iv` (still in surrogate.py)

```python
def nn_iv(model, params, instruments, device):
    feats = [[*params, T_, np.log(K_/F_), F_] for T_, K_, F_ in instruments]
    x = torch.tensor(feats, dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        return model(x).cpu().numpy()
```

One forward pass for ALL instruments. Return numpy so scipy can do its math.

### 4.5 Back-trace using Ctrl+-

`Ctrl+-` (or `Alt+Left`) navigates BACK through your reading history. Step
back from `surrogate.py` → `services.py` → `routes/calibrate.py`. You've now
seen the entire chain. The other endpoints are variations on this same shape.

---

## 5. Phase 3 — Open `/docs` for the runtime view (3 min)

```bash
# In a terminal at api_extension/
uv run uvicorn app.main:app --reload --port 8003
```

Browse to `http://localhost:8003/docs`. This is the **swagger UI** — every
endpoint with its inputs / outputs / status codes. Useful to:
- Confirm your mental model from the static read is right
- Try requests live ("Try it out" button)
- See the auto-generated JSON schemas

Try `POST /price` with these body values:
```json
{
  "params": {"sig_a": 0.18, "sig_c": 0.40, "sabr_alpha": 0.015, "rho_inf": 0.30},
  "instruments": [{"T": 1.0, "K": 0.030, "F": 0.035}]
}
```

You'll get a single predicted IV back. This is the simplest path through the
system — one forward pass, one response.

---

## 6. Phase 4 — Debugger trace (the killer step, 20 min)

The static read tells you WHAT the code does. The debugger shows you HOW it
runs. This is the biggest single jump in understanding.

### 6.1 Set breakpoints — click in the gutter (or F9)

Set them at:
- `app/main.py` — first line inside `lifespan` (`settings = get_settings()`)
- `app/routes/calibrate.py` — first line of `calibrate_endpoint`
- `app/services.py` — first line of `run_calibration`
- `surrogate.py` — first line of `calibrate` (the parent's optimizer function)
- `surrogate.py` — first line of `nn_iv`

### 6.2 Hit F5 to start

The debugger boots uvicorn. The **lifespan breakpoint fires first**:

| Key | What it does |
|---|---|
| **F10** | Step OVER (don't dive into function calls) |
| **F11** | Step INTO (dive into the function) |
| **Shift+F11** | Step OUT (run to end of current function) |
| **F5** | Continue to next breakpoint |

Walk through the lifespan with `F10`:
- `settings = get_settings()` — hover `settings` in the variables panel; expand it; see `mlflow_tracking_uri`, `model_name`, `model_alias`
- `mlflow.set_tracking_uri(...)` — MLflow's global state is now set
- `model, version = load_model_by_alias(...)` — F11 into this to see registry resolution; F11 again INTO `mlflow.pytorch.load_model` if you want to see how MLflow assembles the model
- `app.state.model = model` — now the model is pinned to the app

`F5` to continue. The server is now waiting for HTTP requests.

### 6.3 Fire a real request from another terminal

```bash
curl -X POST http://localhost:8003/calibrate \
  -H 'content-type: application/json' \
  -d '{
    "instruments": [
      {"T": 1.0, "K": 0.030, "F": 0.035},
      {"T": 2.0, "K": 0.040, "F": 0.040}
    ],
    "market_ivs": [0.3591, 0.3657]
  }'
```

### 6.4 Watch the request flow through your breakpoints

1. **`calibrate_endpoint`** fires first. Inspect:
   - `req` in the variables panel → see the validated Pydantic object
   - `model_and_version` → see the tuple `(GraphModule, 1)`

2. **F11** into `run_calibration` (or F5 to skip to the next breakpoint that's already set there).

3. Inside `run_calibration`:
   - Step through with F10 — see `tuples`, `ivs_np`, `x0`, `bounds` materialise
   - F11 on `calibrate(...)` → enter the parent capstone's optimizer function

4. Inside `calibrate` (surrogate.py):
   - F11 on `least_squares(...)` → you enter SCIPY code (set `justMyCode: false` enables this)
   - Or **F5** to let it run — scipy will call your lambda many times
   - The breakpoint at `nn_iv` will fire on **every iteration** of the optimizer
   - Watch the variables panel — see `params` change as scipy explores the parameter space
   - Press F5 a few times to watch the optimizer converge

5. Eventually `run_calibration` returns with `theta_star`. Watch the `verify_rep` get built.

6. Back in the route, the `CalibrateResponse` is constructed and returned.

### 6.5 Look at the Call Stack panel

While paused, look at the **Call Stack** (left sidebar):
```
nn_iv                            ← currently here
  → <lambda>                     ← scipy calling your residual
    → least_squares              ← scipy's optimizer
      → calibrate                ← parent capstone wrapper
        → run_calibration        ← service
          → calibrate_endpoint   ← FastAPI route
            → ... FastAPI internals ...
```

Click any frame to inspect that scope. This is how you understand depth-of-call.

### 6.6 The "aha moment"

After seeing scipy call `nn_iv` ~30 times in a fraction of a second, you'll
understand viscerally what the surrogate pattern achieves. If `nn_iv` called
the REAL MC pricer (mock_lmm_price), each iteration would take seconds and
calibration would take minutes. The neural net forward pass takes microseconds
— same algorithm, three orders of magnitude faster.

---

## 7. Phase 5 — Use tests as ground-truth documentation (5 min)

```
Ctrl+P → tests/test_e2e.py
```

`test_e2e.py` is essentially a SPEC of how the app should behave. Each test
documents one expected behavior:

| Test | Documents |
|---|---|
| `test_price_endpoint_returns_iv` | `POST /price` with valid input → 200 with `ivs` list, `model_version=1` |
| `test_calibrate_endpoint_recovers_true_params` | `POST /calibrate` succeeds, returns `verify` with RMSE numbers |
| `test_models_list_shows_production_alias` | `GET /models` lists registered versions with their aliases |
| `test_promote_endpoint_sets_alias` | `POST /models/.../promote` moves an alias to a different version |
| `test_price_rejects_out_of_bounds_T` | Out-of-bounds inputs are rejected at the boundary (Pydantic 422) |

Run them:

```bash
uv run pytest -v
```

A green run is your assurance that the app behaves as documented. A red test
points at exactly which behavior broke.

---

## 8. Useful VSCode shortcuts (memorize 5)

| Shortcut | What it does | When |
|---|---|---|
| `Ctrl+P` | Quick file open (fuzzy) | Jump to any file |
| `Ctrl+T` | Workspace symbol search | "Where is `Surrogate` defined?" |
| `Ctrl+Shift+F` | Workspace text search | "Find everywhere using `nn_iv`" |
| `F12` / `Ctrl+Click` | Go to definition | Follow imports |
| `Ctrl+-` (Alt+Left) | Navigate BACK | Reading history back-button |
| `Ctrl+Shift+O` | File outline | Skim symbols in current file |
| `Shift+F12` | Find all references | Who calls this function? |
| `F2` | Rename symbol | Refactor safely |
| `F5` / `F10` / `F11` | Debugger: continue / step-over / step-in | Active debugging |

The **most underused** is `Ctrl+-` — your reading history's back-button.
Combined with F12 it lets you dive deep then bounce back without losing
your place.

---

## 9. Self-test — can you answer these without re-reading?

After completing the walkthrough, you should be able to answer:

1. **What file owns the FastAPI lifecycle?** (app/main.py — `lifespan`)
2. **Where does the model get loaded into `app.state`?** (Inside `lifespan`)
3. **What's the difference between `mlflow.set_tracking_uri` and `MLFLOW_TRACKING_URI` env var?** (Same effect; the env var is automatically read by `pydantic-settings` and propagated through `Settings`)
4. **Where is bounds validation done?** (Pydantic validators inside schemas.py — `Instrument` and `Params` model_validators)
5. **Why doesn't `POST /promote` reload the model in-process?** (See `routes/models.py` comment — restart is the atomic swap; hot-reload has consistency issues)
6. **How does scipy's optimizer find θ\* using a neural network?** (The lambda `nn_iv(model, p, ...) - market_ivs` is the residual; scipy calls the NN forward pass many times to converge)
7. **Where does the `surrogate.py` parent capstone get found at import time?** (Via the `sys.path` shim in `app/__init__.py` — or `code_paths` bundled in the model artifact, depending on which fix you implemented)

If you can answer 5/7 without scrolling back, you've internalised the project.

---

## 10. Where to go next

- **Modify a route** — change `POST /price` to also return the model version's `validation_status` tag. Tests will guide you.
- **Add a new endpoint** — `GET /metadata` that returns the loaded model's description + tags from the registry. Practice the layered pattern.
- **Run the OFFLINE workflow** — `uv run python train_and_register.py` to create a v2 of the model. Watch the MLflow UI update. Restart the API to load v2.
- **Read the LESSONS** — [`../LESSONS.md`](../LESSONS.md) captures the patterns this project teaches at a higher level.
- **Read the cheatsheets**:
  - [`../../../../api_engineering/api_engineering_cheatsheet.md`](../../../../api_engineering/api_engineering_cheatsheet.md) — general FastAPI patterns this project uses
  - [`../../../../toolkit/mlflow_cheatsheet.md`](../../../../toolkit/mlflow_cheatsheet.md) — MLflow registry concepts
  - [`../../../../toolkit/ml_project_methodology.md`](../../../../toolkit/ml_project_methodology.md) — ML pipeline general patterns

---

## TL;DR — the 60-minute reading sequence

| Time | Phase | What |
|---|---|---|
| 0-5 min | 1.1 | Read `README.md` |
| 5-10 min | 1.2-1.3 | Skim `pyproject.toml`, build file map |
| 10-15 min | 3.3 | Read `app/main.py` lifespan |
| 15-20 min | 3.4 | Skim `app/schemas.py` shapes |
| 20-25 min | 3.5-3.6 | Skim `app/config.py`, `deps.py`, `registry.py`, `services.py` |
| 25-40 min | 4 | Trace `POST /calibrate` with F12/Ctrl+- through routes → services → surrogate |
| 40-45 min | 5 | Open `/docs`, send live requests |
| 45-60 min | 6 | Debugger walkthrough — F5/F10/F11 through a real `/calibrate` request |
| (later) | 7 | Read `test_e2e.py` for ground-truth behavior |

After that you've seen the project from every angle — static, runtime, dynamic. Each subsequent visit only deepens what's already there.
