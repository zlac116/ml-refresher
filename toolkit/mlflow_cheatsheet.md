# MLflow Cheatsheet (code-first, MLflow ≥ 2.12)

Patterns for tracking, logging, registering, and serving models — covering
sklearn, PyTorch, XGBoost/LightGBM, transformers, and custom `pyfunc`
models. Modern API only (aliases, not stages).

For phase-by-phase ML pipeline patterns see
[`ml_project_methodology.md`](ml_project_methodology.md). For a worked
implementation see
[`../quant_finance/capstones/04_lmm_nn_surrogate/api_extension/`](../quant_finance/capstones/04_lmm_nn_surrogate/api_extension/).

---

## 0. The general pattern — every MLflow run, every framework

Internalise this skeleton; everything else in this doc is a variation on
it. The pattern is the same regardless of framework (PyTorch, sklearn,
XGBoost, LightGBM, custom pyfunc).

```
SETUP (process-level — once per script)
  1. set_tracking_uri          where does metadata go? (sqlite/postgres/server)
  2. set_experiment            which experiment folder groups this run?

INSIDE the run  (with mlflow.start_run() as run:)
  3. log_params                what HYPERPARAMETERS defined this experiment? (one-shot)
  4. <do the work>             train / fit / build
  5. log_metric × N            what was the PROGRESS + final result? (can use step=)
  6. log_model                 save artifact + signature + auto-register in one call

AFTER the run  (with block has exited)
  7. (run is now FINISHED — visible in UI)
  8. set_registered_model_alias        tag the new version (@candidate / @production)
```

**Critical distinction**: `log_params`, `log_metric`, `log_model` live
**inside** the run — they're tied to *this execution*. Registry operations
(`set_registered_model_alias`) live **outside** because they act on the
registry, not the run. Aliasing inside works mechanically but is
semantically wrong — you can't roll back to a previous version's alias
from inside a different run's context.

**Params vs metrics** (often confused):

| | log_param | log_metric |
|---|---|---|
| Captures | **inputs** (knobs that defined the run) | **outputs** (results) |
| Examples | `lr`, `hidden`, `seed`, `n_data` | `train_loss`, `val_accuracy` |
| Mutable? | NO — set once per run (re-set raises) | YES — `step=` makes a time series |
| Type | str / int / float / bool | float only |

Canonical skeleton (PyTorch — adapt the flavour for sklearn/xgboost/etc):

```python
import mlflow, mlflow.pytorch
from mlflow import MlflowClient
from mlflow.models import infer_signature

# 1+2. SETUP
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("my-experiment")

# 3-6. INSIDE THE RUN
with mlflow.start_run(run_name="my-run") as run:
    mlflow.log_params({"lr": 1e-3, "hidden": str([64, 64]), "seed": 0})

    model, history = train(...)

    for epoch, (tr, va) in enumerate(zip(history["train"], history["val"])):
        mlflow.log_metric("train_loss", tr, step=epoch)
        mlflow.log_metric("val_loss",   va, step=epoch)

    sig = infer_signature(X_tr[:5], predict(model, X_tr[:5]))
    mlflow.pytorch.log_model(
        pytorch_model=model,
        name="model",                              # MLflow 3.x — not artifact_path=
        registered_model_name="my-model",
        signature=sig,
        input_example=X_tr[:5],
        serialization_format="pt2",                # PyTorch-specific safety
    )

# 7+8. AFTER THE RUN — registry operation, not a run operation
client = MlflowClient()
latest = max(int(mv.version) for mv in client.search_model_versions("name='my-model'"))
client.set_registered_model_alias("my-model", "candidate", str(latest))
```

**Variations** (all explained below):
- Autolog (sklearn/xgboost/lightgbm/lightning): replaces steps 3 + 5 with one line.
- Nested runs (HPO): `start_run(nested=True)` inside an outer `start_run`.
- Skip `registered_model_name=` if you don't want the model in the registry.
- Multiple `log_model` calls per run = ensembles; use different `name=`.

**Registry hygiene** (§10): for ANY model that goes into the registry, also set:
1. **Description** on the registered model + each version (`client.update_registered_model`, `client.update_model_version`).
2. **Tags** on both layers (owner / domain / framework / task / git_commit / validation_status).
Do this in code (`train_and_register.py`), never via UI "Add" — reproducibility matters.

The rest of this doc fills in the details for each step + the per-flavour
specifics for step 6.

---

## 1. Setup

```python
import mlflow

mlflow.set_tracking_uri("./mlruns")                          # file-based; or "http://host:5000"
mlflow.set_experiment("my-experiment")                       # created on first use
```

| URI | When |
|---|---|
| ❌ `./mlruns` (file-store) | **Deprecated** in MLflow 3.x — refuses to start without `MLFLOW_ALLOW_FILE_STORE=true`. Aliases / new features won't work. |
| ✅ `sqlite:///mlflow.db` | Single-user / dev — the new default. Zero infra, full feature support. |
| ✅ `postgresql://user:pass@host/db` | Team / prod backend store. |
| ✅ `http://server:5000` | Centralised, behind `mlflow server --backend-store-uri postgresql://...`. |

**Critical**: relative URIs (`sqlite:///mlflow.db`) resolve from CWD. Use
`sqlite:////absolute/path/mlflow.db` (four slashes) or pin
`MLFLOW_TRACKING_URI` to an absolute path so script + server + UI all
agree.

**Env var override** (don't hard-code URIs):
```bash
export MLFLOW_TRACKING_URI=http://mlflow.internal:5000
```

---

## 2. Runs — the core context

```python
with mlflow.start_run(run_name="baseline-rf") as run:
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", 0.93)
    # ... do work ...
print(run.info.run_id)                                       # e.g. for nested or linking
```

Common variants:

```python
mlflow.start_run(nested=True)                                # parent/child for HPO
mlflow.start_run(run_id="abc123")                            # resume / amend
mlflow.start_run(experiment_id="...")                        # pin experiment per-run
```

**Trap**: forgetting the `with` → run never closes → next `start_run` fails.

---

## 3. Logging params / metrics / artifacts

```python
# Params — scalars (str/int/float/bool). Set ONCE per run.
mlflow.log_param("lr", 1e-3)
mlflow.log_params({"lr": 1e-3, "hidden": "64,64", "epochs": 200})  # batch

# Metrics — can change over steps (training curves)
mlflow.log_metric("train_loss", 0.42)
mlflow.log_metric("train_loss", 0.31, step=1)
mlflow.log_metrics({"train_loss": 0.31, "val_loss": 0.35}, step=1)

# Artifacts — any file
mlflow.log_artifact("verify_report.csv")                     # one file
mlflow.log_artifacts("outputs/")                             # whole dir
mlflow.log_text(json.dumps(config), "config.json")           # inline string
mlflow.log_dict(config, "config.json")                       # dict → JSON

# Tags — searchable metadata (git SHA, user, etc.)
mlflow.set_tag("git_sha", "abc1234")
mlflow.set_tag("mlflow.user", "your-name")
```

**Trap**: `log_param` is **set-once**. Calling it twice with the same key raises.
For values that change, use a metric instead.

---

## 4. Model logging — sklearn

```python
import mlflow.sklearn
from mlflow.models import infer_signature

pipeline.fit(X_tr, y_tr)
pred = pipeline.predict(X_tr[:5])
signature = infer_signature(X_tr[:5], pred)

mlflow.sklearn.log_model(
    sk_model=pipeline,
    name="model",                                   # `artifact_path=` is deprecated in MLflow 3.x
    registered_model_name="my-classifier",   # auto-registers + new version
    signature=signature,
    input_example=X_tr[:5],
)
```

**Why log the Pipeline, not the bare estimator**: preserves preprocessing.
A `model.predict(raw_X)` after `load_model` then works on RAW inputs —
no separate scaler artifact to drag around.

---

## 5. Model logging — PyTorch

```python
import mlflow.pytorch
from mlflow.models import infer_signature

# Build an input example (numpy) + run it through to get an output example
x_example = X_tr[:5]
with torch.no_grad():
    y_example = model(torch.tensor(x_example, dtype=torch.float32)).cpu().numpy()
signature = infer_signature(x_example, y_example)

mlflow.pytorch.log_model(
    pytorch_model=model,
    name="model",                                   # `artifact_path=` is deprecated in MLflow 3.x
    registered_model_name="my-nn",
    signature=signature,
    input_example=x_example,
)
```

**Why**: `mlflow.pytorch` saves architecture + state_dict together. Loading
later doesn't need the class definition in scope (unlike `torch.save(model)`).
**Best practice (MLflow 3.x)**: pass `serialization_format="pt2"` — saves
via PyTorch's safe graph format. The default (cloudpickle) executes
arbitrary code on load, which MLflow now warns about explicitly.
**Trap (`.eval()` on a pt2-loaded model)**: pt2 returns an `ExportedProgram`,
not a `nn.Module` — `.eval()` raises `NotImplementedError: Calling eval() is
not supported yet.` Unwrap to a `GraphModule` after load:
```python
loaded = mlflow.pytorch.load_model(uri)
model = loaded.module() if hasattr(loaded, "module") else loaded
model.eval()                              # works on GraphModule
```
**Trap (torch `+cpu` label)**: torch installed from the CPU-only index has
a `+cpu` local version label; MLflow strips it to make the requirement
PyPI-installable. If that breaks your inference env, pass
`pip_requirements=["torch==2.12.0+cpu", ...]` explicitly to `log_model(...)`.

---

## 6. Model logging — XGBoost / LightGBM / CatBoost

```python
import mlflow.xgboost
mlflow.xgboost.log_model(xgb_model, "model", registered_model_name="...", signature=sig)

import mlflow.lightgbm
mlflow.lightgbm.log_model(lgb_model, "model", registered_model_name="...", signature=sig)

import mlflow.catboost
mlflow.catboost.log_model(cb_model, "model", registered_model_name="...", signature=sig)
```

Each flavour has a matching `load_model` and `autolog()`. Use the flavour
that matches the library — don't `pickle` it manually.

---

## 7. Model logging — Transformers / LangChain (LLM stacks)

```python
import mlflow.transformers
mlflow.transformers.log_model(
    transformers_model={"model": hf_model, "tokenizer": hf_tokenizer},
    name="model",                                   # `artifact_path=` is deprecated in MLflow 3.x
    task="text-classification",                 # required for the right pipeline at load
    registered_model_name="bert-sentiment",
)

import mlflow.langchain
mlflow.langchain.log_model(chain, "model", registered_model_name="rag-chain")
```

Use these for agent / retrieval workflows so the registry holds the whole
chain, not just one component. (See LMM extension Idea 1 — Drift Monitor —
for how this slots into a multi-agent setup.)

---

## 8. Custom models — `mlflow.pyfunc.PythonModel`

The escape hatch when no built-in flavour fits: ensembles, custom
preprocessing wrappers, RAG retrievers, anything with state.

```python
import mlflow.pyfunc

class MyWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        """Called ONCE when the model is loaded. Read artifacts here."""
        import joblib
        self.scaler = joblib.load(context.artifacts["scaler"])
        self.model  = mlflow.sklearn.load_model(context.artifacts["model"])

    def predict(self, context, model_input, params=None):
        """Called per request. Return a DataFrame / ndarray / list."""
        X_scaled = self.scaler.transform(model_input)
        return self.model.predict(X_scaled)

mlflow.pyfunc.log_model(
    name="wrapped",                              # `artifact_path=` is deprecated in MLflow 3.x
    python_model=MyWrapper(),
    artifacts={
        "scaler": "outputs/scaler.joblib",      # paths to files; copied into the model
        "model":  "runs:/<run_id>/model",       # or a runs:/ URI
    },
    pip_requirements=["scikit-learn==1.4.0", "joblib"],
    signature=signature,
    input_example=X_tr[:5],
    registered_model_name="wrapped-classifier",
)
```

**Why pyfunc**:
- Lets you bundle preprocessing + model + post-processing as ONE registered artifact.
- `load_model` then `predict(raw_X)` works end-to-end — the consumer doesn't need to know about your scaler.
- Standard interface (`predict(model_input)`) → works with `mlflow models serve`, `mlflow.evaluate`, and the registry without modification.

**Trap**: don't put heavyweight loading in `predict()` — it runs every
request. Put it in `load_context()`.

---

## 9. Signatures + input examples (best practice — always)

```python
from mlflow.models import infer_signature

signature = infer_signature(
    model_input=X_tr[:5],                       # numpy / pandas / dict
    model_output=model.predict(X_tr[:5]),       # whatever predict returns
    params={"temperature": 0.7},                # optional inference-time params
)
```

Without a signature, the registry stores a black box — downstream code
can't introspect the I/O contract. Per the docs: **log every model with
`signature=` and `input_example=`**.

`input_example` does double duty: shows up in the registry UI for
quick-glance documentation AND is used by `mlflow models serve` to
warm-up the prediction path.

---

## 10. Registry — modern alias-based workflow

```python
from mlflow import MlflowClient        # modern re-export (not mlflow.tracking)

client = MlflowClient()

# Set / move an alias (atomic, idempotent)
client.set_registered_model_alias(
    name="my-classifier",
    alias="production",                          # any lowercase-kebab string
    version="5",                                 # MUST be str
)

# Resolve an alias → version
mv = client.get_model_version_by_alias("my-classifier", "production")
print(mv.version, mv.run_id)

# List versions + their aliases
for mv in client.search_model_versions("name='my-classifier'"):
    print(mv.version, mv.aliases, mv.run_id)

# Delete an alias
client.delete_registered_model_alias("my-classifier", "candidate")
```

**Stages are deprecated** (`Staging` / `Production` / `Archived`). Use
aliases — they're free-form strings, movable, idempotent, and the
canonical pattern in MLflow ≥ 2.9. The docs explicitly recommend migration.

**Common alias names** (convention, not enforced): `production`,
`candidate`, `staging`, `champion`, `challenger`, `archived`.

### Registry hygiene — descriptions + tags (do this in code, never the UI)

**What's explicitly in the MLflow docs** (verbatim, [model-registry page](https://mlflow.org/docs/latest/ml/model-registry)):

> *"You can annotate the top-level model and each version individually using Markdown, including the description and any relevant information useful for the team"*
>
> *"Tags are key-value pairs that you associate with registered models and model versions, allowing you to label and categorize them by function or status."*

The only tag examples the docs prescribe:
- Model-level: `task: question-answering`
- Version-level: `validation_status: pending` → `approved` (the docs explicitly show this as the deployment-readiness workflow tag)
- `problem_type: regression`

Everything else (the tag schema below — `git_commit`, `owner`, `framework`, etc.) is **general MLOps best practice**, not MLflow-doc-prescribed. The docs say "use tags to categorise" without prescribing which keys.

**Placement** (matches the docs' tutorial pattern):
- The model is **registered inside the run** via `log_model(registered_model_name=...)` (step 6 of §0).
- Description / tag / alias calls happen **after the `with` block exits** — they're registry operations, not run operations. They work in either place mechanically, but "after the run" keeps the semantic split clean (run = execution; registry = catalogue).

Three layers, each with a distinct purpose:

| Layer | What goes here | When it changes |
|---|---|---|
| **Registered-model** description + tags | What the model IS — purpose, owner, framework, ML problem type | Edit once; rarely changes |
| **Model-version** description + tags | What's new in THIS version — training data, metrics, sign-off | Set per version |
| **Alias** | Deployment-state pointer (`@production`) | Movable across versions |

```python
from mlflow import MlflowClient
client = MlflowClient()

# ── REGISTERED MODEL: the contract (what it IS) ──
# Description: docs-recommended (Markdown supported)
client.update_registered_model(
    name="my-model",
    description=(
        "NN surrogate for LMM swaption calibration. Replaces the slow MC "
        "pricer inside scipy.optimize.least_squares.\n\n"
        "**Owner**: rates-quant   |   **Framework**: PyTorch"
    ),
)
# Tags: only `task` is explicitly shown in docs; rest is general MLOps practice.
for k, v in {
    "task":        "regression",           # ← docs' example
    "owner":       "rates-quant",          # general practice — who to ping
    "domain":      "rates",                # general practice — business area
    "model_type":  "surrogate",            # general practice
    "framework":   "pytorch",              # general practice
    "criticality": "low",                  # general practice — risk tier
}.items():
    client.set_registered_model_tag("my-model", k, v)

# ── MODEL VERSION: what changed in THIS release ──
version = "1"                                   # always str
client.update_model_version(
    name="my-model", version=version,
    description=(
        "64×64 MLP, SiLU. seed=0, n_train=10000, final val MSE 4.2e-5.\n"
        "Validation: pending VC review."
    ),
)
# Tags: only `validation_status` is explicitly in docs; rest is general practice.
for k, v in {
    "validation_status":  "pending",       # ← docs' explicit deployment-readiness tag
    "git_commit":         "abc1234",       # general practice — reproducibility
    "training_data_seed": "0",             # general practice
    "n_train":            "10000",         # general practice
    "architecture":       "64,64",         # general practice
    "final_val_mse":      "4.2e-5",        # general practice — denormalised search shortcut
}.items():
    client.set_model_version_tag("my-model", version, k, v)
```

**What's doc-prescribed vs general practice**:
- ✅ **Docs explicit**: descriptions on both layers, `task` tag, `validation_status: pending/approved` tag, aliases for deployment routing.
- 🟡 **General MLOps practice** (not MLflow-doc-prescribed): `git_commit`, `owner`, `framework`, `domain`, `criticality`, `training_data_seed`, `n_train`, `architecture`, denormalised metric tags. Adopt what matches your team's governance.

**Why programmatic, not UI**: a tag added via "Add" in the UI lives only
on that one server. The same call in `train_and_register.py` runs every
time you train — every new version is governed by default, and the code
is the source of truth.

**Run tag vs registry tag** — easy confusion:
- `mlflow.log_param / mlflow.set_tag` inside `start_run()` → attaches to that *execution*.
- `client.set_registered_model_tag / set_model_version_tag` → attaches to the *registry entry*. Survives the run.

The UI shows them in different places — registry tags are searchable via `client.search_registered_models(filter_string="tags.owner = 'rates-quant'")`.

**Don't put metrics in tags** (except as denormalised search shortcuts).
Metrics belong in `mlflow.log_metric`. A tag like `final_val_mse: 4.2e-5`
exists only so registry search can find "versions with val MSE < 1e-4".

---

## 11. Loading models

```python
# By alias (most common — what your service code does at startup)
model = mlflow.pyfunc.load_model("models:/my-classifier@production")

# By exact version (pinned, reproducible)
model = mlflow.pyfunc.load_model("models:/my-classifier/5")

# By run (the original logged artifact, before registration)
model = mlflow.pyfunc.load_model("runs:/<run_id>/model")

# Flavour-specific load (returns the native object, not pyfunc wrapper)
sk_model = mlflow.sklearn.load_model("models:/my-classifier@production")
nn_model = mlflow.pytorch.load_model("models:/my-nn@production")
```

**Rule of thumb**: use `mlflow.pyfunc.load_model(...)` in *service* code
— it gives you a consistent `.predict(X)` interface regardless of
underlying flavour. Use the flavour-specific loader only when you need
the native object (e.g., to keep training in PyTorch, or to use sklearn-specific methods).

---

## 12. Autologging — when to use, when to skip

```python
mlflow.autolog()                                 # universal — detects framework, logs everything
mlflow.sklearn.autolog()                         # sklearn-specific
mlflow.pytorch.autolog()                         # pytorch lightning hooks
mlflow.xgboost.autolog()
```

**Use autolog when**:
- Quick experiments / notebooks.
- You don't care about exactly *which* metrics/params get logged.
- Framework-standard hyperparameters are enough.

**Skip autolog when**:
- You want fine control over what's logged (e.g., custom validation metric).
- You're using a custom training loop (autolog hooks into framework APIs, custom loops bypass them).
- You need to call `log_model` with custom `signature` / `input_example` — autolog's auto-logged model may be less rich.

**Hybrid pattern** — autolog + explicit `log_model` for the final model:

```python
mlflow.sklearn.autolog(log_models=False)         # autolog params/metrics only
# ... training ...
mlflow.sklearn.log_model(pipeline, "model", registered_model_name="...", signature=sig)
```

---

## 13. Pattern — hyperparameter search with nested runs

```python
with mlflow.start_run(run_name="hpo-search"):
    mlflow.log_param("search_space", str(space))

    for trial_i, params in enumerate(grid):
        with mlflow.start_run(run_name=f"trial-{trial_i}", nested=True):
            mlflow.log_params(params)
            model = train(**params)
            val_score = evaluate(model)
            mlflow.log_metric("val_score", val_score)

# In the UI: parent run shows the search; child runs each show one trial.
```

For Optuna specifically: `mlflow.optuna.autolog()` does this automatically.

---

## 14. Pattern — cross-validation as metric series

```python
with mlflow.start_run():
    mlflow.log_params(params)
    scores = []
    for fold_i, (tr, va) in enumerate(kf.split(X)):
        model = train(X[tr], y[tr])
        s = evaluate(model, X[va], y[va])
        scores.append(s)
        mlflow.log_metric("fold_val_score", s, step=fold_i)

    mlflow.log_metric("cv_mean", np.mean(scores))
    mlflow.log_metric("cv_std",  np.std(scores))
```

The metric series `fold_val_score` is plotted in the UI as a curve —
useful to spot folds that misbehave.

---

## 15. Pattern — dataset tracking (`mlflow.data`)

```python
import mlflow.data

dataset = mlflow.data.from_pandas(
    df_train,
    source="s3://my-bucket/train.parquet",
    name="train-2026-06-07",
    targets="y",
)

with mlflow.start_run():
    mlflow.log_input(dataset, context="training")
    # ... train + log model ...
```

**Why**: links the run to its data source — answers "what data was this
model trained on?" in the UI. Supports pandas, numpy, spark, hugging face.

---

## 16. Pattern — model evaluation (`mlflow.evaluate`)

```python
result = mlflow.evaluate(
    model="models:/my-classifier@production",
    data=eval_df,                                # pandas DF
    targets="y_true",
    model_type="classifier",                     # or "regressor"
    evaluators=["default"],
)
print(result.metrics)        # accuracy, f1, log_loss, etc.
print(result.artifacts)      # confusion matrix, feature importance plots
```

Runs a comprehensive evaluation + logs the metrics + plots to the active
run. For custom metrics use `extra_metrics=[mlflow.metrics.make_metric(...)]`.

---

## 17. Serving — quick local

```bash
mlflow models serve -m "models:/my-classifier@production" -p 5001 --env-manager local
```

Then:
```bash
curl -X POST http://localhost:5001/invocations \
  -H 'Content-Type: application/json' \
  -d '{"dataframe_split": {"columns": ["f1","f2"], "data": [[1.0, 2.0]]}}'
```

**`--env-manager local`** uses your current env. Drop it to have MLflow
build a conda env from the model's logged `requirements.txt` (slow, but
reproducible).

For Docker: `mlflow models build-docker -m "models:/..." -n my-image`.

---

## 18. Anti-patterns (recognise instantly)

| Smell | What to do |
|---|---|
| `Stage="Production"` / `transition_model_version_stage(...)` | Use **aliases** (`set_registered_model_alias`) |
| Registered model with no description | `client.update_registered_model(name, description=...)` — the docs recommend annotating top-level + each version (§10) |
| Model version with no description | `client.update_model_version(name, version, description=...)` per release |
| Tags added via UI "Add" button | Add via `set_registered_model_tag` / `set_model_version_tag` inside `train_and_register.py` — reproducible, scriptable, every run governed by default |
| Metric stored as a tag | Use `mlflow.log_metric` (plottable, comparable). Tags are categorical search shortcuts, not numbers |
| Confusing `mlflow.set_tag` (run) with `client.set_registered_model_tag` (registry) | Run tags live on one execution; registry tags survive across runs. Use the right one |
| `artifact_path="model"` in `log_model(...)` | `name="model"` — `artifact_path` is **deprecated in MLflow 3.x** |
| `mlflow.pytorch.log_model(...)` without `serialization_format="pt2"` | Pass `pt2` — cloudpickle default executes arbitrary code on load |
| File-store backend `./mlruns` for tracking URI | `sqlite:///mlflow.db` — file-store is **maintenance-mode in MLflow 3.x**; aliases won't work properly |
| `pickle.dump(model, "model.pkl")` + `log_artifact(...)` | Use the flavour's `log_model` — gives signature, env, loader |
| `mlflow.tracking.MlflowClient` | `from mlflow import MlflowClient` (modern re-export) |
| `mlflow.start_run(...)` without `with` | Always context-manage; runs leak otherwise |
| `version=5` (int) to alias API | Cast `str(version)` — public API types it as str |
| Log dataset as `log_artifact("data.csv")` | Use `mlflow.data.log_input` — gets a proper data record |
| No `signature` / `input_example` on `log_model` | Always pass both — registry needs the contract |
| `log_param` for a value that changes | Use `log_metric` with `step=` |
| Hard-code tracking URI in scripts | Read from `MLFLOW_TRACKING_URI` env var |
| Loading model in `predict()` of a pyfunc | Move to `load_context()` |
| `mlflow.pytorch.log_model(model, "model")` without `registered_model_name` | Register at log time — saves a `register_model` round-trip later |
| One MLflow experiment for everything | One experiment per project / model family |

---

## 19. Configuration that earns its place

```bash
# .env / shell rc
export MLFLOW_TRACKING_URI=http://mlflow.internal:5000
export MLFLOW_EXPERIMENT_NAME=my-project
export MLFLOW_TRACKING_USERNAME=svc-user
export MLFLOW_TRACKING_PASSWORD=...

# Run-time SHA tag (auto-tag every run with the current git commit)
export MLFLOW_RUN_TAG_GIT_COMMIT=$(git rev-parse HEAD)
```

In code:

```python
mlflow.set_tag("git_commit", os.environ["MLFLOW_RUN_TAG_GIT_COMMIT"])
mlflow.set_tag("dataset_version", "2026-06-07")
mlflow.set_tag("owner", "your-name")
```

These tags are the searchable index in the UI — invest in them.

---

## 20. The minimal end-to-end pattern (sklearn example)

```python
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

mlflow.set_experiment("iris-baseline")

with mlflow.start_run(run_name="logreg-v1") as run:
    pipe = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(C=1.0))])
    pipe.fit(X_tr, y_tr)

    val_acc = pipe.score(X_va, y_va)
    mlflow.log_params({"model": "logreg", "C": 1.0})
    mlflow.log_metric("val_accuracy", val_acc)

    sig = infer_signature(X_tr[:5], pipe.predict(X_tr[:5]))
    mlflow.sklearn.log_model(
        sk_model=pipe,
        name="model",                                   # `artifact_path=` is deprecated in MLflow 3.x
        registered_model_name="iris-classifier",
        signature=sig,
        input_example=X_tr[:5],
    )

# Promote (AFTER the run closes — registry op, not a run op):
from mlflow import MlflowClient
client = MlflowClient()
latest = max(int(mv.version) for mv in client.search_model_versions("name='iris-classifier'"))
client.set_registered_model_alias("iris-classifier", "production", str(latest))
```

This is the spine. Adapt for every other flavour by swapping
`mlflow.sklearn.log_model` for the right one. See **§0** for the abstract
pattern explained step-by-step.

---

## 21. Cross-references

- Production wrapping example (FastAPI + MLflow alias serving):
  [`../quant_finance/capstones/04_lmm_nn_surrogate/api_extension/`](../quant_finance/capstones/04_lmm_nn_surrogate/api_extension/)
- ML pipeline patterns (phase-by-phase):
  [`ml_project_methodology.md`](ml_project_methodology.md)
- Deployment downstream: [`deployment.ipynb`](deployment.ipynb)
- Official docs: <https://mlflow.org/docs/latest/index.html>
