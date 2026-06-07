"""Train an LMM NN surrogate and register it with MLflow.

This script is the *offline* half of the workflow: it owns training. The
FastAPI app in `app/` is the *online* half — it never trains, only loads
from the MLflow registry and serves.

You can run this script repeatedly. Every run:
  - creates a new MLflow run (params + metrics + artifacts logged),
  - creates a new version of the `lmm-surrogate` registered model,
  - tags the new version with the `@candidate` alias.

Promotion (`@candidate` -> `@production`) is a SEPARATE human step done
either via the API's POST /models/{name}/promote endpoint or via:

    uv run python -c "from mlflow import MlflowClient; \
        MlflowClient().set_registered_model_alias('lmm-surrogate', 'production', '7')"

Usage:
    uv run python train_and_register.py
    uv run python train_and_register.py --n-data 20000 --epochs 3000 --hidden 64 64

Run from `api_extension/`. The script adds the parent capstone directory
to sys.path so it can re-use generate_data / Surrogate / train_surrogate.
"""
import argparse
import sys
from pathlib import Path

# Make the parent capstone importable WITHOUT modifying it. This is the only
# place the shim lives — once the parent is on sys.path, the rest of the
# extension imports from it normally.
PARENT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PARENT_DIR))

import mlflow                        # noqa: E402
import mlflow.pytorch                # noqa: E402
import numpy as np                   # noqa: E402
import torch                         # noqa: E402
from mlflow import MlflowClient      # noqa: E402  modern re-export
from mlflow.models import infer_signature  # noqa: E402

from lmm_nn_capstone import (        # noqa: E402  (parent capstone)
    LMM_PARAM_HI,
    LMM_PARAM_LO,
    N_FEATURES,
    Surrogate,
    generate_data,
    split_train_val,
    train_surrogate,
)

MODEL_NAME = "lmm-surrogate"
CANDIDATE_ALIAS = "candidate"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train + register an LMM NN surrogate.")
    p.add_argument("--n-data", type=int, default=10_000)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--hidden", type=int, nargs="+", default=[64, 64])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--tracking-uri",
        type=str,
        default="./mlruns",
        help="MLflow tracking URI. Default './mlruns' is file-backed. "
             "For a sqlite-backed server use 'http://localhost:5000'.",
    )
    p.add_argument(
        "--experiment",
        type=str,
        default="lmm-surrogate-training",
        help="MLflow experiment name. Created on first run.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # TODO 1 — Configure MLflow tracking.
    # WHY: every script that talks to MLflow needs to know the tracking
    # backend. Default is './mlruns/' (file-based, zero infra). For a
    # sqlite/Postgres-backed server, this becomes the URL.
    # HINT: mlflow.set_tracking_uri(args.tracking_uri)
    #       mlflow.set_experiment(args.experiment)
    # ------------------------------------------------------------------
    # raise NotImplementedError("TODO 1: set_tracking_uri + set_experiment")
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)

    # ------------------------------------------------------------------
    # TODO 2 — Train the surrogate (delegate to the parent capstone).
    # WHY: this is the work we're already doing in the parent script —
    # generate_data, split, build, train. Re-use, don't re-write.
    # HINT:
    #     torch.manual_seed(args.seed)
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     X, y                   = generate_data(args.n_data, args.seed)
    #     X_tr, y_tr, X_va, y_va = split_train_val(X, y, args.val_frac, args.seed)
    #     model                  = Surrogate(N_FEATURES, tuple(args.hidden)).to(device)
    #     history                = train_surrogate(
    #         model, X_tr, y_tr, X_va, y_va, args.epochs, args.lr, device
    #     )
    # ------------------------------------------------------------------
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, y = generate_data(args.n_data, args.seed)
    X_tr, y_tr, X_va, y_va = split_train_val(X, y, args.val_frac, args.seed)
    model = Surrogate(N_FEATURES, tuple(args.hidden)).to(device)
    history = train_surrogate(model, X_tr, y_tr, X_va, y_va, args.epochs, device)

    # ------------------------------------------------------------------
    # TODO 3 — Open a tracked run + log everything.
    # WHY: a "run" is one execution of training. Logging params + metrics
    # + artifacts inside the run lets you later compare runs in `mlflow ui`
    # and answer "which config gave the best val MSE?".
    # PATTERN:
    #     with mlflow.start_run(run_name=f"surrogate-{args.seed}") as run:
    #         mlflow.log_params({
    #             "n_data":   args.n_data,
    #             "hidden":   str(args.hidden),   # MLflow params want scalars
    #             "lr":       args.lr,
    #             "epochs":   args.epochs,
    #             "val_frac": args.val_frac,
    #             "seed":     args.seed,
    #         })
    #         for epoch, (tr, va) in enumerate(zip(history["train"], history["val"])):
    #             mlflow.log_metric("train_mse", tr, step=epoch)
    #             mlflow.log_metric("val_mse",   va, step=epoch)
    #         mlflow.log_metric("final_train_mse", history["train"][-1])
    #         mlflow.log_metric("final_val_mse",   history["val"][-1])
    #         mlflow.log_metric("best_val_mse",    min(history["val"]))
    # ------------------------------------------------------------------
    # raise NotImplementedError("TODO 3: open a run + log params + per-epoch metrics")
    with mlflow.start_run(run_name=f"surrogate-{args.seed}") as run:
        mlflow.log_params({
            "n_data": args.n_data,
            "hidden": args.hidden,
            "lr": args.lr,
            "epochs": args.epochs,
            "val_frac": args.val_frac,
            "seed": args.seed,
        })
        
        for epoch, (tr, va) in enumerate(zip(history["train"], history["val"])):
            mlflow.log_metric("train_mse", tr, step=epoch)
            mlflow.log_metric("val_mse", va, step=epoch)
        mlflow.log_metric("final_train_mse", history["train"][-1])
        mlflow.log_metric("final_val_mse", history["val"][-1])
        mlflow.log_metric("best_val_mse", min(history["train"]))

    # ------------------------------------------------------------------
    # TODO 4 — Log the trained model AND register it in one call.
    # WHY: this is the bridge from "tracked run" to "registered model".
    # `registered_model_name=MODEL_NAME` causes MLflow to create the
    # registry entry on first call and add a new VERSION on every call
    # after.
    #
    # Best practice: pass `signature` (via `infer_signature`) and
    # `input_example`. The signature is the model's I/O CONTRACT — once it
    # lives in the registry, downstream code can introspect "what shape
    # does this expect?" without loading the model. Skip this and you've
    # registered a black box.
    #
    # PATTERN (still INSIDE the `with mlflow.start_run()` block above):
    #     x_example = X_tr[:5]                              # numpy (5, N_FEATURES)
    #     with torch.no_grad():
    #         y_example = model(
    #             torch.tensor(x_example, dtype=torch.float32).to(device)
    #         ).cpu().numpy()
    #     signature = infer_signature(x_example, y_example)
    #
    #     mlflow.pytorch.log_model(
    #         pytorch_model=model,
    #         artifact_path="model",
    #         registered_model_name=MODEL_NAME,
    #         signature=signature,
    #         input_example=x_example,
    #     )
    # ------------------------------------------------------------------
    raise NotImplementedError(
        "TODO 4: mlflow.pytorch.log_model with signature + registered_model_name"
    )

    # ------------------------------------------------------------------
    # TODO 5 — Tag the new version with @candidate (NOT @production).
    # WHY: aliases are the modern MLflow way to point at "the model to
    # serve". We deliberately do NOT auto-promote — a human (or a
    # downstream gate / approval) decides when a candidate becomes
    # production. Otherwise a flaky training run silently goes live.
    #
    # NB: MLflow's public API types `version` as `str` even though it's
    # a monotonic integer. Cast at the boundary.
    #
    # HINT: AFTER `with mlflow.start_run()` closes, find the version that
    # was just created and alias it.
    #
    #     client = MlflowClient()
    #     latest_version = max(
    #         int(mv.version)
    #         for mv in client.search_model_versions(f"name='{MODEL_NAME}'")
    #     )
    #     client.set_registered_model_alias(
    #         name=MODEL_NAME,
    #         alias=CANDIDATE_ALIAS,
    #         version=str(latest_version),
    #     )
    #     print(f"Registered {MODEL_NAME} v{latest_version} @{CANDIDATE_ALIAS}")
    # ------------------------------------------------------------------
    raise NotImplementedError("TODO 5: tag latest version as @candidate")


if __name__ == "__main__":
    main()
