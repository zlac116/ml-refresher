"""
PROJECT — Deployment: Wrap a Model as a Stateless predict_one Service
=======================================================================

OBJECTIVE
    Take a trained sklearn/LightGBM model and turn it into a production-quality
    prediction interface:

      1. Save a self-contained bundle (model + feature_names + metadata).
      2. predict_one(row) — stateless, reorders features by feature_names.
      3. Pydantic schema generated dynamically from feature_names.
      4. Property test for order invariance (reorder row, same prediction).
      5. Structured JSON logging with a per-call correlation ID.

ESTIMATED TIME
    25 min

TOPICS
    joblib.dump for the bundle (model + metadata in one .pkl)
    pd.DataFrame([row])[feature_names] — the critical reorder
    pydantic.create_model for runtime schemas
    Property test: same data, scrambled key order, identical prediction
    logging.Formatter returning JSON + uuid4 for correlation IDs

REQUIRED PACKAGES
    joblib, pydantic, scikit-learn (run `uv add joblib pydantic scikit-learn`)

EXPECTED OUTPUT
    bundle saved to:        /tmp/bundle.pkl
    predict_one (in-order): a float between 0 and 1
    order invariance:       True (same value within 1e-9)
    schema validation:      OK for good payload, ValidationError for missing field
    log line:               valid JSON containing request_id, features, output
"""
import json
import logging
import uuid
import joblib
import numpy as np
import pandas as pd
from pydantic import create_model, ValidationError
from sklearn.linear_model import LogisticRegression


BUNDLE_PATH = "/tmp/bundle.pkl"
FEATURE_NAMES = ["x_0", "x_1", "x_2", "x_3"]


def _train_demo_model() -> tuple[LogisticRegression, list[str]]:
    """Train a tiny LogReg on synthetic 4-feature binary data; return (model, feature_names)."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (200, 4))
    y = ((X[:, 0] + 0.5 * X[:, 1] - X[:, 3]) > 0).astype(int)
    model = LogisticRegression().fit(X, y)
    return model, FEATURE_NAMES


# ── TASK 1 ─────────────────────────────────────────────────────────────────
def save_bundle(model, feature_names: list[str], path: str = BUNDLE_PATH) -> None:
    """Save a dict containing:
        'model'           : the trained estimator
        'feature_names'   : list[str]
        'model_version'   : str like 'v1.0.0'
        'trained_through' : an ISO date string (use today)
    via joblib.dump(bundle, path).
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 2 ─────────────────────────────────────────────────────────────────
def predict_one(row: dict, path: str = BUNDLE_PATH) -> dict:
    """Load the bundle and predict ONE row.

    Must reorder by feature_names BEFORE prediction (the critical line):
        df = pd.DataFrame([row])[bundle['feature_names']]

    Return a dict with keys:
        'prob_up'         : float (predict_proba[:, 1][0])
        'model_version'   : echoed from bundle
        'trained_through' : echoed from bundle
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 3 ─────────────────────────────────────────────────────────────────
def make_pydantic_schema(feature_names: list[str]):
    """Build a Pydantic model at runtime:
        create_model('PredictRequest', **{f: (float, ...) for f in feature_names})

    Returns the model CLASS (not an instance).
    """
    # TODO: implement
    raise NotImplementedError


# ── TASK 4 ─────────────────────────────────────────────────────────────────
def test_order_invariance(row: dict) -> bool:
    """Call predict_one with the row keys in two different orders.
    Returns True if both predictions agree within 1e-9.
    """
    # TODO: implement
    #   - prob_in_order  = predict_one(row)['prob_up']
    #   - reversed_row   = dict(reversed(list(row.items())))
    #   - prob_reversed  = predict_one(reversed_row)['prob_up']
    #   - return abs(prob_in_order - prob_reversed) < 1e-9
    raise NotImplementedError


# ── TASK 5 ─────────────────────────────────────────────────────────────────
def predict_one_with_logging(row: dict, logger: logging.Logger) -> dict:
    """Call predict_one, then emit a structured JSON log with:
        request_id  =  uuid.uuid4().hex
        input       =  row
        output      =  prediction dict
    Returns the prediction dict.
    """
    # TODO: implement
    raise NotImplementedError


# ── GRADING ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os, datetime as dt

    # Setup: train a demo model and save bundle
    model, feats = _train_demo_model()
    save_bundle(model, feats)
    assert os.path.exists(BUNDLE_PATH)

    bundle = joblib.load(BUNDLE_PATH)
    assert set(bundle.keys()) == {"model", "feature_names", "model_version", "trained_through"}
    assert bundle["feature_names"] == feats

    # predict_one
    row = {"x_0": 0.5, "x_1": -0.3, "x_2": 1.2, "x_3": -0.8}
    result = predict_one(row)
    assert {"prob_up", "model_version", "trained_through"}.issubset(result)
    assert 0.0 <= result["prob_up"] <= 1.0

    # Pydantic schema
    Schema = make_pydantic_schema(feats)
    # Good payload validates
    Schema(**row)
    # Bad payload (missing x_3) raises
    bad = {k: v for k, v in row.items() if k != "x_3"}
    try:
        Schema(**bad)
        raise AssertionError("Schema should have rejected missing x_3")
    except ValidationError:
        pass

    # Order invariance
    assert test_order_invariance(row) is True

    # Logging
    logger = logging.getLogger("deployment_test")
    logger.handlers = []
    logger.setLevel(logging.INFO)
    captured = []
    class _CaptureHandler(logging.Handler):
        def emit(self, record):
            captured.append(self.format(record))
    h = _CaptureHandler()
    h.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(h)

    out = predict_one_with_logging(row, logger)
    assert "prob_up" in out
    assert len(captured) >= 1
    parsed = json.loads(captured[-1])
    assert "request_id" in parsed and "input" in parsed and "output" in parsed

    print(f"bundle saved to:        {BUNDLE_PATH}")
    print(f"predict_one (in-order): prob_up = {result['prob_up']:.6f}")
    print(f"order invariance:       True")
    print(f"schema validation:      OK for good payload, raises for missing field")
    print(f"log line:               {captured[-1][:120]}...")
    print("\n✓ All checks passed.")
