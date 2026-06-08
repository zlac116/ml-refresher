# Capstone — Regression NN with Early Stopping (California Housing)

Direct sibling of the tabular classifier — same skeleton, two new things:
regression target + **early stopping with best-weight restoration**.

**Time budget**: ~2 hours.

## What you build

- Regression on California Housing (8 features → median house value).
- Scale **both X and y** (separate scalers — y_scaler used for inverse-transform at eval).
- MLP with 1 output, **no output activation**.
- Training loop with early stopping (`copy.deepcopy(state_dict)` snapshot + restore).
- Save model + x_scaler + y_scaler.

**Target**: ~0.79+ test R² (~$53k RMSE).

## Files

- `train.py` — skeleton with `# TODO` markers.
- `LESSONS.md` — canonical patterns, including the three-piece-of-state early-stopping bookkeeping.
- `cal_housing_model.pt`, `cal_housing_model_x_scaler.joblib`, `cal_housing_model_y_scaler.joblib` — artifacts (gitignored).

## Run

```bash
uv run python train.py
uv run python train.py --epochs 300 --patience 20 --hidden 128 64
```

## Conceptual focus

- **Early stopping** = three pieces of state (`best_val`, `best_state`, `patience_used`) wrapping an unchanged train/val loop.
- **`copy.deepcopy(state_dict())`** is non-negotiable — without it, your "snapshot" shares references with the live tensors and mutates with every `opt.step()`.
- `<` not `>` direction — inverting the comparison snapshots the WORST weights.
- Inverse-transform predictions AND targets before metrics — RMSE in scaled space is meaningless.

See [`LESSONS.md`](LESSONS.md) for the full pattern catalogue.

## Related

- [`../01_tabular_classifier_wine/`](../01_tabular_classifier_wine/) — the classification sibling, identical bones.
- [`../../neural_networks_cheatsheet.md`](../../neural_networks_cheatsheet.md) — the PyTorch reference.
- [`../../../../quant_finance/capstones/04_lmm_nn_surrogate/`](../../../../quant_finance/capstones/04_lmm_nn_surrogate/) — surrogate-replaces-slow-function pattern using the same training skeleton.
