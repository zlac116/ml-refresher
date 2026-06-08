# Capstone — Tabular Classifier (Wine)

End-to-end multiclass NN classifier on the Wine dataset (3 classes, 13
features). The first PyTorch capstone — focuses on the canonical recipe
(seed → load → split → scale → tensors → model → train → evaluate → save).

**Time budget**: ~2 hours.

## What you build

- Stratified train/val/test split.
- StandardScaler **fit on train only** (leakage rule).
- Small MLP with `n_classes` logits, no output activation.
- Training loop with running loss + per-epoch val pass.
- Save model state_dict + scaler (both required for inference).

**Target**: ~0.95+ test accuracy.

## Files

- `train.py` — skeleton with `# TODO` markers (one per recipe step).
- `LESSONS.md` — canonical code patterns to internalise after the build.
- `wine_model.pt`, `wine_model_scaler.joblib` — artifacts (gitignored, regen on run).

## Run

```bash
uv run python train.py
uv run python train.py --epochs 150 --lr 5e-4 --hidden 64 32
```

## Conceptual focus

- CrossEntropyLoss eats raw logits — no softmax in the model.
- Labels are `int64`, shape `(N,)`, **not** one-hot.
- Save BOTH model and scaler; the model alone is useless on raw inputs.

See [`LESSONS.md`](LESSONS.md) for the full pattern catalogue.

## Related

- [`../02_regression_california_housing/`](../02_regression_california_housing/) — the regression sibling, adds early stopping.
- [`../../neural_networks_cheatsheet.md`](../../neural_networks_cheatsheet.md) — the underlying PyTorch reference.
