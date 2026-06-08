# Neural Networks — Best Practices, End-to-End (PyTorch)

A from-scratch, fully-worked tour of how to run a neural-network **experiment**
correctly for both **regression** and **classification**, using PyTorch and real
scikit-learn datasets.

## Contents

| File | What it is |
|------|------------|
| `neural_networks.ipynb` | The main notebook. Best-practices + framework discussion, then two end-to-end experiments (California Housing regression, Breast Cancer classification), every section explained. |
| `capstone/train.py` | **Your exercise.** A skeleton (function signatures + docstrings + TODOs) to implement: a CLI that applies the recipe to a new task (multiclass Wine classification). Extension challenges in the docstring. |
| `pyproject.toml` / `uv.lock` | uv-managed environment (CPU-only PyTorch). |

## Setup (uv)

This project uses [uv](https://docs.astral.sh/uv/) as the package manager.

```bash
cd ml/neural_networks
uv sync                # creates .venv and installs all deps (CPU torch)
```

That's it — `uv sync` reads `pyproject.toml` + `uv.lock` and builds the
environment. (Torch is pinned to CPU wheels to avoid pulling ~2 GB of CUDA
libraries; remove the `[tool.uv.sources]`/`[[tool.uv.index]]` blocks in
`pyproject.toml` if you want GPU support.)

## Run the notebook

```bash
uv run jupyter lab neural_networks.ipynb     # or: uv run jupyter notebook ...
```

Execute cells top to bottom — each builds on the previous.

## Capstone (your exercise)

`capstone/train.py` is a **skeleton** (~2-hour budget) — each
function raises `NotImplementedError` with a docstring + TODOs describing what to
build. Work through it yourself (multiclass Wine classifier), then run:

```bash
uv run python capstone/train.py
uv run python capstone/train.py --epochs 150 --lr 5e-4 --hidden 64 32
```

When complete it should train, evaluate on a held-out test set, print accuracy +
macro-F1, and save the model weights (`.pt`) and fitted scaler (`.joblib`).
Target ~0.95+ test accuracy. The docstring lists extension challenges once the
base version works. Ask me to review your implementation when you're ready.

## The recipe you're learning

1. **Setup** — seed everything, pick a device.
2. **Split** — train / val / test (stratify for classification).
3. **Scale** — `StandardScaler().fit(X_train)`, transform all sets.
4. **Tensors + DataLoader** — float32 tensors, mini-batches, shuffle train only.
5. **Model** — MLP; output size & final activation set by the task.
6. **Loss + optimizer** — MSE vs BCE/CrossEntropy; Adam @ `lr=1e-3` to start.
7. **Train loop** — `zero_grad → forward → loss → backward → step`.
8. **Evaluate on test once** — task-appropriate metrics, original units.
9. **Diagnose** — loss curves + a prediction plot.
