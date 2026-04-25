# ml-revision

Hands-on Jupyter notebooks for self-paced revision in two layers:

**ML revision** (end-to-end ML workflow on hourly crypto OHLCV — BTC, ETH, SOL, BNB):

| Notebook | Task | Target |
|---|---|---|
| `classification/classification.ipynb` | Direction classification | Sign of BTC's next 4h return |
| `regression/regression.ipynb` | Volatility regression | BTC 24h-ahead realised vol |
| `time-series/time_series.ipynb` | Time-series forecasting | BTC hourly log returns (h=1..24) |

**Fundamentals** (KS3 + GCSE):

| Notebook | Topics |
|---|---|
| `fundamentals/mathematics.ipynb` | 18 sections — arithmetic through to quadratics, trigonometry, vectors |
| `fundamentals/chemistry.ipynb`   | 12 sections — atoms, the mole, pH, energy changes, organic chemistry |
| `fundamentals/physics.ipynb`     | 12 sections — kinematics, forces, waves, circuits, radioactivity |

Every section has the same teaching shape:

1. **Concept intro** — short prose explanation.
2. **Worked example** — a runnable cell demonstrating the section's techniques. Run it, modify it, learn the pattern.
3. **Exercises** — each is *prompt → empty answer cell → hidden solution*. The fundamentals notebooks add a `check()` helper that prints ✅ / ❌ next to your answer.

## Setup

Requires Python 3.12.

```bash
make venv      # create .venv and install requirements.txt (no-op if .venv exists)
```

Override the interpreter with `make venv PYTHON=python3.12.4` if needed.

## Run

```bash
make lab         # launch Jupyter Lab
make notebook    # launch classic Jupyter Notebook
```

Then open any of the three notebooks and work through the exercises.

## Regenerate notebooks

```bash
make validate          # structural check — both ML and fundamentals (~5s)
make classification    # rebuild one ML notebook (5–15 min)
make all               # rebuild all three ML notebooks (30–60 min)
make fundamentals      # rebuild the fundamentals notebooks (~10 s)
make clean             # caches, checkpoints, .bak files
make help              # list every target
```

ML-notebook regeneration executes every code cell in a fresh kernel, so it's slow.
The fundamentals notebooks are quick to rebuild (no heavy training) — `make fundamentals`
runs in seconds.
