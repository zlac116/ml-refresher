# ml-revision

Three hands-on Jupyter notebooks revising the end-to-end ML workflow on hourly
crypto OHLCV data (BTC, ETH, SOL, BNB):

| Notebook | Task | Target |
|---|---|---|
| `classification/classification.ipynb` | Direction classification | Sign of BTC's next 4h return |
| `regression/regression.ipynb` | Volatility regression | BTC 24h-ahead realised vol |
| `time-series/time_series.ipynb` | Time-series forecasting | BTC hourly log returns (h=1..24) |

Each exercise is laid out as **prompt → empty answer cell → expected output →
hidden solution**, with a "Before you start — techniques you'll use" hint block
at the top of every section.

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
make validate          # fast structural check (~1s)
make classification    # rebuild one notebook (5–15 min)
make all               # rebuild all three (30–60 min)
make clean             # caches, checkpoints, .bak files
make help              # list every target
```

Regeneration executes every code cell in a fresh kernel, so it's slow —
`make validate` is what you want for day-to-day checks.
