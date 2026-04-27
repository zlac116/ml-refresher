# ml-revision

Hands-on Jupyter notebooks for self-paced ML revision. Three layers:

**ML pipelines** (end-to-end workflow on hourly crypto OHLCV — BTC, ETH, SOL, BNB):

| Notebook | Task |
|---|---|
| `classification/classification.ipynb` | Direction classification — sign of BTC's next 4h return |
| `regression/regression.ipynb`         | Volatility regression — BTC 24h-ahead realised vol |
| `time-series/time_series.ipynb`       | Time-series forecasting — BTC hourly log returns (h=1..24) |

Each is structured as 10–17 **pipeline stages**. Every stage has the same shape:
*Why we're here* → *30-second concept* → *runnable failure-mode demo* → *decisions* →
*2 focused exercises* → *recap*.

**Toolkit** (`toolkit/`) — task-indexed cheatsheets for the libraries you'll need:
`pandas`, `numpy_scipy`, `plotting`, `sklearn`, `statsmodels`, `gradient_boosting`,
`optuna`, `shap`, `deployment`. Look up the idiom by task ("how do I do X?").
See `toolkit/README.md`.

**Fundamentals** (`fundamentals/`) — KS3+GCSE refreshers for `mathematics`, `chemistry`,
`physics`. Each topic has a short concept intro, a worked example, and exercises
with hidden solutions and a `check()` helper that prints ✅/❌.

**LLM pipeline** (`llm-pipeline/`) — task-indexed LangChain + LangGraph cheatsheet
mirroring the official docs structure. See `llm-pipeline/README.md` for the extra
dependencies.

## Setup

Requires Python 3.12.

```bash
make venv      # creates .venv and installs requirements.txt
```

Override the interpreter with `make venv PYTHON=python3.12.4` if needed.

## Run

```bash
make lab         # launch Jupyter Lab
make notebook    # launch classic Jupyter Notebook
```

Then open whichever notebook you want to work through.

## Maintenance

```bash
make clean       # remove caches, checkpoints, .bak files
make help        # list every target
```
