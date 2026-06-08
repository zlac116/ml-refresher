# ml-revision

Self-paced ML / quant finance / API engineering revision. Hands-on
notebooks + cheatsheets + capstones, organised as a learning path.

## Repository layout

| Top-level | What's in it | Start here |
|---|---|---|
| [`fundamentals/`](fundamentals/) | STEM refresher (maths / physics / chemistry) | when a concept's intuition is missing |
| [`sql/`](sql/) | PostgreSQL cheatsheet + capstone | data querying foundation |
| [`ml/`](ml/) | Core ML — regression → classification → time series → neural networks | the main ML track |
| [`quant_finance/`](quant_finance/) | Options / risk / fixed income / portfolio / volatility / stoch calc + capstones | quant track (heaviest module) |
| [`api_engineering/`](api_engineering/) | Production FastAPI cheatsheet + capstone | shipping models |
| [`llm_pipeline/`](llm_pipeline/) | LangChain + LangGraph cheatsheet + 4 LLM capstones | LLM / agentic systems |
| [`toolkit/`](toolkit/) | Cross-cutting playbooks (eda, ML methodology, MLflow) | reach for from any module |
| [`data/`](data/) | Shared datasets (crypto OHLCV, fetchers) | used by multiple modules |

## Layout conventions

Every numbered module follows the same shape:

```
<module>/
├── README.md                         ← module overview + reading order
├── <module>_cheatsheet.md            ← code-first reference
├── tutorial.ipynb / tutorial/        ← guided walkthrough (where applicable)
└── capstones/                        ← plural, one subdir per capstone
    └── <NN>_<name>/
        ├── README.md                 ← spec + how to run
        ├── LESSONS.md                ← patterns to internalise after
        └── <source files>
```

Numeric prefix (`01_`, `02_`, …) signals **recommended learning order**.

## Where to start

- **New to the repo?** → Read this file, then [`STUDY_GUIDE.md`](STUDY_GUIDE.md) for the 4-week plan.
- **Pure quant interview prep?** → Follow `STUDY_GUIDE.md` directly — `quant_finance/` first.
- **ML engineer prep?** → `ml/` → `api_engineering/` → `quant_finance/capstones/04_lmm_nn_surrogate/` (the cross-cutting capstone).
- **LLM / agent dev?** → `llm_pipeline/llm_cheatsheet.md` then the four capstones in order.

## Setup

Per-capstone isolation: each capstone is a self-contained [uv](https://docs.astral.sh/uv/) project. No top-level virtualenv needed.

```bash
# To work on a specific capstone:
cd <module>/capstones/<NN>_<name>/
uv sync                # installs the capstone's pinned deps + creates .venv
uv run python <file.py>
uv run jupyter lab     # if the capstone has notebooks
```

For pure notebook modules (`fundamentals/`, the `ml/` subjects, `quant_finance/` notebook subjects), each subject folder has its own `pyproject.toml` + `uv.lock`. Same pattern.

## Daily workflow

```bash
cd <module-or-capstone>/
uv sync                     # pull deps (cached after first time)
uv run jupyter lab          # for notebooks
uv run pytest               # if tests
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db   # if MLflow-tracked
```

## Maintenance

- All ML / LLM runtime artifacts (`mlflow.db`, `mlruns/`, `*.pt`, `.langgraph_api/`, `chroma.sqlite3`, etc.) are gitignored — regenerable on next run.
- See [`.gitignore`](.gitignore) for the full list.

### Pre-commit hooks (recommended one-time install)

Defence in depth against accidentally committing large files / secrets / merge conflicts:

```bash
uv tool install pre-commit
pre-commit install         # registers the hooks in .git/hooks/
```

Once installed, every `git commit` runs the checks in [`.pre-commit-config.yaml`](.pre-commit-config.yaml) — file-size cap (10 MB), private-key detector, merge-conflict markers, YAML/TOML/JSON validity, Python lint/format via ruff. Bypass for a single commit (rarely) with `--no-verify`.

## Related references

- [`toolkit/eda_decisions.md`](toolkit/eda_decisions.md) — model selection rules of thumb.
- [`toolkit/ml_project_methodology.md`](toolkit/ml_project_methodology.md) — universal ML pipeline phases.
- [`toolkit/mlflow_cheatsheet.md`](toolkit/mlflow_cheatsheet.md) — tracking + registry + serving.
- Domain cheatsheets live alongside their module — see the table at the top.

## Conventions

- **Modern Python**: `>=3.12`, type hints throughout, `X | None` over `Optional[X]`.
- **uv exclusively** for Python package management (not pip / poetry / conda).
- **Latest stable APIs**: see each cheatsheet's "Modern conventions worth knowing upfront" section.
- **Cheatsheets** lead with a §0 "general pattern" section before drilling into details.
