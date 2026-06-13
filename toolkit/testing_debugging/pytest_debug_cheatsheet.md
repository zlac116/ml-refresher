# pytest debugging cheatsheet

How to debug failing tests with minimum friction. Default to `--pdb`; only reach for `breakpoint()` when you need to pause at a specific line.

---

## §0 General pattern — the debug ladder

When a test fails or hangs, climb this ladder in order. Stop at the first tier that solves your problem.

```
TIER 1 — pytest --pdb           ← default; no code edits
TIER 2 — breakpoint() + -s      ← pause at a specific line
TIER 3 — VSCode debugger        ← clickable breakpoints + variables panel
TIER 4 — pytest --trace         ← drop into PDB before the test body runs
```

**Canonical first move** for any failing test:

```bash
pytest tests/test_x.py::test_y --pdb -x
```

- `--pdb` drops you into PDB at the failure point automatically
- `-x` stops on the first failure (don't waste time running the rest)
- No code edits, no risk of committing a `breakpoint()` by accident

**Key distinction:** `breakpoint()` requires `pytest -s` to disable output capture, or it hangs silently. `--pdb` does not. Always try `--pdb` first.

**Variations:** see §3 for VSCode, §5 for PDB commands, §6 for QoL upgrades.

---

## §1 Tier 1 — `pytest --pdb` (default first move)

```bash
# Failure drops you into PDB at the failing assertion / exception
pytest tests/test_e2e.py::test_price_endpoint_returns_iv --pdb

# Stop on first failure (recommended)
pytest --pdb -x

# Whole suite, drop in on first failure
pytest --pdb -x -v
```

Why this is the default:
- Zero code edits — no `breakpoint()` to clean up
- Pauses exactly at the failure point, not where you guessed
- Catches surprises (exceptions you didn't expect)

---

## §2 Tier 2 — `breakpoint()` + `pytest -s`

Use when you know exactly where you want to pause (mid-fixture, mid-loop) and the failure is upstream of an assertion.

```python
@pytest.fixture
def trained_registry(tmp_path_factory):
    tracking_dir = tmp_path_factory.mktemp("mlruns")
    breakpoint()                     # ← pause here
    mlflow.set_tracking_uri(...)
```

```bash
# -s DISABLES pytest output capture. Without it, breakpoint() hangs silently.
pytest tests/test_e2e.py -s
```

| Flag | Why |
|---|---|
| `-s` | **Required.** Disables `capsys` so PDB can read stdin |
| `--no-header` | Cleaner output around the PDB prompt |
| `-x` | Stop on first failure |

**Anti-pattern**: committing `breakpoint()` calls. Treat them as working-tree-only artifacts. Defense: set `PYTHONBREAKPOINT=0` in CI so stray ones become no-ops.

---

## §3 Tier 3 — VSCode debugger

For clickable breakpoints + variables panel + watches + step-into-libraries.

### One-off (no launch.json needed)

1. Open the test file
2. Look at the **gutter beside each `def test_...` function** — there are play and bug icons
3. Click the **bug icon** → debugs that single test with sensible defaults
4. Set breakpoints by clicking in the gutter beside any line

Or right-click any test in the **Testing** sidebar (beaker icon) → **Debug Test**.

### Reusable launch.json snippet

Drop into `.vscode/launch.json` so `F5` debugs the current test file:

```json
{
  "name": "pytest (current file)",
  "type": "debugpy",
  "request": "launch",
  "module": "pytest",
  "args": ["${file}", "-v", "-s"],
  "cwd": "${fileDirname}/..",
  "python": "${workspaceFolder}/<path-to>/.venv/bin/python",
  "justMyCode": false,
  "console": "integratedTerminal"
}
```

| Field | Why |
|---|---|
| `module: "pytest"` | Launches `python -m pytest` |
| `args: ["${file}", "-v", "-s"]` | Current file, verbose, no capture |
| `cwd` | Parent dir of the test file (usually the package root) |
| `python` | **Verify this path exists** — pin to the project's `.venv` |
| `justMyCode: false` | Step INTO library code (pytest, fastapi, mlflow) |
| `console: integratedTerminal` | See test output as it streams |

### How to add a config without typing JSON

| Path | How |
|---|---|
| **Empty .vscode/** | Run & Debug panel → "create a launch.json file" → pick Python Debugger → pick template |
| **Existing launch.json** | Click **"Add Configuration..."** button (bottom-right of editor) → pick template |
| **Command palette** | `Ctrl+Shift+P` → `Debug: Add Configuration...` |
| **Pytest-only** | Don't bother — use the gutter bug icon (no JSON needed) |

---

## §4 Tier 4 — `pytest --trace`

Drops into PDB at the **start** of the test body, before any line runs.

```bash
pytest tests/test_e2e.py::test_X --trace
```

Use when:
- The test hangs and you can't tell where
- You want to step through the entire test interactively
- A fixture errors and you need to inspect before assertions

---

## §5 PDB reference card

Once you're at the `(Pdb)` prompt:

| Command | Action |
|---|---|
| `n` | Step over one line |
| `s` | Step **into** a function call |
| `c` | Continue until next breakpoint / end |
| `r` | Run until current function returns |
| `l` | List source around current line |
| `ll` | List the whole current function |
| `p <expr>` | Print value |
| `pp <expr>` | Pretty-print |
| `w` | Show call stack |
| `u` / `d` | Move up / down one frame in the stack |
| `a` | Show args of current frame |
| `b <file>:<line>` | Set breakpoint at a location |
| `cl <n>` | Clear breakpoint #n |
| `interact` | Drop into a real Python REPL with all locals available |
| `q` | Quit (test marked errored) |
| `h` | Help on any command |

**Underused gem:** `interact` opens a full Python REPL where every local variable is in scope. Useful for poking at complex objects (`type(x).__mro__`, `dir(x)`, `vars(x)`).

---

## §6 Quality-of-life upgrades

### Use IPython's debugger (syntax highlighting + tab completion)

```bash
uv add --dev ipdb

# Use ipdb everywhere
export PYTHONBREAKPOINT=ipdb.set_trace
pytest -s                    # breakpoint() now uses ipdb

# Or per-invocation
pytest --pdbcls=IPython.terminal.debugger:TerminalPdb --pdb
```

### Make stray `breakpoint()` calls safe in CI

```bash
PYTHONBREAKPOINT=0 pytest    # ignores all breakpoint() calls
```

Add this to your CI env so a forgotten `breakpoint()` doesn't hang the pipeline.

### Show locals in test failure tracebacks (no debugger needed)

```bash
pytest --showlocals          # or -l
```

Sometimes you don't need a debugger — just locals at the point of failure.

### Tee output to file while keeping pytest live

```bash
pytest -s | tee /tmp/test.log
```

### Run only the last-failed tests

```bash
pytest --lf                  # last failed
pytest --ff                  # failed first, then the rest
```

---

## §7 Anti-patterns to avoid

| Anti-pattern | Why it bites |
|---|---|
| `print()` everywhere | Output swallowed by capture; slower than PDB; easy to leave behind |
| `breakpoint()` without `-s` | Hangs silently waiting for stdin pytest is capturing |
| Committing `breakpoint()` | Will hang someone else's pytest run; treat as a working-tree artifact only |
| `pytest --pdb` with `-n auto` (xdist) | PDB can't multiplex across worker processes; disable parallelism when debugging |
| Reaching for VSCode for every debug | Often slower than `pytest --pdb` for a one-off failure |
| Writing assertions without messages | `assert x == 1, f"got {x}"` saves you from re-running with PDB just to see the value |

---

## §8 Quick recipes — copy-paste

```bash
# 1. Failing test, no idea why — DEFAULT FIRST MOVE
pytest tests/test_e2e.py::test_X --pdb -x

# 2. Test passes locally, fails in CI — see locals + verbose output
pytest tests/test_e2e.py::test_X -v --showlocals --tb=long

# 3. Test hangs — drop in at start
pytest tests/test_e2e.py::test_X --trace

# 4. Re-run only last-failed
pytest --lf --pdb -x

# 5. Pause mid-fixture (you placed breakpoint() in fixture)
pytest tests/test_e2e.py -s

# 6. Whole suite, IPython debugger on failure
PYTHONBREAKPOINT=ipdb.set_trace pytest --pdb --pdbcls=IPython.terminal.debugger:TerminalPdb -x

# 7. Debug via VSCode — open test file, click gutter bug icon (no flags needed)
```

---

## §9 Worked example — debugging a fixture failure

Scenario: `client` fixture errors during `TestClient.__enter__` (lifespan failure).

```bash
# Step 1 — see what failed
pytest tests/test_e2e.py::test_price_endpoint_returns_iv -x --tb=short

# Step 2 — drop in at the failure
pytest tests/test_e2e.py::test_price_endpoint_returns_iv --pdb -x
```

At the `(Pdb)` prompt:

```
(Pdb) w                              # see the stack
(Pdb) u                              # move up to the test/fixture frame
(Pdb) p trained_registry             # check the URI passed in
(Pdb) p os.environ.get("MLFLOW_TRACKING_URI")
(Pdb) p get_settings()               # see what Settings ended up with
(Pdb) interact                       # drop into REPL for deeper poking
>>> from mlflow import MlflowClient
>>> c = MlflowClient(tracking_uri=trained_registry)
>>> [m.aliases for m in c.search_model_versions("name='lmm-surrogate'")]
>>> exit()                           # back to PDB
(Pdb) q                              # done
```

You now know exactly which alias / URI / env var is wrong without scattering print statements.

---

## TL;DR

1. **Default**: `pytest tests/test_x.py::test_y --pdb -x`
2. **Mid-fixture pause**: `breakpoint()` + `pytest -s`
3. **IDE breakpoints**: click the gutter bug icon next to a test function
4. **Never commit `breakpoint()`** — set `PYTHONBREAKPOINT=0` in CI as defense
5. **Use `interact`** in PDB to get a full REPL with locals in scope
