# LMM NN Surrogate — Lessons

Reflection on what *this specific project* taught me, beyond the generic
patterns. For the reusable playbook see
[`toolkit/ml_project_methodology.md`](../../../toolkit/ml_project_methodology.md).

---

## 1. Surrogate-replaces-slow-function is the headline pattern

Slow expensive function (MC pricer) → fast NN approximation, used as a
drop-in inside a tight optimisation loop. Recognise this whenever I see:
*"calibration is too slow"*, *"the inner loop is the bottleneck"*, or
*"we need to call this thousands of times"*. Same trick applies to
Bayesian optimisation, hyperparameter search, simulation acceleration.

## 2. Two-residual verification — the real conceptual breakthrough

The verify table has THREE IVs (`market`, `NN(θ*)`, `MC(θ*)`) and TWO
independent residuals:

- `calib = market − MC(θ*)` → "does the model fit market?"
- `surrogate = NN(θ*) − MC(θ*)` → "did the NN tell the truth at θ\*?"

**Anti-pattern** (which I wrote on the first pass): `NN(θ*) − market`.
That's the optimiser's residual — ≈ 0 by construction, **zero audit value**.
Burn this in: never verify by comparing the optimiser's output to its
target. Use a different code path for the truth.

## 3. Activation choice has downstream consequences

`SiLU` not `ReLU` because `scipy.least_squares` computes Jacobians by
finite differences. ReLU's kink at 0 gives noisy gradients. The activation
was picked for the consumer two layers up the stack, not "to be modern".
General lesson: **know the consumer of every function I write.**

## 4. Bounds are a single source of truth

`LMM_PARAM_LO/HI` are the *sampling* range during training, the *bounds*
passed to `least_squares`, and the *validators* on the API schema. One
constant, three roles. Change it once → consistent everywhere. If I'd
duplicated these as literals in each function, divergence would have been
inevitable and silent.

## 5. The simulation-truth setup

In `main`, I hard-coded `true_params = [0.18, 0.40, 0.015, 0.30]` and
derived `market_ivs` from them. This lets me answer the only question
that matters: *"did calibration recover the truth?"* In prod the truth
comes from yesterday's calibration vs today's market; in learning, fake
it from known params. Same shape of test, different source of "truth".

## 6. Tensor-shape silent bugs

`.squeeze(-1)` to collapse `(N, 1)` → `(N,)`. Without it, `(N, 1) - (N,)`
broadcasts to `(N, N)` and trains on nonsense — no error, just garbage.
This is the #1 silent NN bug. Always ask: *"what shape does the loss
compare?"*

## 7. From timestamped dirs to MLflow is the SAME pattern

The parent saves to `out/<timestamp>/{surrogate.pt, run.json, …}`. The
API extension uses MLflow registry with aliases. **Identical idea** — an
immutable artifact per run, indexed for lookup. MLflow is just the
productionised version: registry replaces filesystem, aliases replace
"the latest timestamp".

## 8. Bugs I hit (so I don't repeat them)

- `%Y%m%dT%H%M%S` not `%Y%m%d%T%H%M%S` — `%T` is `HH:MM:SS`, embeds the
  time twice with colons.
- Tuple unpack order matters: `market_instruments` is `(T, K, F)` —
  unpacking `(_T, _F, _K)` silently swaps K and F in the display.
- Pydantic `raise NotImplementedError` inside class bodies executes at
  import time and breaks the whole module. Put TODOs as comments, not
  raises, inside classes.

---

For the generic phase-by-phase playbook (bootstrap → data → split → model
→ train → infer → calibrate → verify → save, with patterns and
anti-patterns), see
[`toolkit/ml_project_methodology.md`](../../../toolkit/ml_project_methodology.md).
