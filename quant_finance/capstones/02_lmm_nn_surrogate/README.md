# Capstone — LMM NN Surrogate for Calibration

> **YOUR exercise.** Skeleton with TODOs; no data shipped. You implement the
> data generator, the NN surrogate, the calibration loop, and the verification
> step. When done, ask me to **review**.

Build the **production NN-surrogate calibration workflow** for an LMM-priced rates
book — but with a mock pricer standing in for the slow MC so it fits in 3 hours.
The workflow is the real thing; only the labels are toy. From the NN's
perspective there is no difference.

The conceptual write-up + a runnable single-file demo live next to this folder:
- `../notes.md` — the cheatsheet (read sections 1–4 before starting)
- `../example.py` — an 80-line illustrative demo

This capstone elevates that demo into a properly structured pipeline with
clean function boundaries, train/val split, saved artifacts, and an instrument-level
verification table. **Read the demo for reference, but write your own version.**

⏱️ **Time budget: ~3 hours (hard cap).** Stretch items are clearly marked.

---

## The workflow you're implementing

```
                      slow MC pricer (here: mock)
                                │
                                ▼
        ┌───────────────────────────────────────────────┐
        │  Generate (LMM params + instrument) → IV rows │   ← step 1
        └───────────────────────────────────────────────┘
                                │
                                ▼
        ┌───────────────────────────────────────────────┐
        │  Train MLP surrogate: features → Black IV     │   ← step 2
        └───────────────────────────────────────────────┘
                                │
                                ▼
   ┌─────────────────────────────────────────────────────────────┐
   │  Calibrate: scipy.optimize.least_squares finds LMM params   │   ← step 3
   │  that match market IVs — using the NN INSIDE the loop       │
   └─────────────────────────────────────────────────────────────┘
                                │
                                ▼
        ┌───────────────────────────────────────────────┐
        │  Verify: reprice with full MC at θ*,          │   ← step 4
        │  Black-invert, compare IV_market | IV_NN | IV_MC │
        └───────────────────────────────────────────────┘
```

The "win" is that step 3 calls the NN (microseconds per evaluation) instead of
the MC pricer (seconds-to-minutes). In production this turns an overnight
calibration into an under-an-hour one.

---

## What's provided vs. what you write

**Provided** (infrastructure that isn't the learning goal — already in the skeleton):
- `mock_lmm_iv(lmm_params, T, K, F)` — the "MC pricer" you train the surrogate against.
  Same toy function as in the example.
- `mock_lmm_price(lmm_params, T, K, F, is_call)` — `black76_price` at `mock_lmm_iv`. Used in `verify`.
- `black76_price(F, K, T, sigma, is_call)` — standard Black-76 formula.
- `black76_implied_vol(price, F, K, T, is_call)` — solves Black for σ via `brentq`.
- Sampling ranges for LMM params + instruments (so your data lives in a sane region).
- Market instruments + `true_params` in `main()` (so you have something to calibrate to).

**You write** (the learning content):
1. **`generate_data(N, seed)`** — sample params + instruments, build the 7-D feature
   matrix, compute IVs by calling `mock_lmm_iv`. Returns `(X, y)`.
2. **`split_train_val(X, y, val_frac, seed)`** — small numpy shuffle + split.
3. **`Surrogate`** — small MLP (~2 hidden layers, ~64 units, SiLU or ReLU) that
   maps 7-D features → 1 scalar IV.
4. **`train_surrogate(...)`** — full-batch training loop with train/val MSE
   tracking; early stop optional.
5. **`nn_iv(model, params, instruments, device)`** — inference helper that
   assembles features and returns predicted IVs for a list of instruments at one
   set of params. Used inside the calibration loop.
6. **`calibrate(model, market_instruments, market_ivs, x0, bounds, device)`** —
   wraps `scipy.optimize.least_squares` with the NN as the residual function.
7. **`verify(model, theta_star, market_instruments, market_ivs, device)`** —
   prints the four-column residual table:
   market | NN(θ*) | MC(θ*) | calib resid | surrogate resid.
8. **`main()`** — argparse + wiring.

---

## Notes on each piece (the **why**)

Each TODO function's docstring carries detailed hints; here's the one-paragraph
**why** for each, so you can see how they fit together as a workflow.

1. **`generate_data`** — the NN's *textbook*. At inference time the calibration
   loop probes random-looking (params, instrument) combinations; the NN can
   only answer for combinations it's seen relatives of. Uniform sampling
   across `[LMM_PARAM_LO, LMM_PARAM_HI] × [T_LO, T_HI] × [F_LO, F_HI] ×
   log-moneyness range` is the simplest way to cover the calibrator's whole
   playing field.

2. **`split_train_val`** — gives you a "have I seen this before?" check during
   training. If train MSE keeps falling but val MSE rises, the NN is
   memorising specific rows instead of learning the function.

3. **`Surrogate`** — small fully-connected MLP. Must be **smooth** (no
   dropout, no batchnorm) because `least_squares` computes its Jacobian by
   finite differences; a noisy surrogate kills the optimiser's convergence.
   2 hidden layers × 64 units is plenty for a smooth 7-D function.

4. **`train_surrogate`** — bog-standard MSE training. Target: train MSE
   around 1e-4. Full-batch is fine at 10k rows. If you've done the previous
   two NN capstones, this is muscle memory.

5. **`nn_iv`** — the bridge between scipy (numpy in / numpy out) and PyTorch
   (tensors, devices, no_grad). Called *thousands* of times inside the
   optimiser, so it must be fast: assemble one feature matrix, one forward
   pass, return numpy. No Python loops.

6. **`calibrate`** — the workflow's whole point. Wrap `scipy.optimize.
   least_squares` around `nn_iv`. **Bounds are critical** — outside the
   training region the NN extrapolates wildly, and the optimiser will happily
   chase that nonsense. Pass `(LMM_PARAM_LO, LMM_PARAM_HI)` so it can't.

7. **`verify`** — the *acceptance test*. Two residuals:
   - **`IV_market − IV_MC(θ*)`** asks "does the LMM fit the market?" (nothing
     to do with the NN — this measures the model itself).
   - **`IV_NN(θ*) − IV_MC(θ*)`** asks "was the NN reliable at θ*?" (measures
     just the surrogate).
   Both must pass independently; the cheatsheet's verification section
   explains the valuation-control / model-risk framing.

---

## Data conventions (match the example so you can A/B against it)

| Quantity | Symbol | Range | Notes |
|---|---|---|---|
| σ-curve level | `sig_a` | `[0.10, 0.25]` | LMM param |
| σ-curve hump | `sig_c` | `[0.30, 0.50]` | LMM param |
| SABR α | `sabr_alpha` | `[0.005, 0.025]` | LMM param |
| Asymptotic correlation | `rho_inf` | `[0.10, 0.50]` | LMM param |
| Time to expiry | `T` | `[0.5, 10.0]` years | Instrument |
| Forward rate | `F` | `[0.02, 0.05]` | Instrument |
| Strike | `K` | `F * exp(U(−0.3, 0.3))` | Instrument; log-moneyness ∈ [−0.3, 0.3] |
| **Target** | `IV` | result of `mock_lmm_iv(...)` | Black implied vol |

**Feature vector (7-D per row):** `[sig_a, sig_c, sabr_alpha, rho_inf, T, log(K/F), F]`.
Using `log(K/F)` instead of raw K is a small win — moneyness is the structurally
meaningful quantity for the smile.

**Dataset size:** ~10 000 rows is plenty. The example uses 5 000 for ~3 s training.

---

## Verification table — what success looks like

After calibrating, print exactly this table (per the example):

```
          instrument | market |  NN(θ*)  |  MC(θ*)  |  calib  | surrogate
------------------------------------------------------------------------------
T=1.0 K=0.030 F=0.035 | 0.3591 | 0.3612  | 0.3600  |  -8.7bp | +12.3bp
T=2.0 K=0.040 F=0.040 | 0.3657 | 0.3654  | 0.3650  |  +6.8bp |  +3.8bp
...
```

**Two residuals**, two tolerances:

| Residual | Question | Pass if |
|---|---|---|
| `IV_market − IV_MC(θ*)` (**calib**) | Is the LMM well-fitted to market? | per-instrument < 25 bp, RMSE < 15 bp |
| `IV_NN(θ*) − IV_MC(θ*)` (**surrogate**) | Was the NN reliable at θ*? | < 10 bp per instrument |

The two checks are independent and both must pass. The cheatsheet's
"§ Verification" section explains why.

---

## Suggested milestones (3 hours)

| # | Step | Est. |
|---|------|------|
| 1 | Skim the existing demo + cheatsheet sections 1–4. | 15 min |
| 2 | Implement `generate_data` + sanity-check shapes / value ranges. | 25 min |
| 3 | Implement `Surrogate` + `train` (you've done this twice now). | 30 min |
| 4 | Implement `nn_iv` inference helper; sanity-check on one instrument. | 15 min |
| 5 | Implement `calibrate` with `least_squares`. | 35 min |
| 6 | Implement `verify` — four-column table + RMSE summary. | 30 min |
| 7 | Wire `main`, run end-to-end, eyeball the residuals. | 20 min |
| 8 | (Stretch) Timing comparison: MC-in-loop vs NN-in-loop wall time. | 15 min |

Total core: ~170 min ≈ 2h50.

---

## Setup + run

The script needs `numpy`, `torch`, `scipy`, `joblib`. The simplest way is to
reuse the existing `ml/neural_networks` uv environment (which already has all
four installed for the NN capstones):

```bash
cd quant_finance/projects/lmm_nn_surrogate/capstone

# Use the existing NN-capstone uv env:
/home/zlac116/Code/learning/ml-revision/ml/neural_networks/.venv/bin/python \
    surrogate.py                                     # defaults
/home/zlac116/Code/learning/ml-revision/ml/neural_networks/.venv/bin/python \
    surrogate.py --n-data 20000 --epochs 3000
```

Or set up a dedicated uv project for this capstone if you prefer
(`uv init && uv add numpy torch scipy joblib` in this folder).

The script saves two artifacts at the end:
- `surrogate.pt` — model state_dict
- `calibration_result.json` — calibrated params + verification table

(Stretch: if you add input scaling, also save `feature_scaler.joblib` — needed to
predict on new raw inputs later. In production this is mandatory; here it's optional
because the feature ranges are already well-conditioned.)

---

## Stretch goals (skip if you're at 3h)

1. **Timing comparison** — wrap a slow_mc_calibrate() that calls `mock_lmm_iv` directly
   inside `least_squares`. Compare wall-time vs the NN-driven path.
2. **Train/val/test split + early stopping** — like your regression capstone.
3. **Feature scaling** — `StandardScaler` on the 7-D input. The inputs are NOT
   well-conditioned: `T` (max 10) and `sabr_alpha` (max 0.025) span ~400×.
   Empirically (10k rows, Adam @ 2e-3, 2000 epochs):

   |               | val MSE @ epoch 200 | val MSE @ epoch 2000 |
   |---------------|---------------------|----------------------|
   | scaled X      | 4.0e-5              | 3.4e-6               |
   | unscaled X    | 2.6e-3              | 2.9e-6               |

   So scaling buys you **~5× faster time-to-convergence** and a smoother
   (monotonic) loss curve. But Adam adapts well enough that at full epoch budget
   the *final* loss is nearly identical (the unscaled run is ~bumpy in the middle
   — the optimiser briefly *increases* loss around epoch 1000 before recovering).
   Production pipelines scale; for a 3-hour capstone with 2000 epochs of headroom
   you can skip it. Cost of adding it: threading `x_scaler` through `nn_iv` and
   `calibrate` so the calibration applies the same transform, plus saving the
   scaler as a third artifact (`feature_scaler.joblib`).
4. **Multi-start calibration** — try 5 different `x0` perturbations and report
   the best fit + variance.
5. **Smile diagnostics** — for a fixed (T, F), sweep K across the strike range
   and plot IV vs log-moneyness. Both NN and "MC" should produce the same curve.

---

## Success criteria

- The four-column verification table prints for ≥ 3 calibration instruments.
- Per-instrument **calib** residuals < 25 bp; RMSE < 15 bp.
- Per-instrument **surrogate** residuals < 10 bp.
- `uv run` (or plain `python`) end-to-end completes in < 30 s for default `N`.
- Saved artifacts exist on disk.

Ask **"review my LMM surrogate capstone"** when ready and I'll assess against this
spec.
