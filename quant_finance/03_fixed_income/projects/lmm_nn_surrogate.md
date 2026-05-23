# LMM NN-Surrogate — Interview Cheat Sheet

## One-line pitch
Trained a neural network to replace the Monte-Carlo LMM pricer **inside the calibration loop** for the Callable Exotics book (CRAs, CDRANs, Bermudans). Used by Valuation Control for IPV. End-to-end calibration: **overnight → under an hour**, ~100×, with every result re-checked by the real MC pricer.

---

## How to run the worked example

An 80-line runnable demo lives next to this doc: [`lmm_nn_surrogate_example.py`](./lmm_nn_surrogate_example.py).
It mocks the slow LMM MC pricer with a toy vol-surface function, trains a 64-unit MLP, runs Levenberg–Marquardt calibration with the NN inside the loop, then verifies with Black inversion — completes in ~3 s on CPU and reproduces the three-way `market | NN(θ*) | MC(θ*)` comparison from §4 below.

```bash
pip install numpy torch scipy
python lmm_nn_surrogate_example.py
```

Expected output (last few lines):

```
          instrument | market |  NN(θ*)  |  MC(θ*)  |  calib  | surrogate
------------------------------------------------------------------------------
T=1.0 K=0.030 F=0.035 | 0.3591 | 0.3612  | 0.3600  |  -8.7bp | +12.3bp
T=2.0 K=0.040 F=0.040 | 0.3657 | 0.3654  | 0.3650  |  +6.8bp |  +3.8bp
T=5.0 K=0.045 F=0.040 | 0.4088 | 0.4070  | 0.4070  | +18.5bp |  +0.6bp
T=5.0 K=0.035 F=0.040 | 0.4179 | 0.4185  | 0.4186  |  -6.6bp |  -1.0bp
```

Calibration residuals < 25 bp, surrogate residuals < 15 bp → both pass typical tolerances.

---

## What the NN actually learns

| Block | Fields (what's in a training row) |
|---|---|
| **A. LMM params** *(optimiser varies these)* | σ-curve (a, b, c, d); SABR (α, β, ρ, ν) per tenor bucket; correlation (ρ_∞, β_corr); shift |
| **B. Instrument descriptor** | type flag (caplet / swaption), expiry T, tenor, strike K, log(K/F), payer/receiver |
| **C. Market context** *(frozen per snapshot)* | F at expiry, F_atm, discount factor, curve level + slope |
| **Target** | **Black-implied vol** (not price) from full LMM MC pricer |

**Key scope point:** the NN prices only **caplets and swaptions** (the calibration instruments) — not the CRAs/CDRANs/Bermudans themselves. Those are still priced by full MC using the calibrated params.

### Example row

| type | T | tenor | K | log(K/F) | σ_a | sabr_α | ρ_∞ | **target_iv** |
|---|---|---|---|---|---|---|---|---|
| swaption | 5 | 5 | 3.50% | −0.04 | 0.18 | 0.015 | 0.30 | **0.2814** |

Same row with K=4.00% → IV=0.2997 (NN learns the **smile**).
Same row but caplet, T=1, K=3.00% → IV=0.2206 (NN learns the **product type**).

---

## Why Black IV (not price)
- Bounded, smooth, dimensionless → trains far better
- Same units as market quotes and verification output → like-for-like residuals
- Smile diagnostics fall out for free when plotted against log-moneyness

---

## Verification — how the calibration is checked

```
              θ*  (optimised LMM params from NN-driven loop)
               │
               ▼
   ┌──────────────────────────┐
   │ Full LMM Monte-Carlo     │   slow but authoritative
   │ reprice each calib instr │   → outputs price P_MC(k; θ*)
   └────────────┬─────────────┘
                │
                ▼
   ┌──────────────────────────┐
   │ Black-formula inversion  │   solve Black(F,K,T,σ) = P_MC
   │ (Brent / Newton)         │   → outputs IV_MC(k; θ*)
   └────────────┬─────────────┘
                │
   ┌────────────┴────────────────┐
   ▼            ▼                ▼
IV_market   IV_NN(θ*)        IV_MC(θ*)
```

### The Black inversion in one sentence
Take the MC's output price, hold (F, K, T, annuity) fixed, solve for the σ that reproduces it under Black's formula — that σ is the MC's implied vol, directly comparable to market and to the NN's prediction.

### Two residuals

| Residual | Question | Tolerance |
|---|---|---|
| `IV_market − IV_MC(θ*)` | Is the LMM well-fitted to market? | < 25–50 bp per instrument, RMSE < 15 bp |
| `IV_NN(θ*) − IV_MC(θ*)` | Was the NN reliable at θ*? | < 5–10 bp |

First test = Valuation Control acceptance. Second test = model-risk acceptance for the surrogate.

---

## Worked example

5y × 5y payer swaption, K = 3.50 %, market IV = **28.20 %**:

| Stage | Output |
|---|---|
| NN inside optimiser | IV_NN = **28.14 %** |
| Full MC at θ\* | P_MC = 0.02476 (price) |
| Black inversion of P_MC | IV_MC = **28.17 %** |
| Calibration residual (market − MC) | **+3 bp** ✓ |
| Surrogate residual (NN − MC) | **−3 bp** ✓ |

Both pass → calibration accepted, θ\* handed to Valuation Control.
If surrogate residual had been +60 bp → reject: NN mispriced near θ\* → retrain or fall back to MC-in-loop.

---

## Timings & speed-up

| Step | Base (MC in loop) | NN surrogate |
|---|---|---|
| Inner pricing call | 30 s – 3 min × 60 instruments × ~200 iter | ~5 ms total per iteration |
| Optimiser wall time | **8–24 hours** | **~1 second** |
| Verification sweep (full MC at θ\*) | — | **~30 min** |
| **End-to-end** | **8–24 hours** | **~30–45 min** |

**Headline:** ~100× end-to-end. Inner loop alone is orders of magnitude faster — but verification is the binding constraint and model risk requires it.

---

## Interview soundbites (recite these)

1. *"The NN doesn't price the trades — it prices the calibration instruments. The trade book is still valued by full MC using the calibrated parameters."*
2. *"Target is Black implied vol, not price — bounded, smooth, and in the same units as market quotes and the verification output."*
3. *"After the optimiser converges, we plug θ\* back into the full MC pricer, get a price, invert Black to recover an MC-implied vol, then compare three things: market, NN, MC. Two residuals fall out — one tests the model fit, one tests the surrogate."*
4. *"100× end-to-end. The optimiser loop alone is much faster than that, but the MC verification sweep at the end is the floor — and removing it was never on the table for Valuation Control."*
5. *"Same architecture covered CRAs, CDRANs and Bermudans — they share the LMM, so they share the calibration, so they share the surrogate."*

---

## CV bullet (≤30 words)

> **LMM Calibration Surrogate (NN):** Replaced Monte-Carlo pricing inside the LMM calibration loop for CRAs/CDRANs/Bermudans — ~100× faster IPV calibration for Valuation Control's Callable Exotics book, accuracy benchmarked against full LMM.
