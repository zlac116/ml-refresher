# SABR Cheatsheet — Smile Interpolator, Not a Pricing Model

**The point:** SABR is a **smile parameterisation**, not a pricing model in its own right. It generates a closed-form *approximation* (Hagan 2002) for the implied volatility a constant-vol Black-76 (or Bachelier) model would need at each strike. You then plug that vol into Black-76 for the actual price. Four parameters, four jobs — and one of them is fixed exogenously.

---

## The single specification

$$\boxed{\;dF_t = \alpha_t \, F_t^{\beta} \, dW_t, \qquad d\alpha_t = \nu \, \alpha_t \, dZ_t, \qquad d\langle W, Z\rangle_t = \rho\, dt\;}$$

Stochastic vol on a stochastic forward, with constant-elasticity-of-variance ($F^\beta$) on the forward.

The deliverable is **Hagan's formula** — a closed-form approximation for the lognormal implied vol $\sigma_{LN}(K)$ that Black-76 needs to reproduce SABR's price at strike $K$. You don't price options with SABR directly; you fit SABR to the smile, then use $\sigma_{LN}(K)$ in Black-76 (or $\sigma_n(K)$ in Bachelier).

---

## The four parameters — one job each

| Param | Range | Controls | How it's calibrated |
|---|---|---|---|
| **$\alpha$** | $> 0$ | Overall vol **level** (ATM) | Solved from ATM market vol given $\beta, \rho, \nu$ |
| **$\beta$** | $[0, 1]$ | Backbone **elasticity** (sticky-strike vs sticky-delta dynamics) | **Fixed exogenously** by market convention |
| **$\rho$** | $[-1, +1]$ | **Skew** (correlation of vol with forward) | Calibrated to ATM-vs-OTM put gap |
| **$\nu$** | $> 0$ | **Curvature** (vol-of-vol) | Calibrated to wing convexity |

**Why $\beta$ is fixed**: with all four free, SABR is **over-parameterised** — many $(\alpha, \beta)$ pairs reproduce the same ATM vol. Market convention picks $\beta$ first:

| Market | Typical $\beta$ | Backbone behaviour |
|---|---|---|
| Equity options | $\sim 1$ | Lognormal — pure sticky-delta |
| Rates (caps, swaptions) | $\sim 0.5$ | CIR-like, between normal and lognormal |
| FX | $\sim 0.5$–$1$ | Varies by pair |
| Negative-rate environment | $\sim 0$ | Normal/Bachelier-like backbone |

---

## Limiting cases — the model degenerates cleanly

| $\beta$ | Limit model | Notes |
|---|---|---|
| **$\beta = 0$** | **Normal SABR** (Bachelier with stoch vol) | Use when forwards can be near zero or negative |
| **$\beta = 1$** | **Lognormal SABR** (Black-76 with stoch vol) | The original Hagan formulation |
| **$\nu = 0$** | **CEV model** | No stochastic vol; just elasticity |
| **$\nu = 0, \beta = 1$** | **Black-76** | Recover the constant-vol baseline |

---

## How to remember

Three sentences, drilled to muscle memory:

1. **"$\alpha$ sets level, $\beta$ fixed, $\rho$ does skew, $\nu$ does curvature."** Four parameters, four jobs, one fixed.
2. **"SABR gives you $\sigma_{LN}(K)$ — plug into Black-76 for the price."** SABR is a smile interpolator, not a pricer.
3. **"$\beta = 1 \approx$ Black-76, $\beta = 0 \approx$ Bachelier."** The two extreme regimes — pick based on the rate level.

---

## When SABR is the right tool

- Smile-aware option pricing on a **single** forward (caps, single-tenor swaptions, FX vanilla)
- Generating $\sigma_{LN}$ at strikes that aren't market-quoted (interpolation/extrapolation)
- Hedging **vega exposure to specific smile features** (skew → $\rho$, curvature → $\nu$)

## When it isn't

- **Multi-forward products** (Bermudan swaptions, callable structures): use **SABR-LMM** (one SABR per forward, unified by an LMM-style correlation)
- **Long-dated** options where Hagan's expansion degrades — refit term structure of $(\alpha, \rho, \nu)$
- **ATM-only** products — Black-76 with the ATM vol is enough, SABR is overkill

---

## Bonus — Hagan's formula, the structure (don't memorise)

$$\sigma_{LN}^{SABR}(F, K, T) \approx \frac{\alpha}{(FK)^{(1-\beta)/2} \cdot \big[1 + \tfrac{(1-\beta)^2}{24} \ln^2(F/K) + \dots\big]} \cdot \frac{z}{x(z)} \cdot \big[1 + (\text{higher-order } T\text{ corrections})\big]$$

with $z = \tfrac{\nu}{\alpha}(FK)^{(1-\beta)/2}\ln(F/K)$ and $x(z)$ a logarithm-of-square-root correction.

**Don't memorise this**. The full expression is in `02_bs_family_and_asset_classes.ipynb`. What you should know:

- Hagan formula is a **small-time Taylor expansion** around ATM — degrades for very long $T$ or very deep OTM strikes
- Production fixes: **Obloj reformulation** (more numerically stable in the wings), **arbitrage-free SABR** (Hagan-Kumar 2014, removes the negative-density issue at deep OTM)

---

## What NOT to memorise (reach-for material)

- The full Hagan formula expansion
- Obloj's reformulation for OTM stability
- Arbitrage-free SABR adjustments for negative density
- ZABR / dynamic SABR extensions
- Calibration loss functions and weighting schemes
- SABR-LMM cross-correlation parameterisation

All in `02_bs_family_and_asset_classes.ipynb` and (for SABR-LMM) `05_libor_market_model.ipynb`.
