# SABR Pricing — End-to-End Walkthrough

Concrete numerical walkthrough of the full pipeline: **calibration → smile vol → price**.

---

## Overall process

```
   ┌────────────────────────────────┐    ┌───────────────────┐    ┌──────────────┐
   │ CALIBRATION (per cell)         │    │ SMILE VOL         │    │ PRICING      │
   │ ─ Fix β by market convention   │    │ For any strike K, │    │ Plug         │
   │ ─ Solve α from ATM (cubic)     │ ─► │ Hagan's formula   │ ─► │ σ_SABR(K)    │
   │ ─ Fit (ρ, ν) to smile          │    │ gives σ_SABR(K)   │    │ into         │
   │   by least-squares             │    │ (closed form)     │    │ Black-76     │
   └────────────────────────────────┘    └───────────────────┘    └──────────────┘
```

SABR is **not a pricing model on its own** — it's a smile interpolator. It tells you what Black-76 volatility to use at each strike, then Black-76 does the actual pricing.

**One fit per `(expiry, tail)` cell of the swaption cube.** A 3 × 3 × 5 cube (3 expiries × 3 tails × 5 strikes) → **9 SABR fits**, one per cell.

---

## Setup (used throughout)

A single cell of the swaption cube: **1y × 5y swaption** (1-year expiry into a 5-year swap).

| Quantity | Value |
|---|---|
| Expiry $T$ | 1.0 year |
| Forward swap rate $F$ | 0.04 (4.0%) |
| Market Black vols at strikes $\{F-100\mathrm{bp}, F-50\mathrm{bp}, F, F+50\mathrm{bp}, F+100\mathrm{bp}\}$ | $\{0.310, 0.275, 0.250, 0.240, 0.245\}$ (typical rates negative-skew smile) |
| Fixed $\beta$ (rates convention) | 0.5 |

Strikes in absolute terms: $\{0.030, 0.035, 0.040, 0.045, 0.050\}$.

---

## CALIBRATION

### Step 1 — Fix $\beta$ by market convention

$\beta$ controls the **backbone** (how the smile shifts as the forward moves). Conventional choices:

| Market | Typical $\beta$ |
|---|---|
| Rates (caps, swaptions) | **0.5** |
| Equity options | $\approx 1.0$ |
| FX | 0.5 – 1.0 |
| Negative rates | $\approx 0.0$ |

For our cell: **$\beta = 0.5$**. Don't try to fit it — SABR is over-parameterised if all four are free.

### Step 2 — Solve $\alpha$ from the ATM vol (cubic equation)

Plug $K = F$ into Hagan's formula and impose $\sigma_{SABR}(F) = \sigma_{ATM}^{\mathrm{market}}$. The result is a **cubic in $\alpha$**:

$$\frac{\alpha}{F^{1-\beta}} \left[1 + \left(\frac{(1-\beta)^2}{24} \cdot \frac{\alpha^2}{F^{2-2\beta}} + \frac{\rho \beta \nu \alpha}{4 \, F^{1-\beta}} + \frac{(2 - 3\rho^2)\nu^2}{24}\right) T \right] = \sigma_{ATM}^{\mathrm{market}}$$

Rearranged into standard form $A \alpha^3 + B \alpha^2 + C \alpha + D = 0$:

```
A = (1 − β)² · T  /  (24 · F^(2 − 2β))
B = ρ · β · ν · T  /  (4 · F^(1 − β))
C = 1 + (2 − 3ρ²) · ν² · T  /  24
D = −σ_ATM · F^(1 − β)
```

Solve with `np.roots([A, B, C, D])` and take the smallest **real positive** root.

**Quick mental check — leading-order shortcut**

If you drop the $O(T)$ correction (small $T$ or small $\nu$), the cubic collapses to:

```
σ_ATM ≈ α / F^(1 − β)     ⟹     α ≈ σ_ATM · F^(1 − β)
```

| β | α ≈ |
|---|---|
| 1.0 (equity, lognormal) | $\sigma_{ATM}$ |
| 0.5 (rates) | $\sigma_{ATM} \cdot \sqrt{F}$ |
| 0.0 (normal / Bachelier-like) | $\sigma_{ATM} \cdot F$ |

For our example ($F = 0.04$, $\sigma_{ATM} = 0.25$, $\beta = 0.5$, $T = 1$):

- Leading-order: $\alpha \approx 0.25 \cdot \sqrt{0.04} = 0.0500$
- Full cubic (with the calibrated $\rho = -0.158$, $\nu = 0.776$ from Step 3): $\alpha \approx 0.0478$

The cubic correction matters most for **long $T$** and **high vol-of-vol $\nu$**. For short-dated quiet markets, the leading-order shortcut is within a few percent of the full answer.

**Note:** the cubic depends on $(\rho, \nu)$ — so this step is nested inside Step 3's optimiser, re-solved at every iteration.

### Step 3 — Fit $(\rho, \nu)$ to the smile by least-squares

For the remaining 4 strikes (everything except ATM), minimise the squared error between market and SABR vols:

$$\min_{\rho, \nu} \sum_{k \neq \mathrm{ATM}} \left(\sigma_k^{\mathrm{market}} - \sigma_{SABR}(K_k; \alpha(\rho, \nu), \beta, \rho, \nu)\right)^2$$

At each optimiser step:
1. Re-solve the ATM cubic for $\alpha$ given the current $(\rho, \nu)$.
2. Compute SABR vol at each non-ATM strike via Hagan's formula.
3. Sum squared residuals.

Bounds: $\rho \in [-0.99, 0.99]$, $\nu > 0$.

**Concrete result** for our example (verified with `scipy.optimize.minimize`):

| Parameter | Fitted value | What it captures |
|---|---|---|
| $\alpha$ | 0.0478 | overall vol level (anchored at ATM) |
| $\beta$ | 0.5 (fixed) | backbone elasticity |
| $\rho$ | $-0.158$ | skew (negative = vol-up when rates-down) |
| $\nu$ | 0.776 | curvature / vol-of-vol |

Fit residuals are within ~10 bp of market — acceptable for a hand-picked smile that isn't a perfect SABR shape.

---

## PRICING

### Smile vol at any strike

Given the fitted $(\alpha, \beta, \rho, \nu)$, compute $\sigma_{SABR}(K)$ via Hagan's formula (see structure below). Then plug into the appropriate Black-style pricer.

### Black-76 vs Bachelier — which pricer?

SABR can output **either** lognormal vol (for Black-76) **or** normal vol (for Bachelier). Choose based on the rate regime:

| Regime | Forward $F$ | Right pricer | SABR output to use |
|---|---|---|---|
| Normal positive rates | $F > 50$ bp | **Black-76** | $\sigma_{LN}^{SABR}(K)$ — lognormal vol |
| Low / near-zero rates | $F < 50$ bp, $\sigma > 0.5 F$ | **Bachelier** | $\sigma_n^{SABR}(K)$ — normal vol |
| Negative rates | $F < 0$ | **Bachelier** (Black-76 breaks: $\ln(F)$ undefined) | $\sigma_n^{SABR}(K)$ |

Hagan's formula has both a **lognormal version** (the classic) and a **normal version** (for $\beta = 0$ or low-rate regimes). Most production code computes both and picks based on the rate level.

**Black-76 form** (positive rates):

$$\mathrm{Swaption}(K) = A(0) \cdot \mathrm{Black76}\!\left(F, K, T, \sigma_{LN}^{SABR}(K)\right)$$

$$\mathrm{Black76\ payer} = F \, N(d_1) - K \, N(d_2), \qquad d_{1,2} = \frac{\ln(F/K) \pm 0.5 \, \sigma^2 T}{\sigma \sqrt{T}}$$

**Bachelier form** (low/negative rates):

$$\mathrm{Swaption}(K) = A(0) \cdot \mathrm{Bachelier}\!\left(F, K, T, \sigma_n^{SABR}(K)\right)$$

$$\mathrm{Bachelier\ payer} = (F - K) \, N(d) + \sigma_n \sqrt{T} \, \phi(d), \qquad d = \frac{F - K}{\sigma_n \sqrt{T}}$$

where $\phi(\cdot)$ is the standard normal **density** (not CDF).

**Conversion** (Hagan-Kennedy): you can convert between lognormal and normal vol via:

$$\sigma_n \approx \sigma_{LN} \cdot F \cdot \left(1 - \frac{1}{24} \ln^2(F/K) + \ldots\right)$$

so if SABR gives you $\sigma_{LN}$ but you need $\sigma_n$, convert rather than re-fit. Useful for rates books that flipped to Bachelier conventions during the negative-rate era (~2014-2022).

### Worked pricing — 1y × 5y payer swaption struck at $K = 0.045$

1. **Compute $\sigma_{SABR}(0.045)$** using Hagan's formula with $(\alpha, \beta, \rho, \nu) = (0.0478, 0.5, -0.158, 0.776)$, $F = 0.04$, $T = 1$:

   $$\sigma_{SABR}(0.045) \approx 0.2414$$

   (Market quote at this strike is 0.2400 — difference of ~14 bp is the calibration residual.)

2. **Black-76** with this vol:

   $$d_1 = \frac{\ln(F/K) + 0.5 \sigma^2 T}{\sigma \sqrt{T}}, \quad d_2 = d_1 - \sigma\sqrt{T}$$

   $$\mathrm{Black76\ payer} = F \, N(d_1) - K \, N(d_2)$$

3. Multiply by annuity $A(0)$:

   $$\mathrm{Swaption\ price} = A(0) \cdot \mathrm{Black76\ payer}$$

That's the price. The whole point of SABR was step 1 — without it, you'd have to guess what vol to use at $K = 0.045$ (the cube only quotes 5 discrete strikes).

### Smile picture

After calibration you can plot $\sigma_{SABR}(K)$ for a fine grid of strikes:

```
σ_SABR  ▲
0.31 ─  ●     ←── market quotes (at 5 strikes)
        │ ─
0.29 ─  │   ─                              ← SABR smile (continuous)
        │    ●
0.27 ─  │      ─
        │        ─
0.25 ─  │          ●────── ATM = F ─────
        │             ─
0.24 ─  │              ─    ●          ●
        │                ─     ─    ─
0.23 ─  └──────────────────────────────────►  K
        0.030  0.035  0.040  0.045  0.050
                ATM=F=0.04
```

Market quotes: 0.310, 0.275, 0.250, 0.240, 0.245 (typical rates negative skew — high vol for OTM puts, low for ATM, slight up-tick in the OTM-call wing from positive $\nu$). SABR fills in the gaps continuously between quotes.

---

## Where SABR fits in the bigger picture

| Use case | Role of SABR |
|---|---|
| **European swaption pricing** | Get $\sigma_{SABR}(K)$, plug into Black-76. Single forward, single expiry. |
| **Caplet smile** | Same — per-caplet SABR fit, then Black-76. |
| **Smile-aware LMM Bermudan calibration** | Use SABR-implied vols at the *Bermudan strike* to calibrate the LMM instantaneous vols (see `lmm_end_to_end_walkthrough.md`). |
| **Hedging skew risk** | Bump $\rho$ → vega-by-skew. Bump $\nu$ → vega-by-curvature. |

---

## Hagan's formula — the structure (don't memorise)

The closed-form vol approximation Hagan derived (Hagan et al. 2002):

$$\sigma_{SABR}(K, F) \approx \frac{\alpha}{(FK)^{(1-\beta)/2}} \cdot \frac{1}{1 + \frac{(1-\beta)^2}{24}\ln^2(F/K) + \ldots} \cdot \frac{z}{x(z)} \cdot \left[1 + (\text{higher-order } T \text{ corrections})\right]$$

with

$$z = \frac{\nu}{\alpha} (FK)^{(1-\beta)/2} \ln(F/K), \qquad x(z) = \ln\!\left(\frac{\sqrt{1 - 2\rho z + z^2} + z - \rho}{1 - \rho}\right)$$

**What you need to know:**

- It's a **small-time Taylor expansion** around ATM — degrades for very long $T$ or deep OTM strikes
- ATM limit ($K \to F$): the $\ln(F/K)$ terms vanish, $z \to 0$, $z/x(z) \to 1$, giving $\sigma_{ATM} \approx \alpha / F^{1-\beta}$ (plus tiny corrections)
- Production replacements: **Obloj reformulation** (numerically stable in the wings), **arbitrage-free SABR** (Hagan-Kumar 2014, fixes negative-density at deep OTM)

---

## SABR vs LMM at a glance

| Property | SABR | LMM |
|---|---|---|
| Forwards modelled | **One** (a single $F$ per cell) | **All** at once |
| Stochastic vol? | Yes ($\alpha$ has its own SDE) | No (instantaneous $\sigma_i$ deterministic) |
| Output | A vol curve $\sigma(K)$ per cell | A 3D array of path scenarios |
| Right tool for | European swaptions, caps, FX vanilla | Bermudans, callable bonds, exotics |
| Pricing | Black-76 with $\sigma_{SABR}(K)$ | Monte Carlo on paths (LSMC for callables) |

They are **complementary**. SABR handles the smile on a single forward; LMM handles the joint evolution across forwards. Combine via **SABR-LMM** when you need both (smile-aware Bermudans).

---

## Quick-reference formulas

| Concept | Formula |
|---|---|
| SABR SDE | $dF = \alpha F^\beta dW$, $d\alpha = \nu \alpha dZ$, $d\langle W, Z \rangle = \rho \, dt$ |
| ATM vol (leading order) | $\sigma_{ATM} \approx \alpha / F^{1-\beta}$ |
| Hagan $z$ | $z = (\nu/\alpha) (FK)^{(1-\beta)/2} \ln(F/K)$ |
| Hagan $x(z)$ | $x(z) = \ln\!\left(\frac{\sqrt{1 - 2\rho z + z^2} + z - \rho}{1 - \rho}\right)$ |
| Cubic for $\alpha$ | $A\alpha^3 + B\alpha^2 + C\alpha + D = 0$ with $A=(1-\beta)^2 T/(24 F^{2-2\beta})$, $B=\rho\beta\nu T/(4 F^{1-\beta})$, $C=1+(2-3\rho^2)\nu^2 T/24$, $D=-\sigma_{ATM} F^{1-\beta}$ |
| Black-76 payer | $A(0) \cdot \left[F \, N(d_1) - K \, N(d_2)\right]$, $d_{1,2} = \frac{\ln(F/K) \pm 0.5 \, \sigma^2 T}{\sigma \sqrt{T}}$ |
| Bachelier payer | $A(0) \cdot \left[(F-K) N(d) + \sigma_n \sqrt{T} \phi(d)\right]$, $d = (F-K)/(\sigma_n \sqrt{T})$ |
| Per-cell calibration | (1) fix $\beta$, (2) solve $\alpha$ from ATM cubic, (3) fit $(\rho, \nu)$ by LSQ to wings |

---

## How to remember

Three sentences:

1. **"$\alpha$ sets level, $\beta$ fixed, $\rho$ does skew, $\nu$ does curvature."**
2. **"SABR gives you $\sigma_{SABR}(K)$ — plug it into Black-76 for the price."** SABR is an interpolator, not a pricer.
3. **"One SABR fit per (expiry, tail) cell."** The cube has many cells; each is independent.
