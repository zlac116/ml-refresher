# LMM Pricing — End-to-End Walkthrough

Concrete numerical walkthrough of the full pipeline: **calibration → simulation → pricing**.

---

## Overall process

```
   ┌─────────────────────────┐    ┌──────────────────┐    ┌──────────────┐
   │ CALIBRATION             │    │ SIMULATION       │    │ PRICING      │
   │ ─ Bootstrap L_i(0)      │    │ Roll forwards    │    │ Caplet:      │
   │   from discount factors │ ─► │ forward in time  │ ─► │   avg payoff │
   │ ─ Bootstrap σ_i         │    │ under terminal   │    │ European:    │
   │   from cap vols         │    │ measure          │    │   avg payoff │
   │ ─ Build ρ_ij            │    │ (Euler-Maruyama) │    │ Bermudan:    │
   │   (Rebonato)            │    │                  │    │   LSMC       │
   └─────────────────────────┘    └──────────────────┘    └──────────────┘
```

**Where SABR fits in:** SABR is a **separate model** for the swaption volatility *smile* (one fit per `(expiry, tail)` cell of the cube). It is **not** part of the LMM pipeline above. Two roles:
1. **European swaption pricing** — direct: use SABR-implied vol in Black-76.
2. **Smile-aware LMM Bermudan calibration** *(production)* — replace cap vols with **co-terminal European swaption Black vols** as the bootstrap target, with SABR providing the strike-correct vol. This document uses caps for simplicity.

---

## Setup (used throughout)

| Quantity | Value |
|---|---|
| Tenor dates $T_i$ | $[1, 2, 3, 4, 5]$ |
| Forwards | $L_0, L_1, L_2, L_3$ |
| $\delta_i$ | $[1, 1, 1, 1]$ |

---

## CALIBRATION

### 1. Bootstrap forwards from discount factors

$$L_i(0) = \frac{1}{\delta_i}(\frac{P(0, T_i)}{P(0, T_{i+1})} - 1)$$

| $P(0, T_i)$ | 0.96 | 0.92 | 0.88 | 0.84 | 0.80 |
|---|---|---|---|---|---|
| $L_i(0)$ | 0.0435 | 0.0455 | 0.0476 | 0.0500 | — |

### 2. Bootstrap instantaneous vols $\sigma_i$ from cap quotes

LMM caplet on $L_i$ has Black-76 closed form:

$$\mathrm{Caplet}_i = \delta_i   P(0, T_{i+1}) \cdot \mathrm{Black76}(L_i(0), K, T_i, \sigma_i^{\mathrm{Black}})$$

A **cap** = sum of caplets, all sharing one quoted flat Black vol. Bootstrap forward-by-forward by matching cumulative variance:

Equate cumulative variances:

$$\sigma_i^2   T_i = (\sigma_i^{\mathrm{Black}})^2   T_i$$

then solve forward-by-forward:

$$\sigma_i = \sqrt{\frac{(\sigma_i^{\mathrm{Black}})^2   T_i - \sum_{k=0}^{i-1} \sigma_k^2 (T_k - T_{k-1})}{T_i - T_{i-1}}}$$

| Forward | $L_0$ | $L_1$ | $L_2$ | $L_3$ |
|---|---|---|---|---|
| $\sigma_i$ | 0.30 | 0.28 | 0.25 | 0.22 |

### 3. Forward-forward correlation matrix $\rho_{ij}$ (Rebonato)

$$\rho_{ij} = \exp(-\beta   |T_i - T_j|), \qquad \beta = 0.1$$

|   | $L_0$ | $L_1$ | $L_2$ | $L_3$ |
|---|---|---|---|---|
| $L_0$ | 1.00 | 0.90 | 0.82 | 0.74 |
| $L_1$ | 0.90 | 1.00 | 0.90 | 0.82 |
| $L_2$ | 0.82 | 0.90 | 1.00 | 0.90 |
| $L_3$ | 0.74 | 0.82 | 0.90 | 1.00 |

---

## SIMULATION

### Model SDE (terminal measure $Q^{T_N}$, numeraire $= P(0, T_N)$)

$$\frac{dL_i(t)}{L_i(t)} = \mu_i(t) dt + \sigma_i(t) dW_i(t)$$

**Drift** under terminal measure:

$$\mu_i(t) = -\sigma_i(t)\sum_{j=i+1}^{N-1}\frac{\delta_j L_j(t)}{1 + \delta_j L_j(t)} \sigma_j(t) \rho_{ij}$$

- $L_{N-1}$ (last forward) is a martingale: $\mu_{N-1} = 0$.
- All earlier forwards have negative drift.

### Euler step (one path, $t \to t + \Delta t$)

```
1.  z̃ ~ N(0, I)                    (independent standard normals)
2.  Z = L · z̃                      (Cholesky L: ρ = L Lᵀ → correlated shocks)
3.  Compute μ_i(t) using current L_j(t)
4.  L_i(t+Δt) = L_i(t) · exp((μ_i − ½σ_i²)Δt + σ_i √Δt · Z_i)
5.  If t ≥ T_i: freeze L_i (it has reset)
```

### Worked single step

$t = 0$, $\Delta t = 1/12$, correlated shocks $Z = (0.5, 0.4, 0.6, 0.7)$.

**Drift for $L_0$:**

$$\mu_0 = -0.30 \times [\tfrac{0.0455}{1.0455}\cdot 0.28 \cdot 0.905 + \tfrac{0.0476}{1.0476}\cdot 0.25 \cdot 0.819 + \tfrac{0.0500}{1.0500}\cdot 0.22 \cdot 0.741] \approx -0.00843$$

**Update:**

$$L_0(\tfrac{1}{12}) = 0.0435 \cdot \exp((-0.00843 - 0.5\cdot 0.30^2)\cdot\tfrac{1}{12} + 0.30\cdot\sqrt{\tfrac{1}{12}}\cdot 0.5) \approx 0.0452$$

| | $L_0$ | $L_1$ | $L_2$ | $L_3$ |
|---|---|---|---|---|
| $t = 0$ | 0.0435 | 0.0455 | 0.0476 | 0.0500 |
| $t = \tfrac{1}{12}$ | 0.0452 | 0.0468 | 0.0496 | 0.0522 |

### Output

After $\sim 100$ steps × $\sim 10{,}000$ paths:

```python
paths.shape == (n_paths, n_time_steps, n_forwards) == (10_000, 100, 4)
```

`paths[p, t, i]` = value of $L_i$ at time-step $t$ on path $p$.

---

## PRICING

### Caplet on $L_0$, strike $K$ (vanilla)

```python
L0_at_reset  = paths[:, t_idx_T1, 0]                    # (n_paths,)
payoff       = np.maximum(L0_at_reset - K, 0) * delta_0
caplet_price = P(0, T_1) * payoff.mean()
```

Should agree with Black-76 to within $\pm 2$ MC standard errors (the caplet repricing test).

### Bermudan payer swaption (LSMC)

Holder can exercise at $T_{e_1}, \dots, T_{e_K}$. Need an exercise policy → discover it from paths.

**Algorithm (backward in time):**

```
For each exercise date T_e, walking BACKWARD:
  1. immediate[p]  = PV at T_e of remaining payer swap on path p
                   = Σ δ_i · (L_i(T_e) − K) · P(T_e, T_{i+1})  for i ≥ k in swap_settle_idxs
  2. future_cf[p]  = discounted PV from later (already-decided) exercises
  3. ITM filter:    keep paths where immediate > 0
  4. Regress future_cf on [1, immediate, immediate²] (Longstaff-Schwartz basis)
  5. cont_value[p] = β₀ + β₁·immediate[p] + β₂·immediate[p]²
  6. Exercise if immediate[p] > cont_value[p]; else carry future_cf

Final: discount each path's exercised PV to t = 0; average → price.
```

**Worked LSMC example (4 paths, 2 exercise dates, $K = 0.04$):**

| Path | Immediate at $T_{e_1}$ | Immediate at $T_{e_2}$ |
|---|---|---|
| 1 | 0.10 | 0.16 |
| 2 | 0.04 | 0.02 |
| 3 | −0.02 | −0.04 |
| 4 | 0.12 | 0.08 |

**At $T_{e_2}$** (last exercise — no waiting beyond): exercise iff immediate > 0.

| Path | PV at $T_{e_2}$ |
|---|---|
| 1 | 0.16 |
| 2 | 0.02 |
| 3 | 0 |
| 4 | 0.08 |

**At $T_{e_1}$:** with $P(T_{e_1}, T_{e_2}) = 0.95$, build `future_cf`:

| Path | Immediate | Future CF | ITM? |
|---|---|---|---|
| 1 | 0.10 | 0.152 | ✓ |
| 2 | 0.04 | 0.019 | ✓ |
| 3 | −0.02 | 0 | ✗ |
| 4 | 0.12 | 0.076 | ✓ |

Regress on ITM paths → say $\widehat{\text{cont}}(x) = 0.05 + 0.6x + 1.2x^2$:

| Path | Immediate | $\widehat{\text{cont}}$ | Decision |
|---|---|---|---|
| 1 | 0.10 | 0.122 | wait |
| 2 | 0.04 | 0.076 | wait |
| 4 | 0.12 | 0.139 | wait |

All three wait → flow $T_{e_2}$ PVs through to $t = 0$. Final price = mean of discounted PVs.

---

## LMM vs Hull-White cross-check

| Property | LMM | Hull-White |
|---|---|---|
| Factors | Multi-factor (one per forward) | Single-factor |
| Forward decorrelation | Yes ($\rho_{ij} < 1$) | No (all forwards co-move) |
| Bermudan price | Slightly lower (decorrelation eats some optionality) | Slightly higher |
| Typical gap | 5–10% | — |
| Red flag | Gap > 15% → calibration or LSMC basis issue | — |

---

## Quick-reference formulas

| Concept | Formula |
|---|---|
| Forward from DFs | $L_i(0) = \frac{1}{\delta_i}(\frac{P(0,T_i)}{P(0,T_{i+1})} - 1)$ |
| Cap-vol bootstrap | $\sigma_i^2 (T_i - T_{i-1}) = (\sigma_i^{\mathrm{Black}})^2 T_i - \sum_{k=0}^{i-1} \sigma_k^2 (T_k - T_{k-1})$ |
| Rebonato | $\rho_{ij} = \exp(-\beta   |T_i - T_j|)$ |
| Terminal-measure drift | $\mu_i = -\sigma_i\sum_{j>i}\frac{\delta_j L_j}{1+\delta_j L_j} \sigma_j \rho_{ij}$ |
| Euler update | $L_i(t+\Delta t) = L_i(t)\exp((\mu_i - \tfrac{1}{2}\sigma_i^2)\Delta t + \sigma_i\sqrt{\Delta t} Z_i)$ |
| Caplet | $\delta_i P(0,T_{i+1})\cdot\mathbb{E}[\max(L_i(T_i) - K, 0)]$ |
| LSMC basis | $\widehat{\text{cont}}(x) = \beta_0 + \beta_1 x + \beta_2 x^2$, ITM only |
