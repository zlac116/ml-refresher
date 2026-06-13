# Itô's Lemma Applied

## Why this matters

**Itô's lemma is the chain rule of stochastic calculus.** Without it, you can't derive the BS PDE, you can't price options, you can't construct hedges. It's also the most-tested topic at quant interviews.

You will be asked:
1. State Itô's lemma in 1D and multi-D.
2. Apply Itô to: $\ln S$ (gives GBM solution), $S^2$, $\sqrt{S}$, $f(t, S)$.
3. **Why does Itô have a $\tfrac{1}{2}\sigma^2 \partial_{SS}$ term that ordinary calculus doesn't?**
4. Derive the BS PDE via Itô + delta hedging (link to notebook 01).
5. **Itô isometry** — what is it and when does it apply?

This notebook is a math-heavy companion to the BS PDE derivation.

## Itô's lemma — 1D

If $X_t$ satisfies $dX = \mu \, dt + \sigma \, dW$ and $f(t, x)$ is twice continuously differentiable in $x$ and once in $t$:

$$df(t, X_t) = \left(\partial_t f + \mu \, \partial_x f + \tfrac{1}{2}\sigma^2 \, \partial_{xx} f\right) dt + \sigma \, \partial_x f \, dW$$

**The $\tfrac{1}{2}\sigma^2 \partial_{xx}$ term is the Itô correction** — it doesn't appear in ordinary chain rule. Why?

Heuristic Taylor expansion:
$$df = \partial_t f \, dt + \partial_x f \, dX + \tfrac{1}{2}\partial_{xx} f \, (dX)^2 + \dots$$

Now, in stochastic calculus:
- $(dt)^2 = 0$
- $dt \, dW = 0$
- $(dW)^2 = dt$ ← **this is the magic!**

Substitute $dX = \mu dt + \sigma dW$:
$(dX)^2 = \sigma^2 (dW)^2 = \sigma^2 dt$

Plug back:
$df = \partial_t f \, dt + \partial_x f (\mu dt + \sigma dW) + \tfrac{1}{2} \partial_{xx} f \, \sigma^2 \, dt$

Collect dt and dW terms → Itô's lemma.

## Five canonical applications

### 1. Apply to $\ln S$ where $dS = \mu S \, dt + \sigma S \, dW$

$f(S) = \ln S$, $\partial_S f = 1/S$, $\partial_{SS} f = -1/S^2$.

$$d \ln S = \frac{1}{S}(\mu S dt + \sigma S dW) + \tfrac{1}{2}(-\tfrac{1}{S^2})(\sigma S)^2 dt = (\mu - \tfrac{1}{2}\sigma^2) dt + \sigma dW$$

Integrate: $\ln S_t = \ln S_0 + (\mu - \sigma^2/2)t + \sigma W_t$. Exponentiate: GBM solution.

### 2. Apply to $S^2$

$f(S) = S^2$, $\partial_S f = 2S$, $\partial_{SS} f = 2$.

$$d(S^2) = 2S \cdot \mu S dt + 2S \cdot \sigma S dW + \tfrac{1}{2} \cdot 2 \cdot \sigma^2 S^2 dt$$
$$= S^2 (2\mu + \sigma^2) dt + 2\sigma S^2 dW$$

The $\sigma^2$ term is the Itô correction — without it, $S^2$ would just have drift $2\mu S^2$.

### 3. Itô product rule

For two correlated Itôs $X, Y$ with $d\langle X, Y \rangle = \sigma_X \sigma_Y \rho \, dt$:

$$d(XY) = X \, dY + Y \, dX + d\langle X, Y \rangle$$

The cross-variation term is the stochastic analogue of the missing piece in $(dX + X)(dY + Y) = dXdY + ...$.

### 4. Apply to $f(t, S)$ — the BS pre-PDE

$f(t, S)$ smooth, $S$ GBM:

$$df = \partial_t f \, dt + \partial_S f \, dS + \tfrac{1}{2}\sigma^2 S^2 \, \partial_{SS} f \, dt$$

This is the PDE-derivation building block — the BS PDE follows by combining this with delta hedging.

## Itô isometry

For an Itô integral $\int_0^T \sigma_t \, dW_t$ with adapted $\sigma_t$ and $\mathbb{E}[\int_0^T \sigma_t^2 dt] < \infty$:

$$\mathbb{E}\!\left[\left(\int_0^T \sigma_t \, dW_t\right)^2\right] = \mathbb{E}\!\left[\int_0^T \sigma_t^2 \, dt\right]$$

**This is how you compute variances of stochastic integrals.** Used everywhere: variance of yields in term-structure models, variance of integrated payoffs in MC pricing, conditional Greeks.

## Setup — Itô SDE simulator + correction verification


```python
import warnings; warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Simulate GBM, then compute log(S) drift via Ito's lemma vs naive (ordinary) drift
S0, mu, sigma, T, n_steps, n_paths = 100, 0.05, 0.30, 1.0, 252, 100000
dt = T / n_steps
rng = np.random.default_rng(42)
Z = rng.standard_normal((n_paths, n_steps))

# Simulate GBM with proper drift (μ - σ²/2 in log-space)
log_returns = (mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z
log_S = np.zeros((n_paths, n_steps + 1))
log_S[:, 0] = np.log(S0); log_S[:, 1:] = np.log(S0) + np.cumsum(log_returns, axis=1)
S = np.exp(log_S)

# Verify: drift of log S in MC matches Ito derivation
log_drift_mc = (log_S[:, -1] - log_S[:, 0]).mean() / T
log_drift_theory = mu - 0.5 * sigma**2
print(f'Drift of log(S) over [0, T]:')
print(f'  MC empirical:           {log_drift_mc:+.6f}')
print(f'  Itô prediction (μ-σ²/2): {log_drift_theory:+.6f}')
print(f'  Naive (no Itô) μ:        {mu:+.6f}')
print()

# Drift of S itself (arithmetic) — should be μ
S_drift_mc = (S[:, -1] - S[:, 0]).mean() / (S0 * T)
print(f'Drift of S (arithmetic) / S₀:')
print(f'  MC: {S_drift_mc:.6f}')
print(f'  Theory (μ): {mu:.6f}')
```

    Drift of log(S) over [0, T]:
      MC empirical:           +0.005421
      Itô prediction (μ-σ²/2): +0.005000
      Naive (no Itô) μ:        +0.050000
    
    Drift of S (arithmetic) / S₀:
      MC: 0.051980
      Theory (μ): 0.050000


## Itô isometry — numerical verification


```python
# E[(int σ dW)^2] = E[int σ² dt]
# Take σ_t = σ (constant). Then int σ dW = σ W_T, and var = σ²T.
sigma_t = 0.30
T_iso = 2.0
n_paths_iso = 100000
n_steps_iso = 1000
dt_iso = T_iso / n_steps_iso
Z_iso = rng.standard_normal((n_paths_iso, n_steps_iso))
W = np.cumsum(Z_iso * np.sqrt(dt_iso), axis=1)
W = np.column_stack([np.zeros(n_paths_iso), W])

# Itô integral with constant σ: σ W_T
ito_integral = sigma_t * W[:, -1]

empirical_var = ito_integral.var()
theory_var = sigma_t**2 * T_iso

print('Itô isometry check (constant σ):')
print(f'  E[(σ W_T)²] empirical: {empirical_var:.4f}')
print(f'  E[σ² T] theory:        {theory_var:.4f}')
print(f'  agree to:              {abs(empirical_var - theory_var):.2e}')
```

    Itô isometry check (constant σ):
      E[(σ W_T)²] empirical: 0.1796
      E[σ² T] theory:        0.1800
      agree to:              4.41e-04


## Apply Itô to $S^2$ — verify the correction


```python
# Theory: d(S²) has drift S²(2μ + σ²) and diffusion 2σS²
# Equivalent: E[S_T²] = S0² e^((2μ + σ²)T)
S0_, mu_, sigma_, T_ = 100, 0.05, 0.30, 1.0
S_squared_mean = (S[:, -1]**2).mean()
theory_S_squared = S0_**2 * np.exp((2*mu_ + sigma_**2) * T_)

print(f'E[S_T²] empirical: {S_squared_mean:.4f}')
print(f'E[S_T²] theory:    {theory_S_squared:.4f}')
print()
print('→ The Itô correction adds σ² to the drift of S². Without it, you\'d miss the lognormal moment factor.')

# Naive (no Itô) prediction: E[S²] would equal (E[S])² = (S0 e^(μT))² — too small
print(f'\n(E[S_T])² = {(S0_ * np.exp(mu_ * T_))**2:.4f}')
print(f'E[S_T²]   = {S0_**2 * np.exp((2*mu_ + sigma_**2) * T_):.4f}')
print(f'Variance  = E[S²] - (E[S])² = {(S0_**2 * np.exp((2*mu_ + sigma_**2) * T_)) - (S0_ * np.exp(mu_ * T_))**2:.4f}')
```

    E[S_T²] empirical: 12119.5574
    E[S_T²] theory:    12092.4960
    
    → The Itô correction adds σ² to the drift of S². Without it, you'd miss the lognormal moment factor.
    
    (E[S_T])² = 11051.7092
    E[S_T²]   = 12092.4960
    Variance  = E[S²] - (E[S])² = 1040.7868


## Exercises

### Exercise 1 — Apply Itô to $f(S) = e^{\alpha S}$

Compute $df$ where $S$ is GBM with $\mu = 0.05, \sigma = 0.30$.


```python
# Your answer here

```

<details>
<summary><b>Reveal solution</b></summary>

```python
# f = e^(αS), df = αe^(αS) dS + (1/2) α² e^(αS) σ² S² dt
# = e^(αS) [α(μS dt + σS dW) + (α²/2) σ² S² dt]
# Drift = e^(αS) S [αμ + (α²/2) σ² S]
# Diffusion = α e^(αS) σ S
# (depends on S, so not closed form unless evaluated at fixed S)
print('df = α e^(αS) S [μ + (α/2) σ² S] dt + α e^(αS) σ S dW')
print('The second-order term shows the Itô correction explicitly.')
```

_Drift has an Itô correction proportional to α² σ² S²._

</details>

### Exercise 2 — Itô applied to $\sqrt{V}$ where $V$ follows Heston dynamics

$dV = \kappa(\theta - V) dt + \xi \sqrt V dW$. Apply Itô to $f(V) = \sqrt V$. (This is how you derive the SDE for vol from variance.)


```python
# Your answer here

```

<details>
<summary><b>Reveal solution</b></summary>

```python
# f = sqrt(V), f'(V) = 1/(2 sqrt V), f''(V) = -1/(4 V^{3/2})
# df = (1/(2 sqrt V)) dV + (1/2) (-1/(4 V^{3/2})) (ξ sqrt V)² dt
# = (1/(2 sqrt V)) [κ(θ - V) dt + ξ sqrt V dW] + (1/2)(-1/(4V^{3/2})) ξ² V dt
# = (κ(θ - V)/(2 sqrt V) - ξ²/(8 sqrt V)) dt + (ξ/2) dW
#
# Note: the noise term becomes constant (ξ/2)! Vol becomes a *bounded-noise* process.
print('d√V = [κ(θ-V)/(2√V) - ξ²/(8√V)] dt + (ξ/2) dW')
print('Key: noise on √V is CONSTANT (ξ/2), not state-dependent.')
```

_Heston's variance has state-dependent noise; vol has constant noise — this is one reason traders prefer to work in vol space._

</details>

### Exercise 3 — Compute Var(W_T²) using Itô

Apply Itô to $f(W_t) = W_t^2$, integrate, take expectation, then compute variance.


```python
# Your answer here

```

<details>
<summary><b>Reveal solution</b></summary>

```python
# d(W²) = 2W dW + dt (Itô correction)
# Integrate from 0 to T: W_T² = 2 ∫W dW + T
# Take expectation: E[W_T²] = T (since martingale)
# To get Var(W_T²): use that W_T² ~ T χ² with 1 d.o.f. (after standardising W_T/sqrt(T))
# χ² with 1 d.o.f. has variance 2, so W_T² has variance 2T²

T_test = 2.0
n_paths_v = 1000000
W_T = np.random.default_rng(42).standard_normal(n_paths_v) * np.sqrt(T_test)
emp_var = (W_T**2).var()
theory_var = 2 * T_test**2
print(f'Var(W_T²) empirical: {emp_var:.4f}')
print(f'Theory (2T²):        {theory_var:.4f}')
```

_Var(W_T²) = 2T². Verify via the χ² connection._

</details>

## Interview Q&A

**Q: State Itô's lemma.**

A: For $X$ with $dX = \mu dt + \sigma dW$ and $f(t, x) \in C^{1,2}$:
$$df = \big(\partial_t f + \mu \partial_x f + \tfrac{1}{2}\sigma^2 \partial_{xx} f\big) dt + \sigma \partial_x f \, dW$$
The $\tfrac{1}{2}\sigma^2 \partial_{xx}$ is the Itô correction.

**Q: Why does Itô have a 2nd-order term?**

A: $(dW)^2 = dt$ in the limit. Stochastic increments scale as $\sqrt{dt}$, not $dt$, so $(dX)^2$ contributes at order $dt$ — same order as the drift. Ordinary calculus has $(dX)^2 = O(dt^2) = 0$, so no correction.

**Q: Apply Itô to $\ln S$ for $dS = \mu S dt + \sigma S dW$.**

A: $f = \ln S$, $f_S = 1/S$, $f_{SS} = -1/S^2$. So $d\ln S = (\mu - \sigma^2/2) dt + \sigma dW$. Integrate: $\ln S_t = \ln S_0 + (\mu - \sigma^2/2)t + \sigma W_t$. This gives the GBM solution.

**Q: State Itô isometry.**

A: $\mathbb{E}[(\int_0^T \sigma_t dW_t)^2] = \mathbb{E}[\int_0^T \sigma_t^2 dt]$ for adapted $\sigma_t$. The LHS is the variance of the Itô integral; the RHS lets you compute it without sampling.

**Q: Multi-D Itô?**

A: For $f(t, X_1, ..., X_n)$ with each $dX_i = \mu_i dt + \sigma_i dW_i$ and cross-variations $d\langle W_i, W_j \rangle = \rho_{ij} dt$:
$$df = \partial_t f dt + \sum_i \partial_i f \, dX_i + \tfrac{1}{2} \sum_{i,j} \partial_{ij} f \, \sigma_i \sigma_j \rho_{ij} dt$$

The cross-second-order terms appear in basket / spread option pricing.

**Q: Itô vs Stratonovich?**

A: Itô integral evaluates the integrand at the *left endpoint* of the partition: $\int f(W) dW \approx \sum f(W_{t_i}) (W_{t_{i+1}} - W_{t_i})$. **Adapted, martingale property**, but no chain rule. Stratonovich evaluates at the midpoint and **does** satisfy ordinary chain rule, but lacks martingale property. Quant finance uses Itô — the martingale property is what enables risk-neutral pricing.

## Pitfalls

| Pitfall | Issue | Fix |
|---|---|---|
| Forgetting the Itô correction | Naive chain rule gives wrong drift | Always include $\tfrac{1}{2}\sigma^2 \partial_{xx}$ |
| Confusing arithmetic vs geometric mean | $E[S_t]$ vs $E[\ln S_t]$ | $E[S_t] = S_0 e^{\mu t}$, but $\ln$ has drift $\mu - \sigma^2/2$ |
| Itô in physical vs risk-neutral | Different drifts | State the measure |
| Stratonovich confusion | Different formula | Quant finance always uses Itô |
| Cross-variation | Need $d\langle W_i, W_j\rangle = \rho_{ij} dt$ | Don't forget for correlated SDEs |

## What you've earned

After this notebook you can:

1. **State** Itô's lemma in 1D and multi-D and explain the $\tfrac{1}{2}\sigma^2$ correction.
2. **Apply** Itô to $\ln S$, $S^2$, $\sqrt V$, $f(t, S)$ — all standard derivations.
3. **State and apply** Itô isometry for variance of stochastic integrals.
4. **Distinguish** Itô from Stratonovich.
5. **Connect** Itô to the BS PDE derivation in notebook 01.

This completes the **stochastic calculus foundations**. Next: GARCH-family vol models in `05_volatility/01_garch.ipynb`.
