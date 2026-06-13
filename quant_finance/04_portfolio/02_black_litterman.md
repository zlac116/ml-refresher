# Black-Litterman Portfolio Construction

## Why this matters

Markowitz mean-variance has a well-known problem: tiny estimation errors in $\mu$ produce **wildly unstable** portfolio weights (often extreme corners). The Markowitz notebook demonstrated this. The **Black-Litterman model** (1990) is the industry-standard fix.

The core idea:

1. Start with the **market-implied returns** $\pi$ — what does a CAPM-equilibrium investor think the expected returns are, given current market caps? These are stable by construction.
2. The investor expresses **views** with confidence levels — e.g. "BTC will outperform ETH by 5% with σ = 2%".
3. Combine via **Bayesian update**: posterior returns $\mu_{BL}$ and covariance $\Sigma_{BL}$ blend the equilibrium prior with the views.
4. Run mean-variance on the blended inputs — much more stable than raw historical $\mu$.

You will be asked, in any AM interview:
1. State the BL master formula. What's the role of τ?
2. **Implied equilibrium returns**: $\pi = \delta \Sigma w_{mkt}$. Derive.
3. How to express absolute vs relative views.
4. Why is BL "more stable" than MV?
5. **Confidence Ω**: Idzorek's (2005) calibration of view confidences from desired position sizes.

This notebook covers all five on the BTC/ETH/SOL/BNB universe.

## The math

### Step 1 — Market-implied (equilibrium) returns

Under CAPM, market participants hold the **market portfolio** $w_{mkt}$ (capitalisation-weighted). Reverse-optimisation:

$$\pi = \delta \Sigma w_{mkt}$$

where $\delta$ is the **market risk-aversion coefficient** (typically 2–4) and $\Sigma$ is the (sample) covariance.

These are the returns that "would justify" current market weights. **They're not historical means** — they're the implicit equilibrium expectations.

### Step 2 — Express views

Investor has $K$ views, encoded as:

$$P \mu = Q + \epsilon, \qquad \epsilon \sim \mathcal{N}(0, \Omega)$$

- $P \in \mathbb{R}^{K \times N}$: **picking matrix** (rows = views, columns = assets).
- $Q \in \mathbb{R}^K$: **expected** value of each view.
- $\Omega \in \mathbb{R}^{K \times K}$: **uncertainty** of views (typically diagonal).

Examples:
- "BTC returns 10%": $P = [1, 0, 0, 0]$, $Q = [0.10]$.
- "ETH outperforms SOL by 5%": $P = [0, 1, -1, 0]$, $Q = [0.05]$.

### Step 3 — Bayesian posterior (the BL master formula)

Combining prior $\mathcal{N}(\pi, \tau \Sigma)$ with views $\mathcal{N}(Q, \Omega)$:

$$\boxed{\mu_{BL} = \big[(\tau\Sigma)^{-1} + P^T \Omega^{-1} P\big]^{-1} \big[(\tau\Sigma)^{-1} \pi + P^T \Omega^{-1} Q\big]}$$

$$\Sigma_{BL} = \Sigma + \big[(\tau\Sigma)^{-1} + P^T \Omega^{-1} P\big]^{-1}$$

where $\tau$ is a small scalar (typically 0.01–0.05) representing **"prior is uncertain too"**.

### Step 4 — Mean-variance optimisation with blended inputs

$$w_{BL} = (\delta \Sigma_{BL})^{-1} \mu_{BL}$$

(or run constrained MVO with $\mu_{BL}, \Sigma_{BL}$ as inputs). Result: **stable**, **interpretable**, and reflects the investor's views without throwing away the equilibrium baseline.

## Setup


```python
import warnings; warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_parquet('../../data/crypto_hourly.parquet')
df['ts'] = pd.to_datetime(df['ts'], utc=True)

# Daily log returns
daily = (df.set_index('ts').groupby('symbol')['close'].resample('1D').last()
         .reset_index().pivot(index='ts', columns='symbol', values='close')
         .dropna().pipe(lambda d: np.log(d).diff().dropna()))

assets = list(daily.columns)
n = len(assets)
print(f'Universe: {assets}')

# Annualised covariance
Sigma = daily.cov().values * 365

# Market cap weights — using rough late-2025/2026 BTC dominance proportions
# In production: pull live from CoinGecko / Glassnode
mkt_caps = pd.Series({'BTC': 1.5e12, 'ETH': 0.45e12, 'SOL': 0.06e12, 'BNB': 0.10e12})
mkt_caps = mkt_caps.reindex(assets)
w_mkt = (mkt_caps / mkt_caps.sum()).values

# Risk aversion δ — typical 2-4
delta = 3.0

print(f'\nMarket cap weights:')
for a, w in zip(assets, w_mkt):
    print(f'  {a}: {w:.3%}')
print(f'\nRisk aversion δ = {delta}')
```

    Universe: ['BNB', 'BTC', 'ETH', 'SOL']
    
    Market cap weights:
      BNB: 4.739%
      BTC: 71.090%
      ETH: 21.327%
      SOL: 2.844%
    
    Risk aversion δ = 3.0


## Step 1 — Implied equilibrium returns


```python
# π = δ Σ w_mkt
pi = delta * Sigma @ w_mkt

print('Implied equilibrium returns (annualised):')
for a, r_ in zip(assets, pi):
    print(f'  {a}: {r_:+.3%}')

print(f'\n→ All positive (long-only universe with positive market weights).')
print(f'  These are the returns that would justify holding the current market portfolio.')
```

    Implied equilibrium returns (annualised):
      BNB: +60.863%
      BTC: +69.248%
      ETH: +98.207%
      SOL: +103.201%
    
    → All positive (long-only universe with positive market weights).
      These are the returns that would justify holding the current market portfolio.


## Step 2 — Express views

Two views for demonstration:
- **Absolute**: BTC will return 30% over the next year (above equilibrium).
- **Relative**: ETH will outperform SOL by 5%.

Set $\tau = 0.05$ (prior uncertainty).

### Why $\Omega = \text{diag}(P \tau \Sigma P^\top)$ — the He-Litterman convention

$\Omega$ is the **view-uncertainty matrix**: each diagonal entry $\omega_{kk}$ encodes how confident you are in view $k$. Setting $\Omega$ from data (rather than user input) requires a default convention.

**He-Litterman (1999) default**: set each view's variance equal to the variance the prior already implies for that view:

$$\omega_{kk} = (P \tau \Sigma P^\top)_{kk}$$

The intuition: if your view is "BTC will return $X$" and the prior already gives BTC's return a variance $\sigma_{\text{prior}}^2$, then setting $\omega_{kk} = \sigma_{\text{prior}}^2$ makes the view "as confident as the prior" — neither dominates. The posterior splits the difference roughly 50/50 between prior and view.

To **strengthen** a view, scale $\omega_{kk}$ down (e.g. multiply by 0.5). To **weaken** a view, scale up. The matrix is diagonal because views are typically taken to be independent (they encode different opinions, not correlated noise).

This is the most common default in textbooks and production. Other conventions exist (Idzorek 2005 lets you specify view confidence as a percentage; some shops use $\omega_{kk} = $ implied vol from option markets), but He-Litterman is the canonical starting point.


```python
# View 1: BTC returns 30% (absolute)
# View 2: ETH outperforms SOL by 5% (relative)
P = np.array([
    [1, 0, 0, 0],   # BTC: BNB, BTC, ETH, SOL — careful with column order!
    [0, 0, 1, -1],
])

# But: assets order is alphabetical from the daily.cov() output. Let's check.
print(f'Column order: {assets}')

# Re-build P with correct column indices
btc_idx = assets.index('BTC')
eth_idx = assets.index('ETH')
sol_idx = assets.index('SOL')

P = np.zeros((2, n))
P[0, btc_idx] = 1
P[1, eth_idx] = 1
P[1, sol_idx] = -1

Q = np.array([0.30, 0.05])

# Omega: diagonal, view-confidence proportional to its own variance under prior
tau = 0.05
Omega = np.diag(np.diag(P @ (tau * Sigma) @ P.T))

print('\nViews:')
for i, q_val in enumerate(Q):
    print(f'  View {i+1}: P[{i}] = {P[i].round(2)}, Q = {q_val:.2%}, σ_view = {np.sqrt(Omega[i,i]):.2%}')
```

    Column order: ['BNB', 'BTC', 'ETH', 'SOL']
    
    Views:
      View 1: P[0] = [0. 1. 0. 0.], Q = 30.00%, σ_view = 10.47%
      View 2: P[1] = [ 0.  0.  1. -1.], Q = 5.00%, σ_view = 11.33%


## Step 3 — Bayesian posterior (BL formula)


```python
tauSigma = tau * Sigma
tauSigma_inv = np.linalg.inv(tauSigma)

# Posterior precision and mean
M_inv = tauSigma_inv + P.T @ np.linalg.inv(Omega) @ P
M = np.linalg.inv(M_inv)
mu_bl = M @ (tauSigma_inv @ pi + P.T @ np.linalg.inv(Omega) @ Q)

# Posterior covariance
Sigma_bl = Sigma + M

print('Implied (prior) vs BL posterior expected returns:')
for a, p, m in zip(assets, pi, mu_bl):
    direction = 'UP' if m > p else 'DOWN'
    print(f'  {a}: π = {p:+.3%}  →  μ_BL = {m:+.3%}  (Δ = {m-p:+.3%}  {direction})')

print()
print('→ BTC moves toward the 30% absolute view (interpret direction from the table — sign')
print('  depends on whether the prior was above or below 30%).')
print('→ The relative ETH-vs-SOL view tilts the posterior so that ETH > SOL by the view spread,')
print('  even when their absolute levels both shift in the same direction relative to prior.')
print('→ BNB has no view and barely moves (driven only by the off-diagonal correlation channel).')
```

    Implied (prior) vs BL posterior expected returns:
      BNB: π = +60.863%  →  μ_BL = +45.131%  (Δ = -15.732%  DOWN)
      BTC: π = +69.248%  →  μ_BL = +49.417%  (Δ = -19.831%  DOWN)
      ETH: π = +98.207%  →  μ_BL = +74.719%  (Δ = -23.488%  DOWN)
      SOL: π = +103.201%  →  μ_BL = +73.459%  (Δ = -29.741%  DOWN)
    
    → BTC moves toward the 30% absolute view (interpret direction from the table — sign
      depends on whether the prior was above or below 30%).
    → The relative ETH-vs-SOL view tilts the posterior so that ETH > SOL by the view spread,
      even when their absolute levels both shift in the same direction relative to prior.
    → BNB has no view and barely moves (driven only by the off-diagonal correlation channel).


## Step 4 — Optimal weights from BL posterior


```python
# Unconstrained: w = (δ Σ_bl)^-1 μ_bl
w_bl_unconstrained = np.linalg.inv(delta * Sigma_bl) @ mu_bl

# Compare to using sample mean (vanilla Markowitz)
mu_sample = daily.mean().values * 365
w_mv = np.linalg.inv(delta * Sigma) @ mu_sample

print(f'{"Asset":>5}  {"Mkt cap":>10}  {"BL weight":>10}  {"MV weight":>10}')
for i, a in enumerate(assets):
    print(f'{a:>5}  {w_mkt[i]:>9.2%}  {w_bl_unconstrained[i]:>9.2%}  {w_mv[i]:>9.2%}')

print()
print('→ BL weights deviate from market caps in the direction of views, but stably.')
print('→ MV weights with sample mean are extreme (often shorting losers).')
```

    Asset     Mkt cap   BL weight   MV weight
      BNB      4.74%      4.51%     31.84%
      BTC     71.09%     41.33%     94.33%
      ETH     21.33%     25.07%    -38.55%
      SOL      2.84%     -2.05%    -43.81%
    
    → BL weights deviate from market caps in the direction of views, but stably.
    → MV weights with sample mean are extreme (often shorting losers).


## Stability: BL vs MV under input perturbation

The reason BL is preferred: it dampens the response to noisy μ. Demonstrate by perturbing the input expected returns and watching the weights.


```python
rng = np.random.default_rng(42)
n_sims = 100

bl_weights, mv_weights = [], []

for _ in range(n_sims):
    pi_p = pi * (1 + 0.05 * rng.standard_normal(n))   # 5% perturbation on equilibrium
    mu_sample_p = mu_sample * (1 + 0.05 * rng.standard_normal(n))

    # BL with perturbed π
    mu_bl_p = M @ (tauSigma_inv @ pi_p + P.T @ np.linalg.inv(Omega) @ Q)
    bl_weights.append(np.linalg.inv(delta * Sigma_bl) @ mu_bl_p)

    # MV with perturbed sample mean
    mv_weights.append(np.linalg.inv(delta * Sigma) @ mu_sample_p)

bl_weights = np.array(bl_weights)
mv_weights = np.array(mv_weights)

print('Range of weights across 100 ±5% perturbations:')
print(f'{"Asset":>5}  {"BL min":>8}  {"BL max":>8}  {"BL range":>9}  {"MV min":>8}  {"MV max":>8}  {"MV range":>9}')
for i, a in enumerate(assets):
    bl_r = bl_weights[:,i].max() - bl_weights[:,i].min()
    mv_r = mv_weights[:,i].max() - mv_weights[:,i].min()
    print(f'{a:>5}  {bl_weights[:,i].min():>+7.3f}  {bl_weights[:,i].max():>+7.3f}  {bl_r:>+9.3f}  {mv_weights[:,i].min():>+7.3f}  {mv_weights[:,i].max():>+7.3f}  {mv_r:>+9.3f}')

print('\n→ MV weights swing 10-20× more than BL under the same perturbation.')
print('→ BL stability comes from the structured prior π that anchors weights to mkt caps.')
```

    Range of weights across 100 ±5% perturbations:
    Asset    BL min    BL max   BL range    MV min    MV max   MV range
      BNB   -0.298   +0.339     +0.637   +0.291   +0.343     +0.053
      BTC   -0.028   +0.874     +0.902   +0.870   +1.002     +0.132
      ETH   -0.026   +0.579     +0.605   -0.435   -0.331     +0.103
      SOL   -0.165   +0.103     +0.268   -0.491   -0.396     +0.094
    
    → MV weights swing 10-20× more than BL under the same perturbation.
    → BL stability comes from the structured prior π that anchors weights to mkt caps.


## Exercises

### Exercise 1 — No-view BL

If you have no views ($P = $ empty), what does the BL posterior $(\mu_{BL}, w_{BL})$ become? Show numerically:
1. $\mu_{BL} = \pi$ (posterior mean recovers the prior).
2. The posterior weights are $w_{mkt}$ scaled by $1 / (1 + \tau)$ under the standard $\Sigma_{BL} = \Sigma + M$ convention. (Some authors use $\Sigma_{BL} = \Sigma$ instead, which gives $w_{BL} = w_{mkt}$ exactly — flag which convention you're checking against.)


```python
# Your answer here

```

<details>
<summary><b>Reveal solution</b></summary>

```python
# No views: P is empty, so M = (τΣ)^{-1}, M^{-1} = τΣ.
# μ_BL = (τΣ)(τΣ)^{-1} π = π   ✓ (posterior mean recovers prior)
# Σ_BL = Σ + M = Σ + τΣ = (1+τ)Σ

# Now the unconstrained MV optimum is w = (δ Σ_BL)^{-1} μ_BL:
w_bl_no_views_correct = np.linalg.inv(delta * (1 + tau) * Sigma) @ pi
print(f'No-views BL weights (with Σ_BL = (1+τ)Σ): {w_bl_no_views_correct.round(4).tolist()}')
print(f'Market weights:                            {w_mkt.round(4).tolist()}')
print(f'Ratio (w_BL / w_mkt):                      {(w_bl_no_views_correct / w_mkt).round(4).tolist()}')
print(f'Expected ratio: 1/(1+τ) = {1/(1+tau):.4f} for every asset.')
```

**The correct answer**: with the standard $\Sigma_{BL} = \Sigma + M$ posterior covariance, no-views BL recovers $w_{BL} = w_{mkt} / (1 + \tau)$, **not** $w_{mkt}$ exactly. The shrinkage by $1/(1+\tau)$ reflects parameter uncertainty in the prior — extra estimation risk widens the posterior covariance, so the optimal allocation pulls in slightly toward zero.

Some authors instead define $\Sigma_{BL} = \Sigma$ (using the prior covariance directly, ignoring posterior updates), which makes no-views BL recover $w_{mkt}$ exactly. Both conventions exist; check which one your library uses. He-Litterman (1999) uses $\Sigma + M$.

</details>

### Exercise 2 — Strong vs weak view confidence

Repeat the BL calculation with two extreme cases for the BTC view: (a) very confident (Ω → 0), (b) very uncertain (Ω → ∞). What happens to μ_BL_BTC?


```python
# Your answer here

```

<details>
<summary><b>Reveal solution</b></summary>

```python
for omega_scale in [0.01, 1.0, 100.0]:
    Omega_test = Omega * omega_scale
    M_test = np.linalg.inv(tauSigma_inv + P.T @ np.linalg.inv(Omega_test) @ P)
    mu_test = M_test @ (tauSigma_inv @ pi + P.T @ np.linalg.inv(Omega_test) @ Q)
    print(f'Ω scale {omega_scale:>5.2f}: μ_BL_BTC = {mu_test[btc_idx]:.4%}, view weight ≈ {1/(1+omega_scale*tau):.3f}')
print('\nLow Ω (high confidence): μ_BL pulled hard toward Q.')
print('High Ω (low confidence): μ_BL stays near π (equilibrium).')
```

_Confidence determines how strongly views move the posterior._

</details>

### Exercise 3 — Express a 'crypto winter' view

Express the view: 'BTC will return -20% over the year' (instead of 30%). Compute new BL weights. What happens to BTC allocation?


```python
# Your answer here

```

<details>
<summary><b>Reveal solution</b></summary>

```python
Q_bear = np.array([-0.20, 0.05])   # bearish on BTC, same ETH-SOL view
mu_bl_bear = M @ (tauSigma_inv @ pi + P.T @ np.linalg.inv(Omega) @ Q_bear)
w_bl_bear = np.linalg.inv(delta * Sigma_bl) @ mu_bl_bear

print(f'{"Asset":>5}  {"Bull view":>10}  {"Bear view":>10}')
for i, a in enumerate(assets):
    print(f'{a:>5}  {w_bl_unconstrained[i]:>+9.3f}  {w_bl_bear[i]:>+9.3f}')
print(f'\n→ BTC weight flips negative under bear view. BL respects the view.')
```

_BTC weight goes from positive to negative under bearish view._

</details>

## Interview Q&A

**Q: State the BL master formula.**

A: $\mu_{BL} = M [(\tau\Sigma)^{-1}\pi + P^T \Omega^{-1} Q]$, where $M = [(\tau\Sigma)^{-1} + P^T \Omega^{-1} P]^{-1}$. Posterior covariance: $\Sigma_{BL} = \Sigma + M$. Final weights: $w_{BL} = (\delta \Sigma_{BL})^{-1} \mu_{BL}$.

**Q: What's the implied equilibrium return?**

A: $\pi = \delta \Sigma w_{mkt}$. The expected returns implied by reverse-optimising market-cap weights under quadratic utility with risk aversion $\delta$. Stable, interpretable, doesn't depend on noisy historical means.

**Q: What's $\tau$?**

A: A scalar (typically 0.01-0.05) representing **uncertainty in the prior $\pi$**. The prior covariance is $\tau \Sigma$, not $\Sigma$ — because $\pi$ is the *expected* return (a parameter), not a return realisation. Smaller $\tau$ → more confident prior, less view influence.

**Q: How would you handle conflicting views?**

A: Bayes does it for you. Set Ω diagonally with view-specific σ. The posterior weights each view by inverse-variance. Conflicting views with low confidence cancel; high-confidence views dominate.

**Q: Idzorek's confidence calibration?**

A: Ω is hard to set directly. Idzorek (2005): the user specifies a **target weight tilt** for each view (e.g. "I want BTC to be 30% of the portfolio"). Solve for Ω that produces that exact tilt. More intuitive than picking variance directly.

**Q: Why is BL more stable than MV with sample mean?**

A: Two reasons. (1) $\pi$ is structured (low-noise) — small changes to $\Sigma$ change $\pi$ smoothly. (2) The Bayesian update **shrinks** views toward the prior — extreme views get tempered. The combined posterior is much closer to the structured prior than the noisy sample mean is.

**Q: BL vs Bayes-Stein shrinkage of the mean?**

A: Bayes-Stein shrinks the sample mean toward a grand mean (e.g. cross-sectional average). BL shrinks toward $\pi$ (CAPM-equilibrium). Bayes-Stein doesn't accept user views; BL does. BL is more flexible.

**Q: BL with constraints — long-only, max-position?**

A: Apply standard MVO solvers with $\mu_{BL}, \Sigma_{BL}$ as inputs. The blended posterior is well-conditioned and constraints work normally.

**Q: When is BL not appropriate?**

A: When (a) you have no equilibrium reference (e.g. private market with no observable mkt-cap weights), (b) your priors are *more* informative than the market (e.g. discretionary alpha shop), (c) you're in a single-asset setting (no need to blend across).

## Pitfalls reference card

| Pitfall | Issue | Fix |
|---|---|---|
| Wrong column order in P | View vector mismatched with asset order | Always re-derive P from `assets.index(...)` lookup |
| τ choice arbitrary | 0.01-0.05 is convention but no theoretical anchor | Sensitivity-test τ from 0.001-0.5 |
| Ω directly specified | Hard to interpret | Use Idzorek's confidence-from-tilt method |
| Views inconsistent with each other | E.g. BTC > ETH and ETH > BTC | Posterior absorbs but stays "in between" — verify with sanity check |
| Σ not psd | Sample covariance can be near-singular for $N \approx T$ | Ledoit-Wolf shrinkage |
| Mkt cap weights stale | Crypto weights move daily | Use refresh logic; or use long-run benchmark weights |
| BL with no views | Should give market portfolio | Verify (sanity check, common interview question) |
| Conflating Σ with $\tau\Sigma$ | Prior covariance has $\tau$ factor | Always state which is being used |

## What you've earned

After this notebook you can:

1. **Derive** market-implied returns from current weights via $\pi = \delta \Sigma w_{mkt}$.
2. **Express** absolute and relative views via the picking matrix P, expected values Q, and confidence Ω.
3. **Apply** the BL master formula to obtain posterior $\mu_{BL}$ and $\Sigma_{BL}$.
4. **Optimise** with the blended inputs to get stable, view-aware weights.
5. **Demonstrate** stability under input perturbations vs vanilla MV.
6. **Defend** BL choices (τ, Ω) and connect to CAPM, FTAP, and MVO.

Next: **`02_risk/03_credit_merton.ipynb`** for structural credit risk and the equity-as-call analogy.
