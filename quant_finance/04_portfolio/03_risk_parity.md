# Risk Parity (Equal Risk Contribution)

## Why this matters

Markowitz allocates *capital* to maximise return for a given variance. **Risk parity** allocates *risk*: each asset contributes equally to portfolio variance. Bridgewater's All Weather, AQR's Risk Parity Fund, and many institutional allocators use it.

You will be asked:
1. Define the risk contribution of asset $i$.
2. State the Equal Risk Contribution (ERC) condition.
3. Algorithm: how do you solve for ERC weights?
4. **Levered risk parity** — why is leverage required?
5. Risk parity vs MV vs 1/N — when does each apply?

This notebook covers all five on the BTC/ETH/SOL/BNB universe.

## Risk contribution

For weight vector $w$, portfolio variance is $\sigma_P^2 = w^T \Sigma w$. The **marginal risk contribution** of asset $i$:

$$\frac{\partial \sigma_P}{\partial w_i} = \frac{(\Sigma w)_i}{\sigma_P}$$

The **risk contribution** (RC):

$$\text{RC}_i = w_i \cdot (\Sigma w)_i / \sigma_P$$

Sum: $\sum \text{RC}_i = \sigma_P$ (Euler decomposition of variance).

**Equal risk contribution (ERC)** condition:

$$\text{RC}_i = \sigma_P / N \quad \text{for all } i$$

Equivalently: $w_i (\Sigma w)_i = w_j (\Sigma w)_j$ for all $i, j$.

### Algorithm — Newton or coordinate descent

ERC has no closed form. Standard solver: minimise $\sum_i (\text{RC}_i - \sigma_P/N)^2$ subject to $\sum w_i = 1, w_i \ge 0$.

Or use **cyclical coordinate descent**: iteratively update each $w_i$ to balance its risk contribution. Spinu (2013) and Maillard-Roncalli-Teiletche (2010) are standard references.

## Setup


```python
import warnings; warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

df = pd.read_parquet('../../data/crypto_hourly.parquet')
df['ts'] = pd.to_datetime(df['ts'], utc=True)

daily = (df.set_index('ts').groupby('symbol')['close'].resample('1D').last()
         .reset_index().pivot(index='ts', columns='symbol', values='close')
         .dropna().pipe(lambda d: np.log(d).diff().dropna()))

assets = list(daily.columns)
n = len(assets)
Sigma = daily.cov().values * 365

print(f'Universe: {assets}')
print(f'Annualised vols: {np.sqrt(np.diag(Sigma)).round(3)}')
```

    Universe: ['BNB', 'BTC', 'ETH', 'SOL']
    Annualised vols: [0.52  0.468 0.71  0.814]


## ERC solver


```python
def risk_contributions(w, Sigma):
    sigma_P = np.sqrt(w @ Sigma @ w)
    return w * (Sigma @ w) / sigma_P


def erc_objective(w, Sigma):
    rc = risk_contributions(w, Sigma)
    target = np.full_like(rc, rc.mean())
    return np.sum((rc - target)**2)


def solve_erc(Sigma):
    n = Sigma.shape[0]
    constraints = [{'type': 'eq', 'fun': lambda w: w.sum() - 1}]
    bounds = [(1e-6, 1.0) for _ in range(n)]
    res = minimize(erc_objective, x0=np.ones(n)/n, args=(Sigma,),
                   constraints=constraints, bounds=bounds, method='SLSQP',
                   options={'ftol': 1e-12, 'maxiter': 1000})
    return res.x


w_erc = solve_erc(Sigma)
rc_erc = risk_contributions(w_erc, Sigma)
sigma_erc = np.sqrt(w_erc @ Sigma @ w_erc)

print(f'\nERC weights and risk contributions:')
for a, w_, rc_ in zip(assets, w_erc, rc_erc):
    print(f'  {a}: w = {w_:.4f}, RC = {rc_:.4f}  ({rc_/sigma_erc:.2%} of total risk)')
print(f'\nPortfolio vol (annualised): {sigma_erc:.4f}  ({sigma_erc*100:.2f}%)')
```

    
    ERC weights and risk contributions:
      BNB: w = 0.2974, RC = 0.1348  (25.00% of total risk)
      BTC: w = 0.3135, RC = 0.1348  (25.00% of total risk)
      ETH: w = 0.2057, RC = 0.1348  (25.00% of total risk)
      SOL: w = 0.1834, RC = 0.1348  (25.00% of total risk)
    
    Portfolio vol (annualised): 0.5393  (53.93%)


## Compare ERC, 1/N, and minimum-variance


```python
# 1/N
w_eq = np.ones(n) / n
rc_eq = risk_contributions(w_eq, Sigma)

# Min-var
ones = np.ones(n)
Sigma_inv = np.linalg.inv(Sigma)
w_mv = Sigma_inv @ ones / (ones @ Sigma_inv @ ones)
rc_mv = risk_contributions(w_mv, Sigma)

comp = pd.DataFrame({
    'ERC weight': w_erc,
    '1/N weight': w_eq,
    'Min-var weight': w_mv,
    'ERC RC fraction': rc_erc / rc_erc.sum(),
    '1/N RC fraction': rc_eq / rc_eq.sum(),
    'MV RC fraction':  rc_mv / rc_mv.sum(),
}, index=assets)

print(comp.round(4).to_string())
print()
print('→ ERC has equal RC fractions (~25% each).')
print('→ 1/N puts equal capital but UNEQUAL risk (high-vol assets dominate).')
print('→ Min-var minimises total vol but concentrates capital.')
```

         ERC weight  1/N weight  Min-var weight  ERC RC fraction  1/N RC fraction  MV RC fraction
    BNB      0.2974        0.25          0.4659             0.25           0.1943          0.4659
    BTC      0.3135        0.25          1.0652             0.25           0.1864          1.0652
    ETH      0.2057        0.25         -0.2758             0.25           0.2896         -0.2758
    SOL      0.1834        0.25         -0.2552             0.25           0.3297         -0.2552
    
    → ERC has equal RC fractions (~25% each).
    → 1/N puts equal capital but UNEQUAL risk (high-vol assets dominate).
    → Min-var minimises total vol but concentrates capital.


## Scaling risk parity to a target volatility

Risk parity portfolios target **equal risk contribution**, not a specific overall risk level. To convert ERC into a tradeable strategy at a chosen target vol $\sigma^*$, scale weights by $\sigma^* / \sigma_{ERC}$:

$$w^* = \frac{\sigma^*}{\sigma_{ERC}} \cdot w_{ERC}$$

Two regimes:

- **Lever up** ($\sigma^* > \sigma_{ERC}$): total weight > 1, requires borrowing. Classic Bridgewater **All Weather** application — equity-bond risk parity has $\sigma_{ERC} \approx 6\text{-}8\%$, levered to ~10-12% target.
- **Lever down** ($\sigma^* < \sigma_{ERC}$): total weight < 1, the rest is held in cash. Common when the underlying universe is volatile (e.g. crypto, where ERC vol can exceed 50%).

The crypto example below uses $\sigma^* = 40\%$ vs $\sigma_{ERC} \approx 54\%$, giving a *deleveraging* factor of ~0.74 (cash buffer of ~26%). To demonstrate leverage instead, set $\sigma^* > \sigma_{ERC}$.


```python
sigma_target = 0.40   # 40% target — below crypto ERC vol of ~54%, so this is DE-leveraging
leverage = sigma_target / sigma_erc
w_levered = w_erc * leverage

print(f'ERC vol:      {sigma_erc:.2%}')
print(f'Target vol:   {sigma_target:.2%}')
print(f'Scale factor: {leverage:.2f}x  ({"LEVERAGE (borrow)" if leverage > 1 else "DE-LEVERAGE (cash buffer)"})')
print()
print('Scaled ERC weights:')
for a, w in zip(assets, w_levered):
    print(f'  {a}: {w:.4f}')
print(f'\nTotal risky weight:  {w_levered.sum():.4f}')
print(f'Cash buffer:         {1 - w_levered.sum():.4f}  (negative = borrowed funds)')
print()

# Now demonstrate the OTHER regime — lift to 80% vol (above ERC) for true leverage
sigma_target_levered = 0.80
leverage_up = sigma_target_levered / sigma_erc
w_up = w_erc * leverage_up
print(f'--- Demonstrating LEVERAGE: target vol = {sigma_target_levered:.0%} ---')
print(f'Scale factor: {leverage_up:.2f}x  (above 1.0 → borrow)')
print(f'Total risky weight:  {w_up.sum():.4f}')
print(f'Borrowed funds:      {w_up.sum() - 1:+.4f}  (positive = borrowed at risk-free rate)')
```

    ERC vol:      53.93%
    Target vol:   40.00%
    Scale factor: 0.74x  (DE-LEVERAGE (cash buffer))
    
    Scaled ERC weights:
      BNB: 0.2205
      BTC: 0.2325
      ETH: 0.1526
      SOL: 0.1360
    
    Total risky weight:  0.7417
    Cash buffer:         0.2583  (negative = borrowed funds)
    
    --- Demonstrating LEVERAGE: target vol = 80% ---
    Scale factor: 1.48x  (above 1.0 → borrow)
    Total risky weight:  1.4834
    Borrowed funds:      +0.4834  (positive = borrowed at risk-free rate)


## Exercises

### Exercise 1 — Verify ERC condition

Confirm that $w_i (\Sigma w)_i$ is constant across assets at the ERC solution.


```python
# Your answer here

```

<details>
<summary><b>Reveal solution</b></summary>

```python
check = w_erc * (Sigma @ w_erc)
print(f'w_i (Σw)_i values:')
for a, c in zip(assets, check):
    print(f'  {a}: {c:.6f}')
print(f'Max deviation: {(check.max() - check.min()) / check.mean() * 100:.2e}%')
```

_All values equal to numerical tolerance._

</details>

### Exercise 2 — Two-asset closed form

For 2 assets with $\sigma_1, \sigma_2, \rho$, derive ERC weights closed-form.


```python
# Your answer here

```

<details>
<summary><b>Reveal solution</b></summary>

```python
# w_1 σ_1² + w_1 w_2 ρ σ_1 σ_2 = w_2 σ_2² + w_1 w_2 ρ σ_1 σ_2
# (w_1 / w_2)² = σ_2² / σ_1² (after some algebra)
# w_1 / w_2 = σ_2 / σ_1 → w_i ∝ 1/σ_i
sigma_a, sigma_b = np.sqrt(Sigma[0,0]), np.sqrt(Sigma[1,1])
w1 = (1/sigma_a) / (1/sigma_a + 1/sigma_b)
w2 = (1/sigma_b) / (1/sigma_a + 1/sigma_b)
print(f'2-asset ERC closed form: w_1 = {w1:.4f}, w_2 = {w2:.4f}')
print(f'Optimised:               {solve_erc(Sigma[:2, :2]).round(4)}')
```

_Two-asset: w_i ∝ 1/σ_i (inverse-vol). Beautifully simple._

</details>

## Interview Q&A

**Q: State ERC condition.**

A: $w_i (\Sigma w)_i = $ constant for all $i$. Each asset contributes equally to portfolio variance.

**Q: Two-asset ERC?**

A: $w_i \propto 1/\sigma_i$. Inverse-vol weighting.

**Q: Why does ERC need leverage?**

A: Without it, ERC produces low-vol portfolio (~5-8% annual). To compete with 60/40 or equity-style allocations (~10-12%), borrow to scale up. Standard practice at All Weather; controversial at retail.

**Q: ERC vs MV — when each?**

A: **ERC** when you don't trust expected-return forecasts (most cases). **MV** when you have strong, defensible μ estimates (rare). ERC is robust; MV is precise but fragile.

**Q: HRP (Hierarchical Risk Parity)?**

A: Lopez de Prado 2016 extension. Cluster assets by similarity, apply ERC within clusters, then between clusters. Less sensitive to covariance estimation noise. State of the art in 2026.

## Pitfalls

| Pitfall | Issue | Fix |
|---|---|---|
| ERC with negative correlations | Weights can blow up | Constrain bounds; check stability |
| Wrong covariance | Sample Σ noisy | Ledoit-Wolf shrinkage |
| Levered RP without funding cost | Leverage isn't free | Subtract financing cost from levered return |
| Rebalancing drift | RP weights change as vols change | Daily/weekly rebalance |
| RP for an alpha portfolio | Designed for systematic risk allocation | Don't use for beating a benchmark |

## What you've earned

You can solve for ERC weights via SLSQP, verify the equal-risk-contribution condition, compare to MV and 1/N, apply leverage to hit a target vol, and defend the framework in interview. Combined with Black-Litterman (notebook 02), the modern AM portfolio toolkit.
