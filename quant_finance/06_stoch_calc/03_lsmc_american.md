# Longstaff-Schwartz MC for American Options

## Why this matters

Binomial trees handle American options well in 1D but fail in higher dimensions (multi-asset, stochastic vol). **Longstaff-Schwartz (2001)** is the workhorse Monte Carlo method for American-style early exercise — used at every option-trading desk for high-dimensional pricing.

You will be asked at any structurer/exotic-derivatives interview:
1. State the LSM algorithm.
2. Why use **regression** for the continuation value?
3. Choice of basis functions — Laguerre, polynomial, or other?
4. **Why is LSM biased low?**
5. Convergence — does LSM converge to the true price?

This notebook covers all five with an American put example.

## The Longstaff-Schwartz algorithm

**Idea**: at each potential exercise time, decide whether to exercise based on **continuation value** = expected discounted payoff from holding. Regress payoff-given-hold against current state to estimate continuation.

**Algorithm** (American put, exercise at any of $t_1 < t_2 < \dots < t_M = T$):

1. Simulate $N$ stock paths under Q.
2. At terminal $t_M$: payoff = $\max(K - S_T, 0)$.
3. **Backward sweep**: for each $t_m$ from $M-1$ down to 1:
   a. Identify in-the-money paths.
   b. Regress *next-step discounted payoff* against polynomial in current $S_{t_m}$ (typically degree 2-3 polynomial or Laguerre basis).
   c. **Continuation value** = regression prediction at the current state.
   d. Exercise if $K - S_{t_m} > $ continuation; else continue.
4. Discount back to $t = 0$.

**Why regression?** The expected continuation value is a conditional expectation. Regression estimates it from sample paths.

### Why it's biased low

The exercise rule from regression is **suboptimal** (you're using estimated continuation value, not the true one). A suboptimal exercise rule can only **underestimate** the option's value. So $V^{LSM}_0 \le V^{true}_0$ in expectation.

## Implementation


```python
import warnings; warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def lsmc_american_put(S0, K, T, r, sigma, n_paths, n_steps, seed=42, basis_degree=3):
    """Longstaff-Schwartz Monte Carlo for American put."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    df_disc = np.exp(-r * dt)

    # Simulate paths under Q
    Z = rng.standard_normal((n_paths, n_steps))
    log_S = np.zeros((n_paths, n_steps + 1))
    log_S[:, 0] = np.log(S0)
    log_S[:, 1:] = np.log(S0) + np.cumsum((r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z, axis=1)
    S = np.exp(log_S)

    # Cash flows: at each step, the option value if exercised at that step (or continued)
    cf = np.maximum(K - S[:, -1], 0)   # terminal payoff

    # Backward sweep
    for m in range(n_steps - 1, 0, -1):
        S_m = S[:, m]
        intrinsic = np.maximum(K - S_m, 0)
        itm = intrinsic > 0

        if itm.sum() < 5:
            cf = cf * df_disc
            continue

        # Discount future cash flows to time t_m
        # Regression: cf_continuation_m = polynomial(S_m)
        S_itm = S_m[itm]
        cf_future = cf[itm] * df_disc

        # Polynomial basis (or Laguerre)
        X = np.column_stack([S_itm**k for k in range(basis_degree + 1)])
        beta_, *_ = np.linalg.lstsq(X, cf_future, rcond=None)
        continuation = X @ beta_

        # Exercise if intrinsic > continuation
        exercise_now = intrinsic[itm] > continuation
        new_cf = np.where(exercise_now, intrinsic[itm], cf_future)

        # Update CF for the ITM paths
        cf_full = cf * df_disc
        cf_full[np.where(itm)[0]] = new_cf
        cf = cf_full

    return cf.mean() * df_disc, cf.std() / np.sqrt(n_paths) * df_disc


# Compare to binomial reference for an American put
S0, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.30
np.random.seed(42)
price_lsm, se = lsmc_american_put(S0, K, T, r, sigma, n_paths=100000, n_steps=50)

# Binomial reference (CRR)
def crr_american_put(S, K, T, r, sigma, n):
    dt = T/n
    u = np.exp(sigma*np.sqrt(dt)); d = 1/u
    p = (np.exp(r*dt) - d) / (u - d)
    j = np.arange(n+1)
    ST = S * u**j * d**(n-j)
    V = np.maximum(K - ST, 0)
    df = np.exp(-r*dt)
    for i in range(n-1, -1, -1):
        V = df * (p*V[1:i+2] + (1-p)*V[0:i+1])
        S_at_i = S * u**np.arange(i+1) * d**(i - np.arange(i+1))
        V = np.maximum(V, np.maximum(K - S_at_i, 0))
    return V[0]

ref_price = crr_american_put(S0, K, T, r, sigma, 1000)

print(f'American put @ S=K=100, σ=30%, T=1, r=5%')
print(f'  CRR (N=1000 ref):  {ref_price:.4f}')
print(f'  LSMC (100k paths): {price_lsm:.4f}  (SE {se:.4f})')
print(f'  Diff:              {price_lsm - ref_price:+.4f}  ({(price_lsm - ref_price)/se:.2f}σ from ref)')
print()
print('→ LSMC matches CRR within MC error. LSMC is biased LOW because exercise rule is suboptimal.')
```

    American put @ S=K=100, σ=30%, T=1, r=5%
      CRR (N=1000 ref):  9.8687
      LSMC (100k paths): 9.7999  (SE 0.0345)
      Diff:              -0.0688  (-1.99σ from ref)
    
    → LSMC matches CRR within MC error. LSMC is biased LOW because exercise rule is suboptimal.


## Convergence


```python
# Convergence in n_paths
n_paths_grid = [1000, 10000, 100000, 500000]
prices = []
for n in n_paths_grid:
    p, se = lsmc_american_put(S0, K, T, r, sigma, n_paths=n, n_steps=50, seed=42)
    prices.append((p, se))

print(f'{"N paths":>10}  {"LSMC":>10}  {"SE":>10}  {"Diff vs ref":>12}')
for n, (p, se) in zip(n_paths_grid, prices):
    print(f'{n:>10}  {p:>10.4f}  {se:>10.4f}  {p - ref_price:>+12.4f}')

print(f'\nReference (CRR N=1000): {ref_price:.4f}')
```

       N paths        LSMC          SE   Diff vs ref
          1000     10.2757      0.3730       +0.4070
         10000      9.9719      0.1113       +0.1032
        100000      9.7999      0.0345       -0.0688
        500000      9.8421      0.0156       -0.0266
    
    Reference (CRR N=1000): 9.8687


## Exercises

### Exercise 1 — Effect of basis degree

Test LSMC with degrees 1, 2, 3, 5. Does higher degree always help?


```python
# Your answer here

```

<details>
<summary><b>Reveal solution</b></summary>

```python
for deg in [1, 2, 3, 5]:
    p, se = lsmc_american_put(S0, K, T, r, sigma, n_paths=100000, n_steps=50, basis_degree=deg)
    print(f'degree={deg}: price = {p:.4f}, SE = {se:.4f}, diff = {p-ref_price:+.4f}')
```

_Higher degree ≠ always better. Trade-off between bias (low degree) and variance (high degree)._

</details>

## Interview Q&A

**Q: State the LSM algorithm.**

A: Simulate paths under Q. Roll back from terminal: at each step, regress next-step discounted payoff against polynomial in current state. Continuation value = regression prediction. Exercise if intrinsic > continuation. Discount to time 0 and average.

**Q: Why is LSM biased low?**

A: The exercise rule (from regression) is suboptimal — you're using an estimated continuation, not the true one. A suboptimal rule can only **underestimate** the option's value. The true price is the supremum over stopping rules; LSM uses one specific rule.

**Q: How to fix the bias?**

A: (1) Use higher-degree basis. (2) **Glasserman-Yu**: pre-compute the exercise rule on a separate set of paths, then evaluate on a fresh set (no in-sample bias). (3) **Andersen-Broadie**: dual upper bound complement; report a confidence interval.

**Q: What basis functions?**

A: Laguerre polynomials are popular (orthogonal under exponential weight, good for stock prices). Plain polynomials in $\ln S$ also common. For multi-asset: tensor-product basis or kernel methods. Degree 2-4 is typical.

**Q: When is LSM appropriate?**

A: Multi-dimensional or complex-payoff American/Bermudan options where binomial fails. Bermudan swaptions, multi-asset baskets with early-exercise, stochastic-vol Americans. Standard production tool.

## Pitfalls

| Pitfall | Issue | Fix |
|---|---|---|
| Wrong basis | Polynomial in S blows up; in $\ln S$ better | Laguerre or Hermite |
| In-sample bias | Same paths used for regression and pricing | Pre-compute exercise rule on separate paths |
| Too few ITM paths | Regression unstable | Increase n_paths or use control variate |
| Using anti-thetic with LSM | Tricky — pair correlations affect regression | Use carefully or skip |
| High-dim curse | Basis grows exponentially | Use sparse grids / neural network LSM |

## What you've earned

You can implement LSMC for American-style options, choose appropriate basis, explain the low-bias issue, and apply it where binomial fails. Combined with binomial trees and MC pricing, the full early-exercise toolkit. **This completes Tier 3 specialist content.**
