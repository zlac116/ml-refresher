# Expected Shortfall (Conditional VaR)

## Why this matters

VaR tells you the loss threshold at confidence $\alpha$. **It says nothing about what happens when that threshold is breached.** A portfolio with 99% VaR of \$10M might lose \$10.1M in the bad case, or \$100M.

**Expected Shortfall (ES)**, also called **Conditional VaR (CVaR)** or **Tail VaR**, is the expected loss *given* loss exceeds VaR:

$$\text{ES}_\alpha = \mathbb{E}[L \mid L > \text{VaR}_\alpha]$$

Equivalently:

$$\text{ES}_\alpha = \frac{1}{1-\alpha} \int_\alpha^1 \text{VaR}_u \, du$$

**ES is coherent (subadditive)**, **ES captures tail severity**, and **FRTB (Basel III post-2016) replaced VaR with ES** at the 97.5% level for market risk capital. Knowing this is interview-mandatory for any risk-related role.

You will be asked:
1. State the definition of ES. Why is it coherent when VaR isn't?
2. Compute ES by all three methods (parametric, historical, MC).
3. Why did FRTB switch to ES at 97.5%, not 99%?
4. Backtest ES — why is it harder than backtesting VaR?
5. Show the ES of a normal distribution = $\mu + \sigma \cdot \phi(z_\alpha) / (1-\alpha)$.

## The 30-second concept

Two equivalent definitions:

1. **Conditional expectation**: $\text{ES}_\alpha = \mathbb{E}[-R \mid -R > \text{VaR}_\alpha]$. Average loss in the worst $1-\alpha$ tail.

2. **Average of VaR over the tail**: $\text{ES}_\alpha = \frac{1}{1-\alpha} \int_\alpha^1 \text{VaR}_u \, du$.

For a normal distribution, the closed form is

$$\text{ES}_\alpha = -\mu + \sigma \cdot \frac{\phi(z_\alpha)}{1 - \alpha}$$

where $\phi$ is the standard normal PDF and $z_\alpha = \Phi^{-1}(\alpha)$. For 97.5%: $z_{0.975} \approx 1.96$, $\phi(1.96) \approx 0.0584$, ratio = $0.0584/0.025 = 2.34$. So ES_97.5% under normal ≈ $\sigma \times 2.34$ vs VaR_99% ≈ $\sigma \times 2.33$ — almost identical, which is **why FRTB chose ES_97.5% to maintain capital roughly equivalent to VaR_99%**.

## Setup


```python
import warnings; warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

df = pd.read_parquet('../../data/crypto_hourly.parquet')
df['ts'] = pd.to_datetime(df['ts'], utc=True)

btc = (df.query('symbol == "BTC"').set_index('ts')[['close']].resample('1D').last().dropna()
       .assign(ret=lambda d: np.log(d['close']).diff()).dropna())
R = btc['ret'].values

print(f'BTC daily returns: n={len(R)}, mean={R.mean()*100:.4f}%, std={R.std()*100:.4f}%')
```

    BTC daily returns: n=730, mean=0.0204%, std=2.4484%


## Three methods of computing ES


```python
def parametric_es(returns, alpha=0.975):
    """Parametric ES under N(mu, sigma^2). ES_alpha = -mu + sigma * phi(z_alpha) / (1-alpha)."""
    mu = returns.mean()
    sigma = returns.std(ddof=1)
    z = stats.norm.ppf(alpha)
    return -mu + sigma * stats.norm.pdf(z) / (1 - alpha)


def historical_es(returns, alpha=0.975):
    """Historical ES: average of returns in the (1-alpha) worst tail."""
    var = -np.quantile(returns, 1 - alpha)
    tail = returns[returns < -var]
    return -tail.mean() if len(tail) > 0 else var


def mc_es_with_t(returns, alpha=0.975, n_sims=1_000_000, seed=42):
    df_t, loc, scale = stats.t.fit(returns)
    rng = np.random.default_rng(seed)
    samples = stats.t.rvs(df_t, loc, scale, size=n_sims, random_state=rng)
    var_t = -np.quantile(samples, 1 - alpha)
    tail  = samples[samples < -var_t]
    return -tail.mean(), df_t


def parametric_var(returns, alpha=0.975):
    return -(returns.mean() + returns.std(ddof=1) * stats.norm.ppf(1 - alpha))


def historical_var(returns, alpha=0.975):
    return -np.quantile(returns, 1 - alpha)


for a in [0.95, 0.975, 0.99]:
    var_p = parametric_var(R, a)
    es_p  = parametric_es (R, a)
    var_h = historical_var(R, a)
    es_h  = historical_es (R, a)
    es_t, df_t = mc_es_with_t(R, a)
    print(f'{a:.2%}: ', end='')
    print(f'VaR(param)={var_p*100:5.2f}% ES(param)={es_p*100:5.2f}% ', end='')
    print(f'VaR(hist)={var_h*100:5.2f}% ES(hist)={es_h*100:5.2f}% ', end='')
    print(f'ES(t,ν={df_t:.1f})={es_t*100:5.2f}%')

print('\n→ ES > VaR by definition (it averages the tail, which is worse than the threshold)')
print('→ Historical and t-MC ES > parametric ES — fat tails make the conditional tail expectation much worse')
```

    95.00%: VaR(param)= 4.01% ES(param)= 5.03% VaR(hist)= 3.72% ES(hist)= 5.36% ES(t,ν=3.6)= 5.80%
    97.50%: VaR(param)= 4.78% ES(param)= 5.71% VaR(hist)= 4.98% ES(hist)= 6.49% ES(t,ν=3.6)= 7.31%


    99.00%: VaR(param)= 5.68% ES(param)= 6.51% VaR(hist)= 5.98% ES(hist)= 8.09% ES(t,ν=3.6)= 9.72%
    
    → ES > VaR by definition (it averages the tail, which is worse than the threshold)
    → Historical and t-MC ES > parametric ES — fat tails make the conditional tail expectation much worse


## ES is coherent (subadditive)

Recall the binary counter-example for VaR: two independent positions with 95% VaR = 0 individually but 95% VaR = $100 combined. **Same example** under ES — does it hold?


```python
rng = np.random.default_rng(0)
n = 1_000_000
A = np.where(rng.random(n) < 0.04, -100, 0)
B = np.where(rng.random(n) < 0.04, -100, 0)
combined = A + B

def es_from_sample(losses, alpha=0.95):
    """ES from samples (positive losses)."""
    var = np.quantile(losses, alpha)
    tail = losses[losses > var]
    return tail.mean() if len(tail) > 0 else var

# Convert to losses (positive numbers)
ES_A = es_from_sample(-A, 0.95)
ES_B = es_from_sample(-B, 0.95)
ES_C = es_from_sample(-combined, 0.95)

print(f'95% ES of position A:  ${ES_A:.2f}')
print(f'95% ES of position B:  ${ES_B:.2f}')
print(f'95% ES of A+B:         ${ES_C:.2f}')
print(f'Sum of individual ES:  ${ES_A + ES_B:.2f}')
print(f'\nES(A+B) ≤ ES(A) + ES(B): {ES_C <= ES_A + ES_B}  ← subadditive ✓')
print('→ Diversification can only reduce ES (never increase it). VaR fails this property.')
```

    95% ES of position A:  $100.00
    95% ES of position B:  $100.00
    95% ES of A+B:         $200.00
    Sum of individual ES:  $200.00
    
    ES(A+B) ≤ ES(A) + ES(B): True  ← subadditive ✓
    → Diversification can only reduce ES (never increase it). VaR fails this property.


## FRTB and the 97.5% choice

Why **97.5% ES** specifically? Because for a Gaussian distribution,

$$\text{VaR}_{99\%} \approx 2.326\sigma, \qquad \text{ES}_{97.5\%} \approx 2.338\sigma$$

— almost identical. Regulators wanted ES (better tail behaviour, coherence) at a level that produced **roughly the same capital charge** as the legacy 99% VaR. They chose 97.5% for ES because it preserves the "comparable capital" criterion under normal assumptions.

Under fat-tailed distributions, ES_97.5% > VaR_99% — capital actually *increases* for fat-tailed books. That's by design: FRTB wanted higher capital where tails are real.

### Student-$t$ ES — closed form and where the fat-tail amplifier comes from

For losses $L = -X$ where $X \sim \sigma \cdot t_\nu$ (zero-mean, $\nu$ degrees of freedom), the analytic ES at confidence $\alpha$ is:

$$\text{ES}_\alpha = \sigma \cdot \frac{f_\nu(t_\alpha)}{1 - \alpha} \cdot \frac{\nu + t_\alpha^2}{\nu - 1}$$

where $f_\nu$ is the Student-$t$ PDF and $t_\alpha$ is the $\alpha$-quantile.

**Read it in two pieces:**

- $\sigma \cdot \dfrac{f_\nu(t_\alpha)}{1 - \alpha}$ — the same form as the normal ES (mean of the truncated tail of a $t$-density divided by tail probability).
- $\dfrac{\nu + t_\alpha^2}{\nu - 1}$ — the **fat-tail amplifier**. Equals 1 when $\nu \to \infty$ (recover normal). Grows as $\nu \to 2^+$. Undefined for $\nu \le 1$ (mean doesn't exist).

This amplifier is why Basel switched from VaR to ES at FRTB: it captures tail thickness directly. For $\nu = 5$ at 99%, the amplifier is roughly $1.5\times$ normal — nearly 50% more capital required for the same parametric vol input.


```python
# Demonstrate the FRTB capital-equivalence under normal vs t
sigma = 0.02
mu = 0.0
print(f'Under normal (μ=0, σ={sigma:.2%}):')
print(f'  VaR 99%   = {-(mu + sigma*stats.norm.ppf(0.01))*100:.4f}%')
print(f'  ES  97.5% = {(-mu + sigma*stats.norm.pdf(stats.norm.ppf(0.975))/0.025)*100:.4f}%')
print(f'  ratio = {((-mu + sigma*stats.norm.pdf(stats.norm.ppf(0.975))/0.025) / (-(mu + sigma*stats.norm.ppf(0.01)))):.4f}')

# Under t with ν = 5 (fat tails)
print(f'\nUnder Student-t ν=5 (fat tails) with same scale:')
df_t = 5
var_t  = -stats.t.ppf(0.01, df_t) * sigma
es_t_alpha = 0.975
# ES under t: -mu + scale * pdf(t_alpha) * (df + t_alpha^2) / ((1-alpha) * (df-1))
t_alpha = stats.t.ppf(es_t_alpha, df_t)
es_t = sigma * stats.t.pdf(t_alpha, df_t) * (df_t + t_alpha**2) / ((1 - es_t_alpha) * (df_t - 1))
print(f'  VaR 99%   = {var_t*100:.4f}%')
print(f'  ES  97.5% = {es_t*100:.4f}%')
print(f'  ratio = {es_t/var_t:.4f}  (>1 — fat tails → ES capital exceeds VaR)')
```

    Under normal (μ=0, σ=2.00%):
      VaR 99%   = 4.6527%
      ES  97.5% = 4.6756%
      ratio = 1.0049
    
    Under Student-t ν=5 (fat tails) with same scale:
      VaR 99%   = 6.7299%
      ES  97.5% = 7.0432%
      ratio = 1.0466  (>1 — fat tails → ES capital exceeds VaR)


## Backtesting ES — Acerbi-Szekely

Backtesting ES is **fundamentally harder** than VaR. VaR is a quantile (one observation per period suffices to test). ES is an expected value over a region — you need *all* tail observations to test correctly.

**Acerbi-Szekely (2014)** provide the standard tests. The simplest, **Z2**, normalises actual tail losses by predicted ES:

$$Z_2 = \frac{1}{n} \sum_i \frac{R_i \cdot \mathbf{1}_{R_i < -\text{VaR}_i}}{(1-\alpha) \cdot (-\text{ES}_i)} - 1$$

Under H₀ (correct ES model): $Z_2$ has mean 0. Reject if $Z_2 < -$ critical (model under-estimates ES, dangerous) or if $Z_2 > +$ critical (over-estimates, conservative).


```python
# Rolling-window historical ES backtest
window = 252
out_real, out_var, out_es = [], [], []
for t in range(window, len(R)):
    sample = R[t-window:t]
    out_real.append(R[t])
    out_var.append(historical_var(sample, 0.975))
    out_es .append(historical_es (sample, 0.975))

out_real = np.array(out_real); out_var = np.array(out_var); out_es = np.array(out_es)
n_test = len(out_real)

# Acerbi-Szekely Z2
exceed = out_real < -out_var
# Z2 < 0 → model under-states tail loss (dangerous). Z2 > 0 → over-states (conservative).
# tail returns are negative; ES is positive (loss number); ratio neg/(neg) → positive sample mean ⇒ Z2 = mean − 1
Z2 = -np.sum(out_real * exceed / ((1 - 0.975) * out_es)) / n_test - 1

print(f'Backtest of 97.5% historical ES on {n_test} days')
print(f'  exceedances (returns < -VaR): {exceed.sum()} / {n_test} ({exceed.sum()/n_test*100:.2f}%, expected 2.5%)')
print(f'  Z2 statistic: {Z2:+.4f}  (closer to 0 = better)')
print(f'  → Z2 < 0: model UNDERSTATES tail loss (dangerous). Z2 > 0: overstates (conservative).')
```

    Backtest of 97.5% historical ES on 478 days
      exceedances (returns < -VaR): 14 / 478 (2.93%, expected 2.5%)
      Z2 statistic: +0.3556  (closer to 0 = better)
      → Z2 < 0: model UNDERSTATES tail loss (dangerous). Z2 > 0: overstates (conservative).


## Exercises

### Exercise 1 — ES under normal distribution closed form

Show numerically that $\text{ES}_\alpha = \mu + \sigma \cdot \phi(z_\alpha) / (1-\alpha)$ for $X \sim \mathcal{N}(\mu, \sigma^2)$ where $z_\alpha = \Phi^{-1}(\alpha)$. Use $\mu = 0.001$, $\sigma = 0.02$, $\alpha = 0.975$. Verify against MC.


```python
# Your answer here

```

<details>
<summary><b>Reveal solution</b></summary>

```python
mu, sigma, alpha = 0.001, 0.02, 0.975
z = stats.norm.ppf(alpha)
formula_es = -mu + sigma * stats.norm.pdf(z) / (1 - alpha)

rng = np.random.default_rng(42)
samples = rng.normal(mu, sigma, 1_000_000)
losses = -samples
var_emp = np.quantile(losses, alpha)
es_emp = losses[losses > var_emp].mean()

print(f'Closed-form ES: {formula_es*100:.4f}%')
print(f'MC ES:          {es_emp*100:.4f}%')
print(f'agreement to: {abs(formula_es - es_emp):.2e}')
```

_Closed-form and MC match to MC error._

</details>

### Exercise 2 — ES at multiple alphas

Plot historical and parametric ES vs $\alpha$ from 0.90 to 0.999 on the BTC sample. Note the divergence at extreme tails.


```python
# Your answer here

```

<details>
<summary><b>Reveal solution</b></summary>

```python
alphas = np.array([0.90, 0.95, 0.975, 0.99, 0.995, 0.999])
es_p = [parametric_es(R, a) for a in alphas]
es_h = [historical_es(R, a) for a in alphas]

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(alphas, np.array(es_p)*100, 'o-', label='Parametric (normal)')
ax.plot(alphas, np.array(es_h)*100, 's-', label='Historical')
ax.set_xlabel('α'); ax.set_ylabel('ES (%)')
ax.set_title('ES vs confidence level — parametric understates extreme tails')
ax.legend(); ax.grid(alpha=0.3); plt.tight_layout(); plt.show()
```

_Parametric understates ES at extreme α (0.99+) due to normal-distribution thin tails._

</details>

### Exercise 3 — Portfolio ES vs sum-of-individual ES

For an equal-weight BTC + ETH portfolio, compute portfolio 97.5% historical ES. Compare to ES(BTC) + ES(ETH). Subadditivity says portfolio ≤ sum.


```python
# Your answer here

```

<details>
<summary><b>Reveal solution</b></summary>

```python
eth = (df.query('symbol == "ETH"').set_index('ts')[['close']].resample('1D').last().dropna()
       .assign(ret=lambda d: np.log(d['close']).diff()).dropna())
common = btc.index.intersection(eth.index)
btc_r = btc.loc[common, 'ret'].values
eth_r = eth.loc[common, 'ret'].values
port_r = 0.5 * btc_r + 0.5 * eth_r

es_btc = historical_es(btc_r, 0.975)
es_eth = historical_es(eth_r, 0.975)
es_port = historical_es(port_r, 0.975)

print(f'BTC ES:        {es_btc*100:.2f}%')
print(f'ETH ES:        {es_eth*100:.2f}%')
print(f'Sum of weighted ES: {0.5*es_btc + 0.5*es_eth:.4%}')
print(f'Portfolio ES:  {es_port*100:.2f}%')
print(f'Subadditive: {es_port <= 0.5*es_btc + 0.5*es_eth}')
```

_Portfolio ES ≤ weighted sum (correlation < 1)._

</details>

## Interview Q&A

**Q: Define ES. How is it different from VaR?**

A: $\text{ES}_\alpha = \mathbb{E}[L \mid L > \text{VaR}_\alpha]$ — expected loss given loss exceeds VaR. VaR is the threshold; ES is the average loss in the tail beyond. Always ES ≥ VaR.

**Q: Why is ES coherent and VaR isn't?**

A: VaR fails subadditivity (binary counter-example). ES is a coherent risk measure (Acerbi 2002, Rockafellar-Uryasev 2002): it satisfies translation invariance, subadditivity, positive homogeneity, and monotonicity. Coherence ↔ economically rational risk measure that respects diversification.

**Q: Why FRTB switched to 97.5% ES?**

A: Three reasons: (1) ES is coherent, VaR isn't. (2) ES captures tail severity, VaR doesn't. (3) ES_97.5% gives roughly the same capital charge as VaR_99% under normal assumptions, so the *capital floor* doesn't change radically — but for fat-tailed books, ES_97.5% > VaR_99%, increasing capital where tails matter. Net: better risk measurement, comparable headline number.

**Q: ES closed form for normal returns?**

A: $\text{ES}_\alpha = \mu + \sigma \cdot \phi(z_\alpha) / (1 - \alpha)$ for losses, where $z_\alpha = \Phi^{-1}(\alpha)$. Under returns: $\text{ES}_\alpha = -\mu + \sigma \cdot \phi(z_\alpha) / (1 - \alpha)$.

**Q: Backtesting ES is harder than VaR. Why?**

A: VaR is a single quantile — counting exceedances suffices (Kupiec). ES is an expected value over the entire tail region — you need full information about every tail observation. Acerbi-Szekely (2014) proposed Z1/Z2/Z3 tests that normalise tail observations against predicted ES. Even these are sensitive to small tail-sample sizes — backtesting at 97.5% on daily data needs many years of data to be statistically meaningful.

**Q: What's "elicitability" and why does it matter?**

A: A statistical functional is **elicitable** if it minimises an expected scoring function. VaR is elicitable (the pinball loss elicits the quantile). ES is **not elicitable** (Gneiting 2011). This was historically a concern for backtesting — but Fissler-Ziegel 2016 showed (VaR, ES) jointly are elicitable, which is enough for joint backtesting. Interview-relevant for senior risk roles.

**Q: When would you NOT use ES?**

A: When you need a single-number summary tied to a regulatory percentile (legacy reporting). When data is too short for reliable tail estimation. When a coherent measure isn't needed — e.g. internal capital allocation where you trust the marginal contribution of each book and don't need subadditivity guarantees.

## Pitfalls reference card

| Pitfall | Issue | Fix |
|---|---|---|
| Reporting only VaR | Misses tail severity | Always pair with ES |
| ES at 99% | Tail too thin to estimate reliably from finite sample | Use 97.5% (FRTB convention) or 95% |
| Parametric ES on fat-tailed data | Severely understates conditional tail | Use t-distribution or historical |
| Backtesting ES with VaR-style exceedance count | ES isn't a quantile — Kupiec doesn't apply | Acerbi-Szekely Z-tests |
| Confusing ES, CVaR, AVaR, TVaR | All synonyms | Pick one term, stick with it |
| ES of a portfolio = average of position ES | Wrong (subadditivity gives ≤, not =) | Compute on portfolio P&L distribution directly |
| Day-count mismatch in scaling | Going from daily ES to monthly ES via √20 | Same caveats as VaR — vol clustering breaks it |

## What you've earned

After this notebook you can:

1. **State and derive** the ES formula for normal and Student-t distributions.
2. **Compute** ES via parametric, historical, and t-MC methods.
3. **Demonstrate** ES subadditivity vs VaR's failure on the binary counter-example.
4. **Explain** the FRTB 97.5% choice — capital equivalence under normal, capital increase under fat tails.
5. **Run** Acerbi-Szekely Z2 backtest and interpret over/under-estimation.
6. **Defend** ES vs VaR in interviews — coherence, tail-severity capture, regulatory adoption.

Next: **`03_fixed_income/01_bond_pricing.ipynb`** — yield-to-maturity, dirty/clean price, day-count conventions.
