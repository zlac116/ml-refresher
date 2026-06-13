# Swaps & Swaptions in Detail

## Why this matters

The **interest rate swap (IRS)** is the world's largest derivatives market by notional ($500T+). Swaptions are options on swaps — the standard hedging tool for callable bonds, prepayment risk, and asset-liability management at insurers/pension funds.

Building on `03_curve_building.ipynb` (curve bootstrap) and `01_options/02_bs_family_and_asset_classes.ipynb` (Black-76 + Bachelier), this notebook covers swap mark-to-market, par-swap rates, swaption pricing under both lognormal (Black-76) and normal (Bachelier) vol conventions.

You will be asked:
1. Price a non-par IRS at mark-to-market.
2. Compute the par-swap rate from a curve.
3. **Black-76 swaption** vs **Bachelier swaption** — when each applies.
4. **DV01** of a swap. Why is it the standard hedging metric?
5. **Forward swap rate** vs spot zero rate — the trap.

This notebook covers all five with a real curve.

## Setup — load curve, build pricer


```python
import warnings; warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq

# Synthesised curve from `03_curve_building.ipynb` outputs
tenors_y   = np.array([1/12, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0])
zero_rates = np.array([0.0428, 0.0420, 0.0410, 0.0395, 0.0380, 0.0370, 0.0365, 0.0370, 0.0380, 0.0390, 0.0400])

def DF(t):
    return np.exp(-np.interp(t, tenors_y, zero_rates) * t)


def black_76(F, K, T, r, sigma, option_type='call'):
    if T <= 0:
        return np.maximum(F - K, 0.0) if option_type == 'call' else np.maximum(K - F, 0.0)
    d1 = (np.log(F/K) + 0.5*sigma**2*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    df_disc = np.exp(-r*T)
    if option_type == 'call':
        return df_disc * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return df_disc * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def bachelier(F, K, T, r, sigma_n, option_type='call'):
    if T <= 0:
        return np.maximum(F - K, 0.0) if option_type == 'call' else np.maximum(K - F, 0.0)
    d = (F - K) / (sigma_n * np.sqrt(T))
    df_disc = np.exp(-r*T)
    if option_type == 'call':
        return df_disc * ((F-K)*norm.cdf(d) + sigma_n*np.sqrt(T)*norm.pdf(d))
    return df_disc * ((K-F)*norm.cdf(-d) + sigma_n*np.sqrt(T)*norm.pdf(d))


print('Curve:')
for t, r_ in zip(tenors_y, zero_rates):
    print(f'  {t:>5}y: {r_*100:.3f}%')
```

    Curve:
      0.08333333333333333y: 4.280%
       0.25y: 4.200%
        0.5y: 4.100%
        1.0y: 3.950%
        2.0y: 3.800%
        3.0y: 3.700%
        5.0y: 3.650%
        7.0y: 3.700%
       10.0y: 3.800%
       20.0y: 3.900%
       30.0y: 4.000%


## IRS pricing — fixed-floating swap

A **payer** swap pays fixed, receives floating. A **receiver** swap is the opposite. PV at trade:

$$\text{PV}^{\text{payer}} = \sum_i (L_i - K) \delta_i D(0, T_i)$$

where $L_i$ are forward floating rates, $K$ is the fixed rate, $\delta_i$ is the accrual fraction. At trade, par swap rate sets PV = 0:

$$K^{\text{par}} = \frac{\sum_i L_i \delta_i D(0, T_i)}{\sum_i \delta_i D(0, T_i)} = \frac{1 - D(0, T_n)}{\sum_i \delta_i D(0, T_i)}$$

(via telescoping of the floating leg).

### From the cashflow PV to the compact $-A(K - K_{par})$ form

The textbook PV form $\sum_i (L_i - K) \delta_i D(0, T_i)$ is correct but cumbersome. There's a much cleaner equivalent that the code uses. Substitute the par-rate identity $1 - D(0, T_n) = K_{par} \cdot A$ (where $A = \sum_i \delta_i D(0, T_i)$ is the annuity) into the floating-leg sum:

$$\text{PV}_{\text{payer}} = \underbrace{\sum_i L_i \delta_i D(0, T_i)}_{\text{floating leg PV} = 1 - D(0, T_n)} - K \cdot A = K_{par} \cdot A - K \cdot A = A \cdot (K_{par} - K)$$

Equivalently:

$$\boxed{\;\text{PV}_{\text{payer}} = -A \cdot (K - K_{par})\;}$$

So a payer swap is **"long the par rate"** — its PV rises by one annuity-unit for every basis point that the par rate exceeds the fixed strike. This is why a swap is also called a "duration trade" — the annuity $A$ IS the dollar-duration per basis point of par-rate move.

The compact form makes mark-to-market trivial: rebuild the par rate from today's curve, multiply the difference vs the strike by today's annuity, scale by notional.


```python
def par_swap_rate(T_start, T_end, dt=1.0):
    pay_dates = np.arange(T_start + dt, T_end + dt/2, dt)
    annuity = sum(dt * DF(t) for t in pay_dates)
    return (DF(T_start) - DF(T_end)) / annuity, annuity, pay_dates


def irs_pv_payer(T_start, T_end, fixed_K, dt=1.0, notional=1e8):
    par_K, annuity, pay_dates = par_swap_rate(T_start, T_end, dt)
    pv = -annuity * (fixed_K - par_K) * notional
    return pv, par_K, annuity


# 5y annual swap struck at 3.50% (off-market)
pv, par_K, _ = irs_pv_payer(T_start=0.0, T_end=5.0, fixed_K=0.035, notional=1e8)
print(f'5y par-rate: {par_K:.4f} ({par_K*100:.4f}%)')
print(f'Off-market PV (long 5y payer @ 3.50%): ${pv:+,.2f}')
print(f'  → Positive: K = 3.50% < par = {par_K*100:.2f}% — paying below market, swap has positive value to receiver of fixed.')
```

    5y par-rate: 0.0372 (3.7240%)
    Off-market PV (long 5y payer @ 3.50%): $+1,003,273.25
      → Positive: K = 3.50% < par = 3.72% — paying below market, swap has positive value to receiver of fixed.


## DV01 of a swap

The price sensitivity to a 1 bp parallel shift in the discount curve. Standard hedging metric.

For a payer swap with current PV $V$ and curve $z$:

$$\text{DV01} = \frac{\partial V}{\partial z} \cdot 0.0001$$

Numerically, central-difference: $\text{DV01} \approx (V_{-1\text{bp}} - V_{+1\text{bp}}) / 2$. (Sign convention: positive DV01 means PV rises when rates fall — typical for a long-duration position.)

### Hedge ratio

To **DV01-neutralise** a position $A$ using a hedge instrument $B$, you take a position in $B$ such that the combined DV01 is zero:

$$N_B \cdot \text{DV01}_B^{\text{per unit notional}} + \text{DV01}_A = 0 \quad \Rightarrow \quad N_B = -\frac{\text{DV01}_A}{\text{DV01}_B^{\text{per unit notional}}}$$

Both DV01s must be **per the same notional convention** (typically per unit of notional, or per dollar). The negative sign means a long position in $A$ is hedged with a short position in $B$ when both DV01s have the same sign.


```python
# DV01 by curve-shift bumping
def shifted_DF(t, shift=0):
    return np.exp(-(np.interp(t, tenors_y, zero_rates) + shift) * t)

def irs_pv_payer_shifted(T_start, T_end, fixed_K, dt=1.0, notional=1e8, shift=0):
    pay_dates = np.arange(T_start + dt, T_end + dt/2, dt)
    annuity = sum(dt * shifted_DF(t, shift) for t in pay_dates)
    par_K = (shifted_DF(T_start, shift) - shifted_DF(T_end, shift)) / annuity
    return -annuity * (fixed_K - par_K) * notional

def irs_dv01(T_start, T_end, fixed_K, notional=1e8):
    """Central-difference DV01 in dollars per 1bp parallel shift."""
    pv_up = irs_pv_payer_shifted(T_start, T_end, fixed_K, notional=notional, shift=+0.0001)
    pv_dn = irs_pv_payer_shifted(T_start, T_end, fixed_K, notional=notional, shift=-0.0001)
    return (pv_dn - pv_up) / 2


T_start, T_end, K_test = 0.0, 5.0, 0.040
notional_5y = 1e8
pv_0  = irs_pv_payer_shifted(T_start, T_end, K_test, notional=notional_5y)
pv_up = irs_pv_payer_shifted(T_start, T_end, K_test, notional=notional_5y, shift=+0.0001)
pv_dn = irs_pv_payer_shifted(T_start, T_end, K_test, notional=notional_5y, shift=-0.0001)
dv01_5y = irs_dv01(T_start, T_end, K_test, notional=notional_5y)

print(f'5y payer swap @ 4%, ${notional_5y/1e6:.0f}M notional:')
print(f'  PV @ curve:    ${pv_0:+,.2f}')
print(f'  PV @ +1bp:     ${pv_up:+,.2f}')
print(f'  PV @ -1bp:     ${pv_dn:+,.2f}')
print(f'  DV01:          ${dv01_5y:+,.2f}  per 1bp')
print()

# DV01-neutral hedge using a 10y payer swap
# Compute the 10y DV01 PER UNIT NOTIONAL (use $1 notional for clarity)
dv01_10y_per_dollar = irs_dv01(0, 10, 0.04, notional=1.0)
hedge_face = -dv01_5y / dv01_10y_per_dollar    # negative because we short to offset
print(f'10y payer swap DV01 per $1 notional:  ${dv01_10y_per_dollar:.6f}')
print(f'Hedge: to DV01-neutralise the long 5y, take {hedge_face:+,.0f} of 10y (negative = short).')
print(f'Check: combined DV01 = ${dv01_5y + hedge_face * dv01_10y_per_dollar:.4f}  (should be ~0).')
```

    5y payer swap @ 4%, $100M notional:
      PV @ curve:    $-1,236,478.52
      PV @ +1bp:     $-1,189,583.12
      PV @ -1bp:     $-1,283,396.64
      DV01:          $-46,906.76  per 1bp
    
    10y payer swap DV01 per $1 notional:  $-0.000854
    Hedge: to DV01-neutralise the long 5y, take -54,929,789 of 10y (negative = short).
    Check: combined DV01 = $0.0000  (should be ~0).


## Swaption pricing — Black-76 vs Bachelier

A **payer swaption** is a call on the swap rate. PV at expiry:

$$V_T = A_T \cdot \max(S_T - K, 0)$$

where $S_T$ is the prevailing swap rate and $A_T$ is the annuity. Under Black-76:

$$\text{Payer}_0 = A_0 \cdot \big[F_S \cdot N(d_1) - K \cdot N(d_2)\big]$$

Under Bachelier (when rates can be near zero/negative):

$$\text{Payer}_0 = A_0 \cdot \big[(F_S - K) \cdot N(d) + \sigma_n \sqrt{T} \cdot \phi(d)\big]$$


```python
# 1y × 5y payer swaption at K = 4% with σ_BS = 30% (lognormal) and σ_n equivalent
T_expiry = 1.0
T_swap_end = 6.0
K_strike = 0.040

F_swap, annuity, _ = par_swap_rate(T_expiry, T_swap_end, dt=1.0)

# Black-76
sigma_bs = 0.30
payer_b76 = annuity * black_76(F_swap, K_strike, T_expiry, r=0, sigma=sigma_bs, option_type='call')

# Bachelier with σ_n ≈ σ_BS · F (rule of thumb)
sigma_n = sigma_bs * F_swap
payer_bach = annuity * bachelier(F_swap, K_strike, T_expiry, r=0, sigma_n=sigma_n, option_type='call')

print(f'1y × 5y payer swaption @ K = {K_strike*100:.1f}%:')
print(f'  Forward swap rate F_S: {F_swap*100:.4f}%')
print(f'  Annuity:               {annuity:.4f}')
print(f'  Black-76 (σ={sigma_bs:.0%}):    {payer_b76*1e4:.2f} bps of notional')
print(f'  Bachelier (σ_n={sigma_n*1e4:.1f} bp): {payer_bach*1e4:.2f} bps of notional')
print(f'\n→ Bachelier and Black-76 prices match closely when F is far from zero — same model in different vol units.')
```

    1y × 5y payer swaption @ K = 4.0%:
      Forward swap rate F_S: 3.6838%
      Annuity:               4.3204
      Black-76 (σ=30%):    136.95 bps of notional
      Bachelier (σ_n=110.5 bp): 129.91 bps of notional
    
    → Bachelier and Black-76 prices match closely when F is far from zero — same model in different vol units.


## Exercises

### Exercise 1 — Payer-receiver parity

For a 2y × 3y swaption at $K = 3.5\%$, $\sigma = 0.30$: verify Payer − Receiver = Annuity × (F_swap − K).


```python
# Your answer here

```

<details>
<summary><b>Reveal solution</b></summary>

```python
T_exp, T_end = 2.0, 5.0
F_S, A, _ = par_swap_rate(T_exp, T_end)
K_, sigma_ = 0.035, 0.30

p = A * black_76(F_S, K_, T_exp, r=0, sigma=sigma_, option_type='call')
r_ = A * black_76(F_S, K_, T_exp, r=0, sigma=sigma_, option_type='put')
parity = A * (F_S - K_)
print(f'P - R = {p - r_:.6f}')
print(f'A(F-K) = {parity:.6f}')
```

_Holds exactly to floating-point precision._

</details>

### Exercise 2 — Vega of a swaption

Compute Vega = ∂V/∂σ via finite difference for our 1y×5y payer.


```python
# Your answer here

```

<details>
<summary><b>Reveal solution</b></summary>

```python
h = 0.0001
V_up = annuity * black_76(F_swap, K_strike, T_expiry, r=0, sigma=sigma_bs+h, option_type='call')
V_dn = annuity * black_76(F_swap, K_strike, T_expiry, r=0, sigma=sigma_bs-h, option_type='call')
vega = (V_up - V_dn) / (2*h) / 100   # per 1% vol change
print(f'Swaption vega: {vega*1e4:.2f} bps per 1% vol change')
```

_Larger annuity → larger vega for the same swaption struture._

</details>

## Interview Q&A

**Q: Price a par swap.**

A: Par K = (DF(T_start) - DF(T_end)) / Σ δ_i DF(T_i). Floating-leg PV telescopes to 1 - DF(T_end). Fixed-leg PV = K × annuity. Set equal.

**Q: Black-76 vs Bachelier swaption?**

A: Black-76 assumes lognormal swap rate (rates can't go negative). Bachelier assumes normal rate (rates can). Post-2014 in EUR/JPY: Bachelier is standard. In USD: lognormal still common, but normal is gaining ground.

**Q: Why is annuity discounting?**

A: Under the **annuity measure** (numéraire = sum of payment-date DFs), the forward swap rate is a martingale. So we can apply Black-76 / Bachelier directly with annuity as the discount.

**Q: DV01 vs duration?**

A: Both measure rate sensitivity. **Duration** is fractional (years). **DV01** is dollar (per 1 bp). For a swap, duration = annuity/notional (×0.5 for fixed-only, ×1 for total), and DV01 = annuity × notional × 0.0001 (approximately).

**Q: Forward swap rate vs spot zero rate?**

A: They're different objects. The 1Y zero rate is a discount rate. The 1Y forward 5Y swap rate is the par fixed rate of a swap that *starts* in 1Y and runs 5Y. Confusing them is a common bug — see notebook 03_curve_building.

## Pitfalls

| Pitfall | Issue | Fix |
|---|---|---|
| Discounting swap CFs at flat YTM | Wrong | Use bootstrapped curve |
| Black-76 with negative forward | log of negative → NaN | Switch to Bachelier or shifted-lognormal |
| DV01 sign convention | Long payer swap has positive DV01 (rate up = +) but long receiver has negative | State direction |
| Annuity wrong frequency | Quarterly fixed-leg has δ = 0.25 not 1.0 | Match the schedule |
| Confusing forward rate measures | Black-76 is under T_pay measure, swaption uses annuity | Different martingale measures for different products |

## What you've earned

After this notebook you can price IRSs at MTM, compute par swap rates from a curve, hedge via DV01, price swaptions under both Black-76 and Bachelier conventions, and verify payer-receiver parity. Combined with `01_options/02_bs_family_and_asset_classes.ipynb` and `03_curve_building.ipynb`, you have the full IR-derivatives toolkit.
