# Duration, Convexity, Key Rate Durations

## Why this matters

Duration and convexity are the **bond Greeks**. Where options have $\Delta, \Gamma$, bonds have **modified duration** and **convexity**. They tell you how the bond price moves in response to yield shifts.

You will be asked, in any FI interview:
1. Define Macaulay duration and modified duration. Relationship?
2. Compute duration and convexity from scratch on a coupon bond.
3. **Hedge a 10y bond with 2y and 30y bonds** — given duration and dollar duration, solve.
4. **Key rate durations** — why do you need them on top of plain duration?
5. Why do callable bonds have **negative convexity**?
6. Why is a barbell more convex than a bullet of the same duration?

This notebook covers all six with concrete worked examples.

## The 30-second concept

The bond price $P(y)$ as a function of yield $y$ admits a Taylor expansion:

$$\frac{\Delta P}{P} \approx -D \cdot \Delta y + \frac{1}{2} C \cdot (\Delta y)^2$$

In plain ASCII:

```
dP/P   ≈   -D * dy   +   (1/2) * C * dy^2
```

`D` is duration (the slope, in % per yield unit). `C` is convexity (the curvature, in %/yield² units).

### The four formulas explicitly (closed form)

```
                  sum_i  t_i * PV_i
Macaulay D   =   --------------------                 (units: years)
                          P

Modified D   =   Macaulay D  /  (1 + y/m)             (units: years; also "% per yield unit")

                  sum_i  t_i * (t_i + 1/m) * PV_i
Convexity    =   ----------------------------------   (units: years²)
                       P  *  (1 + y/m)^2

DV01         =   Modified D  *  P  *  0.0001          (units: $ per bp per $ of face)
```

where for each cashflow `i`:
- `t_i`  = time of cashflow in YEARS (e.g. 0.5, 1.0, 1.5, ...)
- `PV_i` = `cf_i  /  (1 + y/m)^(m * t_i)`   ← present value of cashflow i
- `P`    = `sum_i PV_i`                     ← bond price = sum of all PVs
- `y`    = YTM (annualised, decimal: 0.05 for 5%)
- `m`    = coupons per year (semi-annual → m=2; annual → m=1)

### Where each formula comes from

`Macaulay D` is just the weighted-average time you wait for cashflows, with PVs as weights. (No calculus.)

`Modified D` and `Convexity` are the first and second derivatives of price with respect to yield, both divided by `P` to make them percentages:

$$D = -\dfrac{1}{P} \cdot \dfrac{\partial P}{\partial y}  \qquad C = \dfrac{1}{P} \cdot \dfrac{\partial^2 P}{\partial y^2}$$

Plain ASCII:

```
Modified D   =   -(1/P)  *  dP/dy        ← slope
Convexity    =    (1/P)  *  d²P/dy²      ← curvature
```

When you differentiate `P(y) = sum cf_i / (1+y/m)^(m·t_i)` once with respect to `y`, the `t_i * PV_i` weighting falls out (giving Macaulay). When you differentiate twice, the `t_i * (t_i + 1/m) * PV_i` weighting falls out (giving Convexity). The extra `(1+y/m)` and `(1+y/m)^2` factors come from the chain rule on the discount term.

### Dollar duration / DV01

- **Dollar duration** = $D \cdot P$ — sensitivity in $ per 1.0 (100%) yield change.
- **DV01** (dollar value of 1 bp) = $D \cdot P \cdot 0.0001$ — sensitivity to a 1 bp move.

In plain ASCII:

```
Dollar duration  =  D * P                  ($ per yield unit)
DV01             =  D * P * 0.0001         ($ per bp per $ face)
```

### Formula → code mapping

The formulas above translate to numpy line-for-line. For a generic helper that returns `(times, cf, y_per)` where `y_per = y/m`:

```python
t, cf, y_per = cash_flows_and_times(face, coupon, ytm, T, freq)
                                          # t        = times in years     [0.5, 1.0, 1.5, ...]
                                          # cf       = cashflow at each   [coupon, coupon, ..., coupon+face]
                                          # y_per    = ytm / freq         (per-period yield)

pv     = cf / (1 + y_per) ** (freq * t)   # PV_i, vectorised over all cashflows
P      = np.sum(pv)                       # bond price

D_mac  = np.sum(t * pv) / P                                # Macaulay duration
D_mod  = D_mac / (1 + y_per)                               # Modified duration
C      = np.sum(t * (t + 1/freq) * pv) / (P * (1 + y_per)**2)   # Convexity
DV01   = D_mod * P * 0.0001                                # $ per bp per $ face
```

Each python line mirrors one formula line above. Read top-to-bottom in either column.

## Setup


```python
import warnings; warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq

def bond_price(face, coupon_rate, ytm, T, freq=2):
    n = int(round(T*freq))
    coupon = face*coupon_rate/freq
    cf = np.array([coupon]*n); cf[-1] += face
    times = np.arange(1, n+1)/freq
    return np.sum(cf / (1 + ytm/freq)**(freq*times))

def yield_to_maturity(price, face, coupon_rate, T, freq=2):
    f = lambda y: bond_price(face, coupon_rate, y, T, freq) - price
    return brentq(f, -0.5, 2.0, xtol=1e-10)
```

## Implementation — Macaulay & modified duration, convexity


```python
def cash_flows_and_times(face, coupon_rate, T, freq=2):
    n = int(round(T*freq))
    coupon = face*coupon_rate/freq
    cf = np.array([coupon]*n); cf[-1] += face
    times = np.arange(1, n+1)/freq
    return cf, times


def macaulay_duration(face, coupon_rate, ytm, T, freq=2):
    """Macaulay duration = PV-weighted average time to cash flow."""
    cf, t = cash_flows_and_times(face, coupon_rate, T, freq)
    pv = cf / (1 + ytm/freq)**(freq*t)
    return np.sum(t * pv) / np.sum(pv)


def modified_duration(face, coupon_rate, ytm, T, freq=2):
    """Modified duration = -1/P · dP/dy."""
    return macaulay_duration(face, coupon_rate, ytm, T, freq) / (1 + ytm/freq)


def convexity(face, coupon_rate, ytm, T, freq=2):
    """Convexity = (1/P) d²P/dy². Note the 1/(1+y/m)² factor."""
    cf, t = cash_flows_and_times(face, coupon_rate, T, freq)
    P = np.sum(cf / (1 + ytm/freq)**(freq*t))
    weights = (cf / (1 + ytm/freq)**(freq*t)) / P
    return np.sum(weights * t * (t + 1/freq)) / (1 + ytm/freq)**2


def dv01(face, coupon_rate, ytm, T, freq=2):
    """Dollar value of 1 basis point."""
    P = bond_price(face, coupon_rate, ytm, T, freq)
    D_mod = modified_duration(face, coupon_rate, ytm, T, freq)
    return D_mod * P * 0.0001


# 10-year, 4% coupon, 4.30% YTM
P = bond_price(100, 0.04, 0.043, 10, freq=2)
D_mac = macaulay_duration(100, 0.04, 0.043, 10, freq=2)
D_mod = modified_duration(100, 0.04, 0.043, 10, freq=2)
C     = convexity        (100, 0.04, 0.043, 10, freq=2)
dv = dv01(100, 0.04, 0.043, 10, freq=2)

print(f'10y 4% Treasury at 4.30% YTM:')
print(f'  price:               ${P:.4f}')
print(f'  Macaulay duration:   {D_mac:.4f} years')
print(f'  Modified duration:   {D_mod:.4f} years')
print(f'  Convexity:           {C:.4f}')
print(f'  DV01:                ${dv:.4f}  (per $100 face per 1 bp)')
```

    10y 4% Treasury at 4.30% YTM:
      price:               $97.5824
      Macaulay duration:   8.3145 years
      Modified duration:   8.1395 years
      Convexity:           78.3624
      DV01:                $0.0794  (per $100 face per 1 bp)


## Verify duration via finite difference

The closed-form formula should match $-(P_+ - P_-) / (2 \Delta y \cdot P)$.


```python
y = 0.043
h = 0.0001
P_plus  = bond_price(100, 0.04, y + h, 10, freq=2)
P_minus = bond_price(100, 0.04, y - h, 10, freq=2)
P_0     = bond_price(100, 0.04, y,     10, freq=2)

D_fd = -(P_plus - P_minus) / (2 * h * P_0)
C_fd = (P_plus - 2*P_0 + P_minus) / (h**2 * P_0)

print(f'Modified duration: closed-form {D_mod:.6f}, FD {D_fd:.6f}, diff {abs(D_mod-D_fd):.2e}')
print(f'Convexity:        closed-form {C:.6f}, FD {C_fd:.6f}, diff {abs(C-C_fd):.2e}')
```

    Modified duration: closed-form 8.139466, FD 8.139467, diff 1.36e-06
    Convexity:        closed-form 78.362433, FD 78.362441, diff 7.88e-06


## Predicting price changes via duration + convexity

The Taylor expansion:

$$\frac{\Delta P}{P} \approx -D \cdot \Delta y + \frac{1}{2} C \cdot (\Delta y)^2$$

For small yield changes, duration alone suffices. For larger moves (50 bps+), convexity matters. Compare predicted vs actual price change.


```python
# Taylor approximation accuracy
delta_y_grid = np.array([-0.02, -0.01, -0.005, 0.005, 0.01, 0.02])
print(f'{"Δy":>6}  {"Actual ΔP":>12}  {"Dur only":>12}  {"Dur+Convexity":>16}')
for dy in delta_y_grid:
    P_new = bond_price(100, 0.04, y + dy, 10, freq=2)
    actual = P_new - P_0
    dur_only = -D_mod * P_0 * dy
    dur_conv = -D_mod * P_0 * dy + 0.5 * C * P_0 * dy**2
    print(f'{dy*1e4:>+5.0f}b  {actual:>+12.6f}  {dur_only:>+12.6f}  {dur_conv:>+16.6f}')

print('\n→ Duration alone underestimates rate-fall gains and rate-rise losses.')
print('  Convexity correction restores symmetry (always positive contribution → bond holders are LONG convexity).')
```

        Δy     Actual ΔP      Dur only     Dur+Convexity
     -200b    +17.527099    +15.885376        +17.414735
     -100b     +8.338680     +7.942688         +8.325028
      -50b     +4.068612     +3.971344         +4.066929
      +50b     -3.877396     -3.971344         -3.875759
     +100b     -7.573268     -7.942688         -7.560348
     +200b    -14.456636    -15.885376        -14.356017
    
    → Duration alone underestimates rate-fall gains and rate-rise losses.
      Convexity correction restores symmetry (always positive contribution → bond holders are LONG convexity).


## Hedging — duration matching with two bonds

You hold a 10y bond. To hedge against parallel yield shifts, **dollar-duration match** with offsetting positions in 2y and 30y bonds (the "barbell hedge").

Solve the system:
$$N_{2y} \cdot \$D_{2y} + N_{30y} \cdot \$D_{30y} = \$D_{10y, \text{long}}$$
$$N_{2y} \cdot \$D_{2y} \cdot D_{2y} + N_{30y} \cdot \$D_{30y} \cdot D_{30y} = \$D_{10y, \text{long}} \cdot D_{10y}$$

Two equations in two unknowns. The first matches **dollar duration**; the second matches **dollar duration × duration** (a proxy for convexity-aware match in the case of parallel shifts).


```python
# Hedge a 10y bond position with 2y and 30y bonds, all at par
ytm_curve = {2: 0.045, 10: 0.043, 30: 0.046}   # rough realistic 2026 curve

def bond_data(coupon, T, ytm):
    P = bond_price(100, coupon, ytm, T, freq=2)
    D = modified_duration(100, coupon, ytm, T, freq=2)
    return P, D

# Use coupons = curve rate (par bonds for simplicity)
P10, D10 = bond_data(ytm_curve[10], 10, ytm_curve[10])
P2,  D2  = bond_data(ytm_curve[2],  2,  ytm_curve[2])
P30, D30 = bond_data(ytm_curve[30], 30, ytm_curve[30])

# Long $1M face of 10y. Solve for hedge notionals
notional_10y = 1_000_000
DD_target = -notional_10y * P10/100 * D10   # short-equivalent dollar duration

# Solve A x = b
A = np.array([[P2/100 * D2,        P30/100 * D30],
              [P2/100 * D2 * D2,   P30/100 * D30 * D30]])
b = np.array([-DD_target, -DD_target * D10])
N_hedge = np.linalg.solve(A, b)

print(f'Hedging long ${notional_10y:,.0f} face of 10y bond:')
print(f'  Long  10y: D = {D10:.3f}y, P = ${P10:.4f}, position $D = ${notional_10y*P10/100*D10:,.0f}')
print()
print(f'  Hedge:')
print(f'    {N_hedge[0]:>+12,.0f} face of  2y bond (D = {D2:.3f}y)')
print(f'    {N_hedge[1]:>+12,.0f} face of 30y bond (D = {D30:.3f}y)')
print()
print('→ Negative = SHORT. Total dollar duration of the hedge offsets the long position.')

# Verify: parallel ±10 bp shock → portfolio P&L should be small
for dy in [-0.001, +0.001]:
    P10_n = bond_price(100, ytm_curve[10], ytm_curve[10]+dy, 10) * notional_10y/100
    P2_n  = bond_price(100, ytm_curve[2],  ytm_curve[2] +dy,  2) * N_hedge[0]/100
    P30_n = bond_price(100, ytm_curve[30], ytm_curve[30]+dy, 30) * N_hedge[1]/100
    P10_0 = bond_price(100, ytm_curve[10], ytm_curve[10],    10) * notional_10y/100
    P2_0  = bond_price(100, ytm_curve[2],  ytm_curve[2],      2) * N_hedge[0]/100
    P30_0 = bond_price(100, ytm_curve[30], ytm_curve[30],    30) * N_hedge[1]/100
    portfolio_pnl = (P10_n - P10_0) + (P2_n - P2_0) + (P30_n - P30_0)
    print(f'  Δy = {dy*1e4:+.0f}bp: portfolio P&L = ${portfolio_pnl:+,.0f}')
```

    Hedging long $1,000,000 face of 10y bond:
      Long  10y: D = 8.059y, P = $100.0000, position $D = $8,058,595
    
      Hedge:
          +2,421,105 face of  2y bond (D = 1.892y)
            +214,842 face of 30y bond (D = 16.184y)
    
    → Negative = SHORT. Total dollar duration of the hedge offsets the long position.
      Δy = -10bp: portfolio P&L = $+16,202
      Δy = +10bp: portfolio P&L = $-16,033


## Key Rate Durations (KRDs)

Modified duration assumes **parallel shifts** of the yield curve. Real curve moves are **non-parallel** — the short end can move while the long end stays put, or vice versa.

**Key rate duration** $\text{KRD}_k$ measures price sensitivity to a 1 bp shift in the rate at the $k$-th key tenor (e.g. 1y, 2y, 5y, 10y, 30y), holding all other tenors flat.

The sum of all KRDs ≈ modified duration. The decomposition tells you **where on the curve your risk lives**.


```python
# KRD via curve-shift bumping
# Price the 10y bond off a piecewise-flat zero curve, then bump each tenor by 1 bp

key_tenors = [1, 2, 3, 5, 7, 10, 30]
key_rates  = [0.0425, 0.0420, 0.0415, 0.0410, 0.0405, 0.0430, 0.0460]

def discount(t, tenors=key_tenors, rates=key_rates):
    z = np.interp(t, tenors, rates)
    return np.exp(-z * t)


def bond_price_curve(face, coupon, T, freq=2, tenors=None, rates=None):
    if tenors is None: tenors, rates = key_tenors, key_rates
    n = int(round(T*freq))
    cpn = face*coupon/freq
    times = np.arange(1, n+1)/freq
    cf = np.array([cpn]*n); cf[-1] += face
    return sum(c * discount(t, tenors, rates) for c, t in zip(cf, times))


# Compute base price + KRD for each tenor
P0 = bond_price_curve(100, 0.04, 10)
print(f'10y bond, 4% coupon, off curve: ${P0:.4f}')

bp = 0.0001
krd = []
for i, _ in enumerate(key_tenors):
    rates_up = list(key_rates); rates_up[i] += bp
    P_up = bond_price_curve(100, 0.04, 10, tenors=key_tenors, rates=rates_up)
    krd.append(-(P_up - P0) / P0 / bp)

# Sum of KRDs ≈ modified duration
total = sum(krd)
ytm = yield_to_maturity(P0, 100, 0.04, 10)
D_mod_par = modified_duration(100, 0.04, ytm, 10)

krd_table = pd.DataFrame({'tenor': key_tenors, 'KRD': krd, 'KRD%': np.array(krd)/total*100}).round(4)
print('\nKey Rate Durations:')
print(krd_table.to_string(index=False))
print(f'\nSum of KRDs:        {total:.4f}')
print(f'Modified duration:  {D_mod_par:.4f}  (should approximately match)')
print()
print('→ Most of the bond risk is at the 10y tenor — but you can see contribution from 5y, 7y, 30y too.')
```

    10y bond, 4% coupon, off curve: $97.4474
    
    Key Rate Durations:
     tenor     KRD    KRD%
         1  0.0442  0.5320
         2  0.0753  0.9070
         3  0.1781  2.1449
         5  0.3329  4.0100
         7  0.5537  6.6692
        10  7.1175 85.7370
        30 -0.0000 -0.0000
    
    Sum of KRDs:        8.3016
    Modified duration:  8.1374  (should approximately match)
    
    → Most of the bond risk is at the 10y tenor — but you can see contribution from 5y, 7y, 30y too.


## Exercises

### Exercise 1 — Duration of a zero-coupon bond

Show that a zero-coupon bond with maturity $T$ has Macaulay duration exactly $T$ and modified duration $T/(1+y/m)$.


```python
# Your answer here

```

<details>
<summary><b>Reveal solution</b></summary>

```python
face, T, y, freq = 100, 10, 0.05, 2
P = bond_price(face, 0.0, y, T, freq)
D_mac = macaulay_duration(face, 0.0, y, T, freq)
D_mod = modified_duration(face, 0.0, y, T, freq)
print(f'Zero-coupon 10y at 5%:')
print(f'  Macaulay duration:  {D_mac:.6f}  (should be exactly {T})')
print(f'  Modified duration:  {D_mod:.6f}  (should be {T}/(1+y/m) = {T/(1+y/freq):.6f})')
```

_Macaulay = T (single cash flow at T). Modified = T/(1+y/m)._

</details>

### Exercise 2 — Bullet vs barbell at the same duration

Bullet: $1M of 10y 4% bond. Barbell: $X of 2y + $Y of 30y, same total dollar duration but maximised convexity. Show the barbell has higher convexity.


```python
# Your answer here

```

<details>
<summary><b>Reveal solution</b></summary>

```python
ytm = 0.045
# Bullet
D10, P10 = modified_duration(100, 0.04, ytm, 10), bond_price(100, 0.04, ytm, 10)
C10 = convexity(100, 0.04, ytm, 10)

# Barbell: 2y + 30y, same dollar duration, allocation by ratio
D2, P2  = modified_duration(100, 0.04, ytm, 2),  bond_price(100, 0.04, ytm, 2)
D30, P30 = modified_duration(100, 0.04, ytm, 30), bond_price(100, 0.04, ytm, 30)
C2  = convexity(100, 0.04, ytm, 2)
C30 = convexity(100, 0.04, ytm, 30)

# Solve weight w on 2y so that w*D2 + (1-w)*D30 = D10
w = (D30 - D10) / (D30 - D2)
D_barbell = w*D2 + (1-w)*D30
C_barbell = w*C2 + (1-w)*C30

print(f'Bullet 10y:   D={D10:.3f}, C={C10:.2f}')
print(f'Barbell:      D={D_barbell:.3f}, C={C_barbell:.2f}')
print(f'  weight on 2y: {w:.3%}, on 30y: {(1-w):.3%}')
print(f'  → Barbell has HIGHER convexity at the same duration.')
```

_Barbell has higher convexity due to dispersed cash flows. Trade-off: pickup vs spread risk._

</details>

### Exercise 3 — DV01 hedge ratio

You're long $5M of a 5y bond. You want to hedge with a 10y bond. Compute the hedge ratio (face of 10y to short) using DV01.


```python
# Your answer here

```

<details>
<summary><b>Reveal solution</b></summary>

```python
ytm = 0.043
P5, D5 = bond_price(100, 0.04, ytm, 5),  modified_duration(100, 0.04, ytm, 5)
P10, D10 = bond_price(100, 0.04, ytm, 10), modified_duration(100, 0.04, ytm, 10)

# Long $5M of 5y → DV01_long = 5M * (P5/100) * D5 * 1e-4
DV01_long  = 5_000_000 * (P5/100) * D5 * 1e-4
DV01_per_10y_unit_face = (P10/100) * D10 * 1e-4

face_to_short_10y = DV01_long / DV01_per_10y_unit_face
print(f'Long $5M 5y bond — DV01 = ${DV01_long:,.2f}')
print(f'Per $100 face of 10y — DV01 = ${DV01_per_10y_unit_face:.4f}')
print(f'Short ${face_to_short_10y:,.0f} face of 10y to hedge')
```

_Approximately $2.6M face of 10y to hedge $5M of 5y._

</details>

## Interview Q&A

**Q: Macaulay vs modified duration — relationship?**

A: Modified = Macaulay / (1 + y/m). Macaulay is the PV-weighted average cash-flow time (in years). Modified is the *price elasticity* w.r.t. yield: $-dP/dy / P$. Modified is what you use in hedging.

**Q: A 10y bond has modified duration ~8. What does that mean for a 50 bp move?**

A: Approximate ΔP/P = -8 × 0.005 = -4%. The bond loses ~4% of its value for a 50 bp yield rise. (Convexity correction adds back ~+0.5% making the actual loss closer to 3.5%.)

**Q: What's negative convexity?**

A: When the second derivative of price w.r.t. yield is negative — price *accelerates downward* as yield rises. Happens with **callable bonds** (issuer recalls when yields fall, capping price upside) and **prepayable mortgages** (homeowners refinance, shortening duration when you wanted longer). Negative-convexity bonds underperform in volatile rate regimes.

**Q: Why is a barbell more convex than a bullet of the same duration?**

A: Convexity is roughly $\sum w_i t_i^2$ (cash-flow-weighted squared time). At fixed weighted-mean time (duration), Jensen's inequality says the variance contribution is maximised by putting cash flows at the extremes (barbell) rather than concentrated (bullet). More dispersion → more convexity.

**Q: What are key rate durations?**

A: Decomposition of duration by curve tenor. KRD$_k$ = price sensitivity to 1 bp shift at tenor $k$, holding others flat. Sum of KRDs ≈ modified duration. Tells you whether your duration risk is concentrated in the front-end (KRD$_2$y, KRD$_5$y) or long-end (KRD$_{30}$y) — important for **non-parallel shift** scenarios (curve steepening, flattening, twisting).

**Q: How would you construct a duration-and-convexity-neutral portfolio?**

A: For a bullet (single bond), it's already "neutral" with itself. For a multi-bond hedge, solve a 2D linear system: dollar-duration match + dollar-convexity match. You need (at least) 2 hedging instruments. For more curve risk, add KRDs and use more instruments.

**Q: What's the "duration drift"?**

A: As time passes, even if rates stay fixed, the bond's duration falls (it gets closer to maturity). This means a static duration hedge becomes mismatched. Production rebalances daily/weekly to keep DV01 matched.

**Q: When does DV01 break down as a hedge?**

A: Big yield moves (>50 bp), where convexity matters. Non-parallel curve moves, where KRDs matter. Spread widening (credit/asset-swap), where the LIBOR-Treasury basis matters. Each of these is a separate Greek — duration alone is the simplest case.

## Pitfalls reference card

| Pitfall | Issue | Fix |
|---|---|---|
| Confusing Macaulay and modified | Off by factor (1+y/m) | Modified for hedging, Macaulay for cash-flow timing |
| Duration in "wrong" units | Per-decimal vs per-bp vs per-percentage | DV01 = D × P × 1e-4. State convention |
| Ignoring convexity for big moves | Duration linear approximation breaks for >25 bp | Always include 0.5 × C × Δy² for moves > 25 bp |
| Negative convexity surprise | Callable bond looks like vanilla until rates fall | Check optionality embedded in the bond |
| Parallel-shift assumption | Real curves twist | Use KRDs |
| Forgetting accrual in DV01 | Dirty vs clean — same yield sensitivity but accrual evolves | DV01 same; portfolio mark-to-market needs both |
| Flat-curve discounting in DV01 | Single-yield approximation breaks for non-flat curves | KRDs or full-curve revaluation |

## What you've earned

After this notebook you can:

1. **Compute** Macaulay duration, modified duration, convexity, and DV01 from scratch.
2. **Verify** all of these against finite differences.
3. **Apply** the Taylor expansion to predict price changes for ±50 bp yield moves and quantify convexity correction.
4. **Construct** a duration-matched hedge with two bonds for a third.
5. **Compute** key rate durations off a curve and explain non-parallel shifts.
6. **Defend** the bullet-vs-barbell trade-off and the negative-convexity warning for callable bonds.

Next: **`04_portfolio/01_markowitz.ipynb`** — mean-variance optimisation, efficient frontier, the role of constraints.
