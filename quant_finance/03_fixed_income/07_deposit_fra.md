# Money-Market Deposits & Forward Rate Agreements (FRAs)

## Why this matters

The **money market** is the very short end of the rates curve — overnight to ~1 year — where banks fund themselves, central banks transmit policy, and trillions of dollars in SOFR / €STR / SONIA settle every day. The two foundational instruments are:

- **Term deposits / unsecured borrowing**: pay simple interest on ACT/360 day-count.
- **Forward Rate Agreements (FRAs)**: lock in a future deposit rate today.

Together they build the short end of every swap curve, and the FRA fair-rate formula is the simplest case of forward-rate no-arbitrage you'll ever derive.

You will be asked, in any rates / treasury / curve-desk interview:

1. **State the simple-interest formula** and explain ACT/360 vs ACT/365.
2. Why is the **effective annual rate** different from the quoted nominal rate?
3. **Derive the no-arbitrage FRA rate** from two deposit rates.
4. What is a **3x6 FRA** and how does it settle?
5. Why is the FRA settled at **fixing** (not at maturity), and what's the **PV correction**?
6. Why do banks **discount cash flows at OIS** but **forecast LIBOR / SOFR fixings on a separate curve**? (Multi-curve world.)
7. **Convexity bias** between FRAs and futures — what is it?

This note covers all seven on the standard $1M ACT/360 deposit + 3x6 FRA pair.

## The 30-second concept

```
Term deposit                              FRA
────────────                              ─────

Today: deposit $N for d days              Today: agree on a rate F for the
       at simple rate r                          future period [T_1, T_2]
                                                 (no cash changes hands)

Maturity:                                 At T_1: pay/receive based on
   receive  N · (1 + r · d/360)                    (F  vs  the market rate
                                                     L observed at T_1)
```

A deposit is "lend money, get simple interest." An FRA is "lock in today the rate at which a deposit *between two future dates* will be done." The FRA's fair price is determined entirely by no-arbitrage against today's deposit curve — it's not a market opinion, it's a math identity.

## Money-market simple interest

The single most important formula in money markets:

$$\text{Interest} \;=\; N \cdot r \cdot \dfrac{d}{B}$$

Plain ASCII:

```
                          d
Interest   =   N  ·  r  · ───
                          B
```

where:
- $N$ = notional borrowed/lent
- $r$ = annualised simple rate (decimal, e.g. 0.05 for 5%)
- $d$ = actual number of calendar days between value date and maturity
- $B$ = day-count basis: **360** for USD / EUR / CHF / JPY money market; **365** for GBP / AUD / CAD money market

The cash returned at maturity is:

$$\text{FV} \;=\; N \cdot \left(1 + r \cdot \dfrac{d}{B}\right)$$

### Day-count conventions in practice

| Currency / market | Day-count | Notes |
|---|---|---|
| **USD** money market, SOFR | **ACT/360** | "30-day month" assumption baked into the 360 |
| **EUR** money market, €STR | **ACT/360** | Same as USD |
| **GBP** money market, SONIA | **ACT/365** | Historical convention; still in force |
| **JPY** money market, TONA | **ACT/365** (since 2021); was ACT/360 | Switched during JPY-LIBOR transition |
| **Government bonds** (most) | ACT/ACT or 30/360 | Bond market uses *compounded* conventions, not simple |

**The day-count drives a real cash difference.** A 6% ACT/360 rate on a 365-day year actually pays 365/360 = 1.39% MORE than a 6% ACT/365 rate. So `6% (ACT/360) ≈ 6.083% (ACT/365)` for the same cash.

### Effective annual rate vs nominal money-market rate

A money-market rate of $r$ on a deposit of $d$ days is a **simple** rate. To convert to an **effective annual** (compounded) rate:

$$\text{EAR} \;=\; \left(1 + r \cdot \dfrac{d}{B}\right)^{B/d} - 1$$

For a 3M (91-day) deposit at 5% ACT/360:

```
EAR  =  (1 + 0.05 · 91/360)^(360/91)  -  1
     =  (1.012639)^(3.9560)  -  1
     =  0.05095
     ≈  5.10 %
```

Slightly higher than the nominal 5% — that's the compounding pickup.

## Forward Rate Agreements — the no-arbitrage derivation

### The setup

You see two deposit rates today:
- $r_{\text{short}}$ for the period $[0, T_1]$ (e.g. 3-month rate)
- $r_{\text{long}}$ for the period $[0, T_2]$ (e.g. 6-month rate)

The FRA gives you the right to deposit at rate $F$ for the **forward period** $[T_1, T_2]$. What's the fair value of $F$ today?

### The no-arbitrage identity

Two ways to deposit \$1 from today until $T_2$:

1. **Direct**: do a single 6-month deposit at $r_{\text{long}}$:
   $$\text{growth}_{\text{direct}} \;=\; 1 + r_{\text{long}} \cdot \dfrac{d_{\text{long}}}{B}$$

2. **Two-stage**: deposit at $r_{\text{short}}$ for the first 3M, then *roll the entire balance* into the FRA-locked forward deposit at $F$:
   $$\text{growth}_{\text{rollover}} \;=\; \left(1 + r_{\text{short}} \cdot \dfrac{d_{\text{short}}}{B}\right) \cdot \left(1 + F \cdot \dfrac{d_{\text{long}} - d_{\text{short}}}{B}\right)$$

By no-arbitrage, these two paths must produce the **same growth** — otherwise you'd short the cheap path and long the rich one for a riskless profit. Set them equal:

$$\boxed{\; 1 + r_{\text{long}} \cdot \dfrac{d_{\text{long}}}{B} \;=\; \left(1 + r_{\text{short}} \cdot \dfrac{d_{\text{short}}}{B}\right) \cdot \left(1 + F \cdot \dfrac{d_{\text{long}} - d_{\text{short}}}{B}\right)\;}$$

Plain ASCII:

```
                                         ⎛                d_short  ⎞     ⎛           d_long - d_short  ⎞
1  +  r_long  ·  d_long/B    =          ⎜ 1 + r_short  · ───────  ⎟  ·  ⎜ 1  +  F · ────────────────── ⎟
                                         ⎝                   B     ⎠     ⎝                  B          ⎠

\______________________________/         \________________________/      \____________________________/
   direct 6M growth              =        first-3M growth                  forward 3M growth (the FRA)
```

### Solving for F

Solve the identity for $F$:

$$F \;=\; \dfrac{1}{\delta} \cdot \left( \dfrac{1 + r_{\text{long}} \cdot d_{\text{long}}/B}{1 + r_{\text{short}} \cdot d_{\text{short}}/B}  \;-\;  1 \right) \quad\text{where}\quad \delta = \dfrac{d_{\text{long}} - d_{\text{short}}}{B}$$

Or, simplifying the combined fraction (this is the elegant form):

$$F \;=\; \dfrac{r_{\text{long}} \cdot t_{\text{long}}  \;-\;  r_{\text{short}} \cdot t_{\text{short}}}{\delta \cdot \left(1 + r_{\text{short}} \cdot t_{\text{short}}\right)} \quad\text{where}\quad t_X = d_X / B$$

Plain ASCII:

```
                  r_long · t_long  -  r_short · t_short
F      =       ─────────────────────────────────────────────────
                  δ  ·  (1  +  r_short · t_short)
```

### Why this form is the most teachable

Lined up against the **continuous-compounded** forward rate (the one your zero-rate exercise uses):

```
Continuous-compounded (WRONG for money market — different compounding!):
                                                                       |
                  r_long · t_long  -  r_short · t_short                |
        f   =   ──────────────────────────────────────                 |
                                δ                                      |
                                                                       |
                                                                       |
Simple-compounded (RIGHT for ACT/360 deposits + FRAs):                 |
                                                                       |
                  r_long · t_long  -  r_short · t_short                |
        F   =   ──────────────────────────────────────                 |
                  δ  ·  (1 + r_short · t_short)                        |
                       \_______________________/                       |
                          the simple-compounding correction            |
```

The only structural difference between the two forms is the **extra `(1 + r_short · t_short)` divisor**. That factor is the "simple-compounding correction" — it captures the fact that for simple rates, gross factors compose multiplicatively, not additively.

For small `r·t`, the correction is close to 1 and the two formulas agree to first order. For longer tenors or higher rates, they diverge — by 5–6 bp in our worked example below.

### Two equivalent forms, same number

| Form | Where it's useful |
|---|---|
| $F = \dfrac{1}{\delta}\left(\dfrac{\text{gross}_{\text{long}}}{\text{gross}_{\text{short}}} - 1\right)$ | Production code; cleanest extension to multi-period strips |
| $F = \dfrac{r_L \cdot t_L - r_S \cdot t_S}{\delta \cdot (1 + r_S \cdot t_S)}$ | Pedagogy; shows the simple-vs-continuous correction in one factor |

Pick whichever maps best to your mental model — both produce the same number to machine precision.

## Worked example — 3x6 FRA

3M and 6M deposit rates quoted today:

```
r_short = 4.5%   d_short =  91 days   →  t_short = 91/360  = 0.252778
r_long  = 4.8%   d_long  = 183 days   →  t_long  = 183/360 = 0.508333
                                          δ      = (183-91)/360 = 0.255556
```

### Apply the formula

```
Numerator    =  r_long  · t_long  -  r_short · t_short
             =  0.048 · 0.508333  -  0.045 · 0.252778
             =  0.024400  -  0.011375
             =  0.013025

Denominator  =  δ  ·  (1 + r_short · t_short)
             =  0.255556  ·  (1  +  0.045 · 0.252778)
             =  0.255556  ·  1.011375
             =  0.258462

F  =  0.013025 / 0.258462
   =  0.050394
   ≈  5.04 %
```

Cross-check with the gross-factor form:

```
gross_long  =  1 + 0.048 · 183/360  =  1.024400
gross_short =  1 + 0.045 ·  91/360  =  1.011375
ratio       =  1.024400 / 1.011375  =  1.012879

F  =  (1.012879  -  1) / 0.255556  =  0.050394    ✓ (matches to 1e-15)
```

### Sanity-check the answer

`F = 5.04%` is **above the short rate (4.5%) and just above the long rate (4.8%)**. That's what you'd expect on an upward-sloping curve — the forward rates over later periods are higher than the spot rates that include those periods. If the curve were inverted (`r_long < r_short`), the FRA rate would be *below* the short rate.

## Implementation

```python
def deposit_interest_act360(notional: float, rate: float, days: int) -> float:
    """Simple interest on an ACT/360 deposit."""
    return notional * rate * days / 360


def deposit_future_value_act360(notional: float, rate: float, days: int) -> float:
    """FV = N · (1 + r · d/360)."""
    return notional * (1 + rate * days / 360)


def fra_rate(r_short: float, days_short: int,
             r_long: float,  days_long: int,
             basis: int = 360) -> float:
    """No-arb forward simple rate over [T_short, T_long]:

           1 + r_long · d_long/B  =  (1 + r_short · d_short/B) · (1 + F · (d_long - d_short)/B)

    Returns F (annualised simple rate, ACT/B).
    """
    gross_long  = 1 + r_long  * days_long  / basis
    gross_short = 1 + r_short * days_short / basis
    return (gross_long / gross_short - 1) * basis / (days_long - days_short)


def effective_annual_rate_from_money_market(rate: float, days: int,
                                            basis: int = 360) -> float:
    """Convert a money-market simple rate to an effective annual rate."""
    return (1 + rate * days / basis) ** (basis / days) - 1
```

### Sanity check

```python
# 3x6 FRA from 3M (91d) and 6M (183d) ACT/360 deposits
fra = fra_rate(r_short=0.045, days_short=91,
               r_long =0.048, days_long =183)
# → 0.050394   ✓

# 3M $1M deposit at 5%
interest = deposit_interest_act360(1_000_000, 0.05, 91)
# → 12,638.89

# EAR for the same 3M deposit
ear = effective_annual_rate_from_money_market(0.05, 91)
# → 0.05095  (5.10%)
```

## Real-world context

### Quoting convention — what "3x6 FRA" means

The market quotes FRAs as `M_start × M_end` where the numbers are **months from spot** to the **start** and **end** of the forward period:

| Quote | Forward window | Common name |
|---|---|---|
| **1×4** | Months 1 to 4 (3M forward starting in 1M) | "one by four" |
| **3×6** | Months 3 to 6 (3M forward starting in 3M) | "three by six" |
| **6×12** | Months 6 to 12 (6M forward starting in 6M) | "six by twelve" |
| **3×9** | Months 3 to 9 (6M forward starting in 3M) | "three by nine" |

The **second number minus the first** is the *length* of the deposit being locked in; the *first* is when it starts.

### FRA settlement — and why it discounts at fixing

A real FRA doesn't actually do a deposit. At **fixing date** (a few days before $T_1$, the start of the period), the floating rate $L$ (e.g. 3M USD SOFR) is observed and a **single cash payment** changes hands, settling the whole contract:

$$\text{Payment at } T_1 \;=\; \dfrac{N \cdot (L - F) \cdot \delta}{1 + L \cdot \delta}$$

Plain ASCII:

```
                    N  ·  (L  -  F)  ·  δ
Payment at T_1  =  ───────────────────────
                       1  +  L  ·  δ
```

**Why the discount factor `1 / (1 + L·δ)`?** Because the *real* economic exposure is the interest difference `N · (L − F) · δ` paid at $T_2$ (the end of the deposit period). FRAs settle at $T_1$, so we discount that payment back from $T_2$ to $T_1$ at the now-known floating rate $L$:

$$\text{PV at } T_1 \;=\; \dfrac{N \cdot (L - F) \cdot \delta}{1 + L \cdot \delta}$$

That's the **canonical FRA settlement formula**. Buyer (long FRA, *receive floating*) pockets `(L − F)` if rates rose above the contracted `F`; loses if rates fell.

### Multi-curve world — discounting vs forecasting

Pre-2008, banks used a **single LIBOR curve** to both:
1. Discount cashflows
2. Forecast future floating fixings

Post-Lehman, the LIBOR-OIS basis blew out to ~360 bps. Banks split:

| Curve | What it does | Built from |
|---|---|---|
| **Discount curve** (OIS / SOFR-OIS) | Discounts all cashflows | OIS swaps, central-bank-target-rate index |
| **Projection curve** (LIBOR / SOFR-3M) | Forecasts future floating-leg fixings | Deposits + FRAs + IRS at the same tenor |

The FRA formula above uses **single-curve math** — both legs on the same compounding curve. In production, the floating-leg fixing comes from the *projection* curve while the FRA's PV is discounted at the *OIS* curve. The single-curve formula is the right intuition; the multi-curve formula is the right number for booking.

### FRA vs interest-rate futures — the convexity adjustment

A **Eurodollar / SOFR future** has nearly the same economic exposure as a 3M FRA, BUT:

| FRA | Future |
|---|---|
| Settles once at fixing, off-exchange (OTC) | Marked-to-market daily on exchange, with margin calls |
| No daily cashflow until settlement | Daily P&L converted to cash (variation margin) |
| Linear in the rate | Quasi-linear, with a **convexity adjustment** because variation margin gets reinvested at the prevailing rate |

The mark-to-market mechanism creates a *positive* expected reinvestment of margin when rates rise (margin received earns higher interest) → futures trade at a slightly **higher implied rate** than the matched FRA. The gap is called the **convexity bias** or **convexity adjustment**:

$$F_{\text{future}} \;-\; F_{\text{FRA}} \;\approx\; \dfrac{1}{2}\,\sigma^2 \cdot T_1 \cdot T_2$$

Plain ASCII:

```
                                    σ²  ·  T_1  ·  T_2
F_future   -   F_FRA      ≈        ────────────────────
                                            2
```

For short tenors this is < 1 bp and ignored; for 5y+ futures it's 5–20 bp and material to curve building. Production curve-builders apply the adjustment when bootstrapping from futures.

## Interview Q&A

**Q: Derive the no-arbitrage FRA rate from two deposit rates.**

A: Two paths from today to $T_2$: (1) direct deposit at $r_{\text{long}}$; (2) deposit at $r_{\text{short}}$ to $T_1$, then enter the FRA at $F$ for the residual period. No-arb requires equal growth: $1 + r_L t_L = (1 + r_S t_S)(1 + F \delta)$. Solve: $F = (1/\delta) \cdot (\text{gross}_L / \text{gross}_S - 1)$. The simpler form is the elegant `(r_L t_L − r_S t_S) / [δ · (1 + r_S t_S)]` — same number, just rearranged.

**Q: Why is the effective annual rate higher than the quoted money-market rate?**

A: Money-market rates are *simple* (no compounding within the deposit period); EAR is *compounded annually*. For a 3M 5% ACT/360 deposit, EAR = `(1.012639)^3.956 − 1 ≈ 5.10%`. The gap widens with rate level and reduces with tenor (for a 1y deposit at 5%, EAR = 5%·365/360 ≈ 5.07%).

**Q: What does "3x6 FRA" mean?**

A: A 3-month forward-starting 3-month deposit rate. Forward period is months 3 through 6. Settles at month 3 (the start of the deposit period) with a discounted payment of `N · (L − F) · δ / (1 + L·δ)` where `L` is the observed fixing.

**Q: Why is the FRA payment discounted at the fixing rate at $T_1$?**

A: The economic interest differential is `N · (L − F) · δ`, paid at $T_2$ (end of the deposit). But FRA settlement is at $T_1$. We discount the differential back from $T_2$ to $T_1$ at the now-known rate `L`, giving the canonical `/(1 + L·δ)` term.

**Q: How is an FRA different from a Eurodollar / SOFR future on the same window?**

A: Same notional exposure to rates, BUT the future is daily-marked-to-market with variation margin. Variation margin earns/loses interest at the prevailing rate, creating a positive expected reinvestment that pushes the future's fair rate above the FRA's by `½·σ²·T_1·T_2` (convexity adjustment). FRA = no daily settlement; future = daily settlement → convexity bias.

**Q: Why must banks use a multi-curve framework post-2008?**

A: The OIS-LIBOR basis blew out to ~360 bps in 2008. Discounting at LIBOR overstates the value of collateralised trades (which actually fund at OIS). Banks split: **OIS curve** for discounting, **LIBOR/SOFR projection curve** for forecasting fixings. The single-curve FRA formula is correct intuition; the production formula uses both curves.

**Q: ACT/360 vs ACT/365 — which currencies use which?**

A: USD, EUR, CHF, JPY money market: **ACT/360**. GBP, AUD, CAD: **ACT/365**. (JPY switched from 360 to 365 during LIBOR transition.) A 6% ACT/360 rate pays ~1.39% more cash than 6% ACT/365 over a full year — the convention matters.

## Pitfalls reference card

| Pitfall | What goes wrong |
|---|---|
| Using the continuous forward formula `(r_L T_L − r_S T_S)/δ` on simple rates | Off by 5–10 bp on a 3x6 FRA, exactly the compounding correction `(1 + r_S t_S)` |
| Hard-coding `/360` everywhere | Breaks for GBP/AUD/CAD (ACT/365). Always parameterise the basis. |
| Treating quoted FRA rate as bp-equivalent to deposit rate | They're both simple ACT/360 rates, but `F` is forward-starting and `r` is spot — the curve shape determines the level relationship |
| Forgetting the `1 / (1 + L·δ)` settlement discount | Off by ~`L·δ/2` ≈ 25–125 bp depending on rate level. The FRA settlement formula MUST discount the differential back from $T_2$ to $T_1$ |
| Ignoring convexity bias when bootstrapping from futures | At 5y+ tenors the gap is 5–20 bp, enough to make curve assertions fail |
| Single-curve discounting in a multi-curve world | OK for intuition, wrong for production — collateralised trades discount at OIS, not at LIBOR |

## What you've earned

- The **money-market simple-interest formula** and what ACT/360 vs ACT/365 actually mean.
- **EAR conversion** from a money-market rate.
- The **no-arbitrage FRA derivation** — two paths, set growths equal, solve for $F$.
- **Two equivalent forms** of the FRA formula, and when to use each.
- The **simple-vs-continuous correction** factor `(1 + r_S t_S)` that distinguishes money-market FRA math from zero-rate forward math.
- **FRA settlement mechanics** including the discount-at-fixing $/(1 + L·δ)$ term.
- **Multi-curve framework** intuition: OIS for discount, LIBOR/SOFR for projection.
- **Convexity bias** between FRAs and futures — why they're priced slightly differently.

This is the foundation for:
- **Curve bootstrapping** at the short end — see `03_curve_building.md` for the integrated workflow.
- **Swap pricing** — every floating leg is a chain of forward rates derived this way.
- **Eurodollar / SOFR futures** — same window, different settlement mechanics.
- **LIBOR Market Model** — simulates the entire forward-rate vector under specific measures; built on the same simple-compounded forwards.
