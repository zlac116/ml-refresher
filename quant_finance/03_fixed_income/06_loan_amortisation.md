# Loan Amortisation — Mortgages, Annuities, Schedules

## Why this matters

Mortgages, auto loans, term loans, and most retail credit products are **amortising** — the borrower pays a level monthly amount that gradually retires the principal. Banks hold trillions of these on their balance sheets, securitise them into MBS, and manage their interest-rate risk through ALM. You can't move around fixed income without understanding amortisation.

You will be asked, in any FI / ALM / structured-products interview:

1. Derive the annuity (level-payment) formula.
2. Explain the interest-vs-principal split — why does interest dominate the early years?
3. **Why is the effective annual rate higher than the nominal rate?** Compute both.
4. **Duration of a fixed-rate mortgage vs a bullet bond of the same maturity** — which is shorter and why?
5. What is **prepayment risk** and why does it make mortgage duration *uncertain*?
6. How does a bank hedge its mortgage book?
7. **Negative convexity in callable / prepayable assets** — what is it and where does it come from?

This note covers all seven on a standard 30-year fixed-rate mortgage.

## The 30-second concept

A fully-amortising loan is just a **finite annuity**: a stream of equal payments that, in total, pay back the principal plus all accrued interest.

```
You borrow P today.
You pay PMT every month for n months.
At the end, balance = 0.
```

Two things happen each month:

1. **Interest accrues** on the outstanding balance:    $\text{interest}_i = B_{i-1} \cdot r$
2. **Principal is paid down** by the rest of the payment: $\text{principal}_i = \text{PMT} - \text{interest}_i$

In month 1 the balance is large, so interest is large and principal repayment is small. As the balance shrinks, the interest portion shrinks and the principal portion grows. The PAYMENT is constant; the SPLIT shifts.

That mechanic is the entire story. Everything below is plugging it into formulas.

### The four formulas

Level monthly payment (annuity formula):

$$\text{PMT} \;=\; P \cdot \frac{r \, (1+r)^n}{(1+r)^n - 1}$$

Per-period split:

$$\text{interest}_i \;=\; B_{i-1} \cdot r \qquad \text{principal}_i \;=\; \text{PMT} - \text{interest}_i \qquad B_i \;=\; B_{i-1} - \text{principal}_i$$

Effective annual rate vs nominal:

$$\text{EAR} \;=\; \left( 1 + \dfrac{r_{\text{nom}}}{m} \right)^{m} - 1$$

Plain ASCII (same formulas, render-safe):

```
                              r * (1 + r)^n
PMT       =     P     ×    -------------------          ← level monthly payment
                              (1 + r)^n  -  1


interest_i   =   B_{i-1}  ×  r                          ← per-period interest

principal_i  =   PMT  -  interest_i                     ← per-period principal repaid

B_i          =   B_{i-1}  -  principal_i                ← updated balance


EAR  =  (1 + r_nominal / m)^m  -  1                     ← effective annual rate
                                                          (different from nominal!)
```

where:
- $P$ = principal borrowed
- $r$ = per-period rate $= r_{\text{nom}}/m$ (for 6% nominal annual, monthly: $r = 0.005$)
- $n$ = total number of payments $= \text{years} \cdot m$
- $m$ = payments per year (12 for monthly, 4 for quarterly)
- $B_i$ = outstanding balance after payment $i$; $B_0 = P$, $B_n = 0$.

## Deriving the annuity formula (from first principles)

Start with the recursion. After applying one month's interest and one payment:

$$B_k \;=\; B_{k-1}(1+r) \;-\; \text{PMT}$$

Iterate from $k=1$:

$$\begin{aligned}
B_1 &= B_0(1+r) - \text{PMT} \\
B_2 &= B_1(1+r) - \text{PMT} = B_0(1+r)^2 - \text{PMT}\,(1+r) - \text{PMT} \\
B_3 &= B_0(1+r)^3 - \text{PMT}\!\left[\,(1+r)^2 + (1+r) + 1\,\right] \\
&\;\;\vdots \\
B_k &= B_0(1+r)^k \;-\; \text{PMT}\sum_{j=0}^{k-1}(1+r)^j
\end{aligned}$$

The geometric series collapses:

$$\sum_{j=0}^{k-1}(1+r)^j \;=\; \frac{(1+r)^k - 1}{r}$$

So:

$$B_k \;=\; P\,(1+r)^k \;-\; \text{PMT} \cdot \frac{(1+r)^k - 1}{r}$$

For the loan to be **fully amortising**, the balance must hit zero after $n$ payments:

$$\begin{aligned}
B_n &= 0 \\
\Rightarrow \quad P\,(1+r)^n &= \text{PMT} \cdot \frac{(1+r)^n - 1}{r} \\
\Rightarrow \quad \boxed{\;\text{PMT} \;=\; P \cdot \frac{r\,(1+r)^n}{(1+r)^n - 1}\;}
\end{aligned}$$

Plain ASCII (same derivation):

```
B_1  =  B_0 (1 + r)  -  PMT
B_2  =  B_1 (1 + r)  -  PMT  =  B_0 (1 + r)^2  -  PMT (1 + r)  -  PMT
B_3  =  B_0 (1 + r)^3  -  PMT [ (1 + r)^2  +  (1 + r)  +  1 ]
 ...
B_k  =  B_0 (1 + r)^k  -  PMT × sum_{j=0}^{k-1} (1 + r)^j

geometric series:   sum_{j=0}^{k-1} (1 + r)^j  =  ((1 + r)^k - 1) / r

so:                  B_k  =  P (1 + r)^k   -   PMT × ((1 + r)^k - 1) / r

set B_n = 0  →  PMT  =  P × r × (1 + r)^n  /  ((1 + r)^n - 1)        ✓
```

That's the formula. It's just "what monthly payment makes the balance reach exactly zero after $n$ months."

## Worked example — 30-year, $200k, 6% mortgage

Inputs: $P = \$200{,}000$, $r_{\text{nom}} = 6\%$, $n = 360$ months, monthly rate $r = 0.005$.

$$\text{PMT} \;=\; 200{,}000 \cdot \frac{0.005 \cdot (1.005)^{360}}{(1.005)^{360} - 1} \;=\; 200{,}000 \cdot \frac{0.005 \cdot 6.0226}{5.0226} \;\approx\; \$1{,}199.10 / \text{month}$$

**Total paid over 30 years:**

$$\text{total payments} = 1{,}199.10 \cdot 360 = \$431{,}676 \qquad \text{total interest} = 431{,}676 - 200{,}000 = \$231{,}676$$

You borrowed \$200k and paid back \$431k — almost as much in interest as in principal. That's the cost of stretching it over 30 years.

**The interest/principal split shifts dramatically:**

| Month | Balance start | Interest | Principal | Note |
|---:|---:|---:|---:|---|
| 1     | $200{,}000 | $1{,}000 | $199    | Only \$199 of the \$1199 retires principal |
| 60    | $186{,}109 | $930   | $269    | After 5 years, principal share has grown ~35% |
| 240   |  $93{,}054 | $465   | $734    | Past the halfway-balance point |
| 360   |       ~$0  |   ~$6  | ~$1{,}193 | Almost pure principal at the end |

For the first ~20 years of a 30-year mortgage, you're paying mostly interest. Critical to understand for refinance decisions: refinancing in year 20 doesn't save much, because by then you're mostly paying down principal anyway.

## Effective vs nominal annual rate

A US mortgage rate quoted as "6.00%" is the **nominal annual rate, compounded monthly**. The **effective annual rate** (the actual yield you pay) is higher because each month's unpaid interest itself accrues interest:

$$\text{EAR} \;=\; \left(1 + \dfrac{r_{\text{nom}}}{m}\right)^{m} - 1 \;=\; \left(1 + \dfrac{0.06}{12}\right)^{12} - 1 \;=\; 1.005^{12} - 1 \;=\; 0.061678 \;=\; 6.168\%$$

So a "6% mortgage" is really 6.168% in pure annual terms. This is the **APR / APY gap** quoted on retail products in many jurisdictions. The difference compounds with rate level — at higher rates the gap widens.

## Implementation

```python
import numpy as np


def monthly_payment(principal: float, annual_rate_nominal: float,
                    years: int) -> float:
    """Level monthly payment via the annuity formula."""
    r = annual_rate_nominal / 12
    n = years * 12
    return principal * r * (1 + r) ** n / ((1 + r) ** n - 1)


def amortisation_schedule(principal: float, annual_rate_nominal: float,
                          years: int) -> dict[str, np.ndarray]:
    """Build the full month-by-month schedule.

    Returns dict with three arrays:
        balance    length n+1, balance[0] = principal, balance[n] ≈ 0
        interest   length n,   interest portion of each payment
        principal  length n,   principal portion of each payment
    """
    r = annual_rate_nominal / 12
    n = years * 12
    pmt = monthly_payment(principal, annual_rate_nominal, years)

    balance      = np.empty(n + 1)
    interest_pay = np.empty(n)
    princ_pay    = np.empty(n)
    balance[0]   = principal

    for i in range(n):
        interest_pay[i] = balance[i] * r
        princ_pay[i]    = pmt - interest_pay[i]
        balance[i + 1]  = balance[i] - princ_pay[i]

    return {"balance": balance, "interest": interest_pay, "principal": princ_pay}


def effective_annual_rate(annual_rate_nominal: float, freq: int = 12) -> float:
    """EAR = (1 + r_nominal/freq)^freq - 1.   Higher than nominal."""
    return (1 + annual_rate_nominal / freq) ** freq - 1
```

### IRR / yield check (sanity)

The cashflows from the *borrower's perspective* are: receive $+P$ at $t=0$, then pay $-\text{PMT}$ for $n$ months. The IRR is the rate $x$ that makes the NPV zero:

$$\text{NPV} \;=\; P + \sum_{i=1}^{n} \frac{-\text{PMT}}{(1+x)^i} \;=\; 0$$

Solve for $x$ and it should equal the per-period contractual rate $r$.

```python
from scipy.optimize import brentq

P, rnom, yrs = 200_000.0, 0.06, 30
n   = yrs * 12
pmt = monthly_payment(P, rnom, yrs)
cfs = np.concatenate(([P], np.full(n, -pmt)))

# IRR is the rate that makes the NPV of cfs equal to zero
irr_monthly = brentq(
    lambda x: sum(c / (1 + x) ** i for i, c in enumerate(cfs)),
    -0.5, 1.0,
)

print(f"IRR per month:    {irr_monthly:.8f}")   # → 0.00500000 (= 6%/12 exactly)
print(f"IRR per year EAR: {(1 + irr_monthly)**12 - 1:.6f}")
```

The IRR exactly equals `0.005` (= the per-period contractual rate). The annuity formula and IRR are two views of the same fact: a fully amortising loan's yield IS the contractual rate.

## Real-world context

### Fixed vs floating-rate loans

| Loan type | Rate behaviour | Bank's hedging concern |
|---|---|---|
| **Fixed-rate** (most US mortgages) | Locked at origination for 30 years | Big duration risk — if rates rise, the bank loses on the book |
| **Floating / ARM** (Adjustable-Rate Mortgage) | Resets to a benchmark (e.g. SOFR + 200bp) every period | Near-zero duration — the rate re-quotes, the price stays near par |
| **Hybrid** (5/1 ARM, 7/1 ARM) | Fixed for first 5/7 years, then floats | Duration of a 5y / 7y bond initially, collapses after reset |

### Amortising vs bullet loans — duration is different

A bullet bond pays small coupons + a big principal lump at maturity. An amortising loan trickles principal back every month. Macaulay D is the PV-weighted average time, so dispersed cashflows pull D down.

| Instrument | Macaulay D | Notes |
|---|---:|---|
| 5y bullet bond, 6% coupon | $\approx 4.4$ yr | Most weight at year 5 |
| 5y amortising loan, 6% rate | $\approx 2.5$ yr | About HALF the bullet — principal returned along the way |
| 30y bullet bond, 6% coupon | $\approx 14$ yr | Most weight in late years |
| 30y fixed mortgage, 6% rate (no prepay) | $\approx 9$ yr | Amortisation alone collapses to ~9y |
| 30y fixed mortgage, 6% rate (with realistic prepay) | $5\text{–}7$ yr | Prepayment cuts further |

**Rule of thumb**: an amortising loan has roughly **half the duration of a bullet of the same maturity**. This drops further when prepayment is modelled.

### Prepayment risk — the big one

US mortgages have a **free option to prepay** (refinance) at any time. So actual cashflows depend on borrower behaviour:

- When rates **fall**, borrowers refinance — the bank loses its high-coupon stream early
- When rates **rise**, borrowers stay put — the bank is stuck with a low-coupon stream that's now sub-market

**The bank is short the prepayment option both ways.** This is what creates **negative convexity** in mortgage-backed securities:

```
For a callable / prepayable asset:

  rates fall          rates rise
  ────────────        ────────────
  price gains         price losses
  capped              uncapped
  (borrower calls)    (no relief)


Net effect:  the price-yield curve BENDS THE WRONG WAY.  Modified duration
is less reliable; you need a more sophisticated stochastic model
(prepayment model + option-adjusted spread / OAS analysis).
```

This is why MBS desks don't just use Macaulay duration — they use **OAS-adjusted duration** that explicitly accounts for the option.

### Where this lives at a bank

| Area | Use |
|---|---|
| **Retail mortgage book** | Origination + servicing; basic amortisation tracking |
| **ALM / Treasury** | Duration matching against deposit liabilities (deposits typically have ~3y behavioural duration → match against ~3y mortgage tranche) |
| **MBS / structured credit desk** | Securitises mortgages into pools; prices the negative convexity |
| **IRRBB regulatory** | Banks compute Economic Value of Equity (EVE) sensitivity on the loan book |

## Interview Q&A

**Q: Derive the annuity payment formula.**

A: Set up the recursion `B_k = B_{k-1}(1+r) - PMT`. Iterate to get `B_k = P(1+r)^k - PMT × [(1+r)^k - 1]/r`. Set `B_n = 0` and solve for PMT. Answer: `PMT = P × r × (1+r)^n / ((1+r)^n - 1)`. Geometric series + boundary condition.

**Q: Why does interest dominate the early payments?**

A: Interest each month = `balance × r`. Early on the balance is at its maximum (= principal), so interest is at its maximum. As principal is paid down, the balance shrinks, so the interest portion shrinks. The PAYMENT is constant; what changes is the split. Mathematically the principal repayment grows at `(1+r)` per period — a geometric series — so the back end is mostly principal.

**Q: What's the effective annual rate on a 6% monthly-compounding mortgage?**

A: `(1 + 0.06/12)^12 - 1 = 6.168%`. The nominal rate doesn't capture intra-year compounding — the EAR does. Always quote EAR when comparing across compounding conventions.

**Q: Duration of a 30-year fixed mortgage — same as a 30-year bond?**

A: No. (1) Amortisation pulls duration down to ~9 years (no prepay) because principal is returned along the way. (2) Prepayment optionality pulls it further down to ~5–7 years effective. Compare to a 30-year zero (duration = 30) or a 30-year coupon bond (duration ≈ 14). Mortgages are *much* shorter-duration than their stated maturity suggests.

**Q: Why are mortgages "negatively convex"?**

A: Borrowers have a free option to prepay. When rates fall (good for fixed-coupon bondholders normally), borrowers refinance — the bank loses the high-coupon stream. The price gain that duration would predict gets capped. When rates rise, borrowers stay put — the bank takes the full loss. Asymmetric payoff → the price-yield curve curves the "wrong" way (concave instead of convex). Modified duration alone is unreliable; need to compute **OAS-adjusted (option-adjusted) duration and convexity**.

**Q: How does a bank hedge its mortgage book?**

A: Match the *effective* duration of the asset book against the duration of the funding (deposits, term debt). Layer in interest-rate swaps and Treasury futures to neutralise residual duration gap. For prepayment risk specifically, hedge with **swaptions** (gain on falling rates, offsetting the prepayment-driven loss) or by holding **interest-only strips** (which have *positive* exposure to prepayment slowdowns).

**Q: Why does the US have 30-year fixed mortgages while most other countries don't?**

A: Government-sponsored entities (Fannie Mae, Freddie Mac, Ginnie Mae) buy mortgages from banks, securitise them as MBS, and guarantee the credit. This absorbs the prepayment + credit risk that would otherwise make 30-year fixed lending unattractive. In the UK / EU most mortgages are 2/3/5-year fixed → reset to floating, because no equivalent securitisation infrastructure exists.

## Pitfalls reference card

| Pitfall | What goes wrong |
|---|---|
| Quoting nominal rate as "the cost" | EAR is the actual cost. For 6% monthly, EAR is 6.17%. Compare loans on EAR, not nominal. |
| Computing duration as if mortgage = bond | Amortisation cuts duration ~50%; prepayment cuts further. Use *effective* duration on projected cashflows. |
| Using Macaulay duration on a callable mortgage | Negative convexity makes it misleading. Use OAS-adjusted duration. |
| Forgetting that the schedule MUST close | Floating-point error accumulates over 360 iterations; `balance[-1]` should be ~0 to within 1e-6. If it doesn't, your annuity formula or rate conversion is wrong. |
| Mixing day-count conventions | A 6% loan compounded daily ≠ monthly ≠ continuous. EAR varies by ~0.02% across conventions — small but matters for fair-value reporting. |
| Treating the borrower IRR as identical to bank yield | Includes origination fees, servicing fees, escrow → borrower's TRUE all-in rate (APR) is higher than nominal. Banks book the spread between funding cost and contractual rate. |

## What you've earned

- The **annuity formula** and where it comes from (geometric series + boundary condition).
- The **interest-vs-principal split** mechanic and why interest dominates early.
- The **nominal vs effective rate** distinction — always quote EAR.
- The **schedule-building algorithm** — recursive, three lines per iteration.
- The **duration of amortising loans** — about half a comparable bullet.
- **Prepayment risk** and why it creates **negative convexity** in mortgages.
- How a bank hedges its mortgage book (duration matching + swaptions).

This is the foundation for:
- **MBS pricing** (uses CPR / PSA prepayment models on top of this schedule)
- **Asset-Liability Management (ALM)** for retail banks
- **Effective duration / OAS analysis** for callable products generally
- **Auto / student / personal loan** analytics — same formulas, different tenors
