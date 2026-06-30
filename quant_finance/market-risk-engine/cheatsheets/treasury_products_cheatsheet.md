# Treasury Products — Gold Reference

Complete reference for every product in the market-risk-engine drills. Ordered **simple → complex**.
Each entry: **(1) what it is → (2) payoff → (3) valuation formula → (4) toy example → (5) intuition → (6) closer (par strike + risk factors)**.

## Contents

**Part 0 — The Master Recipe** (read this FIRST) — [The single identity + 4-step recipe + substitution table](#part-0--the-master-recipe-read-this-first)

**Part I — Cash & Money Market**
1. [FX Spot](#1-fx-spot)
2. [Repo / Reverse Repo](#2-repo--reverse-repo)
3. [Securities Lending](#3-securities-lending)
4. [Amortising Loan](#4-amortising-loan)

**Part II — Forwards**

5. [FX Forward](#5-fx-forward)
6. [FRA — Forward Rate Agreement](#6-fra--forward-rate-agreement)

**Part III — Bonds (cashflow strips)**

7. [Vanilla Bond](#7-vanilla-bond)
8. [Foreign-Currency Bond](#8-foreign-currency-bond)
9. [Linker — Inflation-Linked Bond](#9-linker--inflation-linked-bond)

**Part IV — Swaps**

10. [Interest Rate Swap (IRS)](#10-interest-rate-swap-irs)
11. [Zero-Coupon Inflation Swap (ZCIS)](#11-zero-coupon-inflation-swap-zcis)
12. [Tenor Basis Swap](#12-tenor-basis-swap)
13. [XCCY Basis Swap](#13-xccy-basis-swap)
14. [Total Return Swap (TRS)](#14-total-return-swap-trs)

**Part V — Credit**

15. [Credit Default Swap (CDS)](#15-credit-default-swap-cds)

**Part VI — Options**

16. [FX Option](#16-fx-option)
17. [Swaption](#17-swaption)

**Part VII — Liabilities**

18. [Annuity Liability](#18-annuity-liability)

**Part VIII — Risk Metrics**

19. [Duration · Modified Duration · DV01 · Convexity](#19-risk-metrics--duration-modified-duration-dv01-convexity)

**Part IX — Extended Universe** (E1–E15: T-Bill, CP, Zero-Coupon, FRN, Callable, Convertible, OIS, CMS, Asset Swap, YoY Swap, Cap/Floor, Bermudan Swaption, CDS Index, MBS, Variance Swap, Equity TRS, Commodity)

**Part X — Universal Patterns** ([A linear-in-strike](#pattern-a-linear-in-strike) · [B basis × annuity](#pattern-b-basis--annuity) · [C cashflow strip](#pattern-c-cashflow-strip-pv))

**Part XI — [Risk factor reference](#risk-factor-quick-reference)**

**Part XII — [Convention reminders](#convention-reminders-where-people-slip-up)**

**Appendix — [Real-world ALM caveats](#appendix--real-world-alm-caveats--what-the-drills-gloss-over)**

---

## Universal symbols

```
N        notional (your stake), in domestic currency unless flagged "foreign"
T        maturity in years
τ        year fraction of one accrual period (1.0 annual, 0.25 quarterly, etc.)
K        STRIKE / contracted rate — locked at trade time, never changes
DF(t)    discount factor today for cash arriving at year t       (= exp(-z(t)·t))
DF(t,s)  spread-adjusted discount factor                          (= exp(-(z(t)+s)·t))
S, fx    spot FX rate today (domestic-per-foreign)
b        breakeven inflation rate today (market's implied)
σ        implied volatility
N(·)     standard-normal CDF (for Black-Scholes-style options)
A(T)     annuity factor over a coupon strip   (= Σ τ·DF(t_i))
```

Master skeleton for linear products:    `V = (market_implied − contract) · scale · discount`.

### Reference curve used in every worked example

```
z(1y)  = 4.50%   →  DF(1)  = 0.95600
z(2y)  = 4.40%   →  DF(2)  = 0.91576
z(3y)  = 4.30%   →  DF(3)  = 0.87897
z(4y)  = 4.25%   →  DF(4)  = 0.84366       (interpolated)
z(5y)  = 4.20%   →  DF(5)  = 0.81058
z(10y) = 4.10%   →  DF(10) = 0.66365

Σ DF(1..5)       = 4.40498       ← annuity factor for §10–§13
```

DFs shown to 5 places; numerical outputs computed from full-precision values.

### Quick-jump matrix (find anything in 5 seconds)

| Product | § | Pattern | Primary risk factor |
|---|---|---|---|
| FX Spot | 1 | none (direct) | FX delta |
| Repo / Reverse Repo | 2 | discount × notional | rate |
| Sec Lending | 3 | B (basis × annuity) | rate (small) |
| Amortising Loan | 4 | C (cashflow strip) | rate + credit spread |
| FX Forward | 5 | A (linear-in-strike) | FX delta + rate |
| FRA | 6 | A | rate |
| Vanilla Bond | 7 | C | rate + credit spread |
| Foreign Bond | 8 | C + FX | rate + credit + FX |
| Linker | 9 | C + inflation uplift | rate + breakeven |
| IRS | 10 | A (or C × 2 legs) | rate |
| ZCIS | 11 | A | rate + breakeven |
| Tenor Basis Swap | 12 | B | basis spread + rate |
| XCCY Basis Swap | 13 | B + FX | basis + FX + rate |
| TRS | 14 | reference-asset PV − notional | inherits from reference |
| CDS | 15 | A (with hazard rate substitution) | credit spread |
| FX Option | 16 | Black on forward | FX + vol + rate |
| Swaption | 17 | Black on swap rate | rate + vol |
| Annuity Liability | 18 | C (NEGATIVE sign) | rate + breakeven |
| Duration / DV01 / Convexity | 19 | metric, not product | derived from rate sensitivity |
| Extended Universe (T-bill, FRN, OIS, CMS, MBS, ...) | E1–E15 | same patterns | varies |

---

# Part 0 — The Master Recipe (read this FIRST)

Every product in this document is a special case of **one identity**. Learn this and you can derive any valuation from scratch — no memorisation.

## The single identity

```
                                          (no-arbitrage / risk-neutral price)
                    T
        V₀ =       Σ    E^Q[ CF_t ]  ·  DF(t)
                  t=t₁

            "expected cashflow"  "today's price of $1 at t"
            under the right         (curve-derived)
            risk-neutral measure
```

**That's it.** Every PV formula in this document is `Σ (expected CF) · (discount)`. The trick is knowing **which substitution to apply** to each cashflow.

## The 4-step recipe

```
STEP 1.  Write down every cashflow:    when, in what currency, contingent on what?

STEP 2.  Replace each random cashflow with its expected value under
         the risk-neutral measure — see substitution table below.

STEP 3.  Discount each expected cashflow to today:    multiply by DF(t).
         For credit-risky issuers, use the SPREAD-ADJUSTED DF(t, s).

STEP 4.  Sum.   V = Σ.
```

That's the whole skill. Steps 1, 3, 4 are bookkeeping. **Step 2 — the substitution — is the only piece that requires insight.**

## The substitution table — what replaces each random CF

| CF depends on… | Replace with | Why it works |
|---|---|---|
| **Nothing** (deterministic — e.g. bond coupon) | the cashflow itself | trivially deterministic |
| **A future floating rate `L_i`** (FRA, IRS float leg, FRN) | the curve's implied forward `fwd_i` | `L_i` is a martingale under the `T_{i+1}`-forward measure |
| **An inflation-index ratio at T** (ZCIS, Linker) | `(1 + b)^T` where `b` is breakeven | breakeven IS the market's implied inflation |
| **A spot FX rate at T** (FX forward) | `S · DF_for(T)/DF_dom(T)` (the forward FX) | interest-rate parity (no-arb between currencies) |
| **An option payoff `max(S_T − K, 0)`** (caplet, FX option, swaption) | `DF(T) · [F · N(d1) − K · N(d2)]` — Black formula on the forward × discount | Black model: forward is the martingale, vol is known. The `DF(T)` is the standard prefactor — already-discounted form. |
| **A survival event up to t** (CDS premium leg) | `Q(t) = exp(−λ·t)` (survival probability) | hazard-rate model gives `Q` |
| **A default event in (t_{i-1}, t_i]** (CDS protection leg) | `Q(t_{i-1}) − Q(t_i)` | survival difference |
| **A SWAP RATE at t** (CMS) | forward swap rate + **convexity adjustment** | the swap rate is NOT a martingale → needs a correction |

The first 7 rows cover ~95% of treasury products. The 8th (CMS / quanto / non-trivial measure changes) is where the math gets harder and you need explicit measure-change machinery.

## Three worked walk-throughs of the recipe

### Walk-through 1 — FRA (replace floating rate with curve forward)

```
STEP 1.  Cashflow:  pay N · K · τ        at T₂        (you contracted)
                    receive N · L · τ    at T₂        (L observed at T₁, random)
                    Net:  N · (L − K) · τ              at T₂

STEP 2.  L is a floating rate — replace with the curve's implied forward:
                fwd  =  (DF(T₁)/DF(T₂) − 1) / τ

STEP 3.  Discount the expected net to today:    · DF(T₂)

STEP 4.  Sum (single CF):
            V  =  N · (fwd − K) · τ · DF(T₂)
```

That IS the FRA formula in §6. You didn't memorise it — you derived it.

### Walk-through 2 — ZC Inflation Swap (replace realised inflation with breakeven)

```
STEP 1.  Cashflow:  receive  N · (1 + π_real)^T      at T   (random)
                    pay      N · (1 + K)^T            at T   (fixed)
                    Net:     N · [(1+π_real)^T − (1+K)^T]

STEP 2.  Replace (1 + π_real)^T with (1 + b)^T  where b = market breakeven:
            E^Q[(1+π_real)^T]  =  (1+b)^T

STEP 3.  Discount to today:    · DF(T)

STEP 4.  Sum:
            V  =  N · DF(T) · [(1+b)^T − (1+K)^T]
```

That's §11. Same recipe, different substitution.

### Walk-through 3 — CDS (replace survival/default events with hazard probabilities)

```
STEP 1.  Premium leg cashflows:  pay  N · K · τ  at each tᵢ  IF still alive at tᵢ
         Protection leg:         receive  N · (1−R)  at default time τ_default

STEP 2.  Replace "alive at tᵢ" with Q(tᵢ) = exp(−λ tᵢ)
         Replace "default in (tᵢ₋₁, tᵢ]" with Q(tᵢ₋₁) − Q(tᵢ)

STEP 3.  Discount each leg's expected CF to today: · DF(t)

STEP 4.  Sum (premium leg − protection leg, from buyer's perspective):
            V_buyer  =  N · [Σ(Q(tᵢ₋₁) − Q(tᵢ)) · (1−R) · DF(tᵢ)        ← protection PV
                            − Σ K · τ · Q(tᵢ) · DF(tᵢ)]                  ← premium PV
```

That's §15. Three recipes, three products — all derived top-down from the same identity.

## The three "magic" substitutions you must internalise

These are the no-arbitrage SHORTCUTS that make Step 2 quick. Memorise these three:

```
1. FORWARD-RATE replacement:
      E^Q[ L(T) ]  =  (DF(T₁) / DF(T₂) − 1) / τ           ← curve-implied forward
   Use for: FRA, IRS float leg, FRN, caplet underlying.


2. RISK-NEUTRAL FORWARD replacement:
      E^Q[ S(T) ]  =  S₀ · DF_for(T) / DF_dom(T)           ← FX forward
      E^Q[ S(T) ]  =  S₀ / DF(T)                            ← stock forward (with no divs)
   Use for: FX forward, equity forward, option underlyings.


3. SURVIVAL-PROBABILITY replacement:
      E^Q[ 1_{still alive at t} ]  =  Q(t)  =  exp(−λt)    ← hazard-rate model
   Use for: CDS, credit-risky bonds, CVA on any product.
```

Plus the trivial fourth: **deterministic CF → itself** (bond coupons, principal, fixed-leg payments).

## When the recipe needs adjustment (advanced — flag these and look up)

Most products in this doc work with steps 1-4 unchanged. A few require **convexity / quanto / measure-change corrections** the recipe doesn't capture:

| Situation | Why the simple recipe fails | What to add |
|---|---|---|
| **CMS** (swap rate paid in cash) | Swap rate isn't a martingale under the payment measure | Convexity adjustment (Hagan replication via swaptions) |
| **In-arrears FRA** (rate paid AT fixing, not later) | Compounding date ≠ payment date → bias | Timing/convexity adjustment |
| **Quanto option** (foreign asset paying in domestic) | Foreign asset isn't a martingale in domestic measure | Quanto adjustment via correlation × vol × vol |
| **YoY inflation** (vs ZCIS) | Year-by-year payments under nominal measure | Inflation/rate correlation adjustment |
| **Bermudan / American optionality** | Holder optimally exercises → no analytical closed form | LSMC, lattice, PDE |
| **Defaultable counterparty** (any product) | Subject to your COUNTERPARTY'S default | CVA / DVA — bilateral xVA |

**Practical rule**: if step 2 requires you to compute the expectation of a random variable under a measure where it ISN'T naturally a martingale, you need a convexity adjustment. The 99% case (FRA, IRS, ZCIS, vanilla bond, vanilla CDS) doesn't.

## The "no-arbitrage replication" mental model (alternative derivation)

Equivalent way to derive any of these — DON'T use risk-neutral expectations, build the product from STATIC INSTRUMENTS instead.

```
Want to value:  contract paying f(market) at T

Find a self-financing portfolio of TRADED instruments that pays the same f(market) at T.

No-arbitrage:  contract value today  =  cost of the replicating portfolio today.
```

Example for FX Forward (parallel to walk-through 2):
- Want: at T, pay K_dom and receive 1 foreign.
- Replicate today:
  - Sell `DF_for(T)` foreign bonds → receive `S · DF_for(T)` in domestic today; owe 1 foreign at T ✓
  - Buy `K · DF_dom(T)` domestic bonds → pay `K · DF_dom(T)` today; receive K at T ✓
- Net cost today = `K · DF_dom(T) − S · DF_for(T)`.
- For a 0-cost forward (par strike): set net = 0 → `K_par = S · DF_for(T)/DF_dom(T)`. ✓

This is the "deposit-replication" argument from `03_curve_building.md`. Both methods give the same answer; pick whichever is faster on a given problem.

## How to use this when DERIVING a new product

```
1. Read the contract: list every cashflow, who pays whom, when, contingent on what.
2. For each cashflow, classify the underlying (deterministic / floating rate / FX /
   inflation / survival / option).
3. Apply the matching substitution from the table.
4. Multiply each expected CF by its DF(t).
5. Sum.
6. Sanity check: does it reduce to a known case at boundary values?
   (e.g. set b = K → ZCIS should give 0;  set σ → 0 → option → intrinsic value;
   set λ → 0 → CDS protection → 0.)
```

Any cheatsheet entry below is just the result of running this 6-step process on a particular contract. You can re-derive any of them from a blank sheet.

## The Portfolio Recipe (companion to the pricing recipe)

The Master Recipe values ONE contract. The Portfolio Recipe sizes a HEDGE BOOK across many contracts. Same 4-step structure, different verbs:

```
STEP 1.  EXPOSURE VECTOR.
         For each portfolio position, compute its sensitivities — bucket by:
         {DV01 per tenor (KRD: 2y, 5y, 10y, 30y), IE01, CS01, FX delta, vega per tenor}.

STEP 2.  AGGREGATE TO PORTFOLIO LEVEL.
         Sum signed exposures across all instruments (assets POSITIVE,
         liabilities NEGATIVE) → net exposure vector for the book.

STEP 3.  CHOOSE A HEDGING SET.
         Pick a small basket of liquid instruments (IRS strip at 2y/5y/10y/30y,
         Linker strip, nominal bonds, OIS, ZCIS) — one per factor you want
         to neutralise.

STEP 4.  SOLVE FOR HEDGE NOTIONALS.
         Linear system: HEDGE_NOTIONALS · HEDGE_EXPOSURES = −NET_EXPOSURE.
         Residual: report any unhedged buckets (typically long-end KRD,
         AA-vs-swap basis, prepay risk) as the "stuff you can't hedge".
```

**Sanity check**: after applying the hedge, re-run a stress scenario (parallel ±100 bp, steepener, +50 bp inflation, +50 bp credit) on the COMBINED book. Residual P&L should be ≪ unhedged. Anything that's not is your remaining basis risk — surface it explicitly.

This is the meta-skill for ALM, LDI, treasury risk: think in vectors of factors, hedge factor-by-factor, residualise what's left.

---

# Part I — Cash & Money Market

## 1. FX Spot

**What**: own foreign currency. No maturity, no interest, no contract — you just hold the asset.

**Payoff today**: `N_for` units of foreign currency, worth `N_for · S · fx_factor` in domestic.

**Value today (domestic currency)**:

```
V  =  N_for · S · fx_factor
       foreign      today's FX rate
       amount       (× 1 + fx_shock)
```

**Toy example** (`FXSpot(80.0, 1.25)`):

| Input | Value | Meaning |
|---|---|---|
| `N_for` | 80 | own 80 units of foreign currency |
| `S` | 1.25 | spot = 1.25 (domestic per foreign) |
| `fx_factor` | 1.0 | no FX shock applied |

```
V  =  80 · 1.25 · 1.0   =   100.00
```

**Intuition**: pure FX exposure — every 1% FX move changes value by 1%. No rate sensitivity (no DF), no time (no τ or T).

**Risk factors**: FX (delta only). FX delta = `N_for · S`.

---

## 2. Repo / Reverse Repo

**What**: secured short-term borrowing (repo) or lending (reverse repo). You sell a security today for cash, agree to buy it back at a higher price at maturity. The price difference IS the interest.

**Payoff at T** (reverse repo = lender): receive `N · (1 + r_repo · T)`.

**Value today** (PV of the maturity cashflow minus the cash you put up):

```
V  =  N · (1 + r_repo · T) · DF(T)  −  N
       PV of repaid cash + interest      cash you advanced
                                          (cost basis)
```

For the borrower (repo): the value is the negative — you owe that PV.

**Toy example** (`Repo(100.0, 0.043, 0.5, is_reverse=True)`):

| Input | Value | Meaning |
|---|---|---|
| `N` | 100 | $100 of cash advanced |
| `r_repo` | 0.043 | repo rate 4.3% (simple interest, ACT/360-ish) |
| `T` | 0.5 | 6-month repo |
| `DF(0.5)` | ≈ 0.978 | half-year discount factor from the curve |

```
maturity cash  =  100 · (1 + 0.043 · 0.5)   =   102.15
PV             =  102.15 · 0.978             =    99.88
V              =  99.88 − 100                =   −0.12
```

**Intuition**: at trade time, repo rate ≈ short funding rate → V ≈ 0. The small residual reflects that the contracted repo rate differs slightly from the curve's implied funding rate. **If the repo rate > funding rate** → reverse repo is a positive NPV trade (you lent above market).

**Risk factors**: rate (very small — short tenor → tiny DV01).

---

## 3. Securities Lending

**What**: lend out securities you own; receive a fee every period.

**Payoff at each `t_i`**: receive `(fee_bp/10000) · N · τ` paid at the end of each period.

**Value today**:

```
V  =  (fee_bp / 10000) · N · Σ τ · DF(t_i)
       └──────┬───────┘   ↑   └──────┬────────┘
       fee in decimal   notional    annuity factor
                                     (sum of DFs)
```

**Toy example** (`SecLending(100.0, 20.0, (1,2,3), 1.0)`):

| Input | Value | Meaning |
|---|---|---|
| `N` | 100 | $100 of securities lent |
| `fee_bp` | 20.0 | 20 bp annual fee |
| `times` | (1,2,3) | fees paid at end of years 1, 2, 3 |
| `τ` | 1.0 | annual accrual |
| `Σ τ·DF` | 0.95600+0.91576+0.87897 = 2.75073 | 3-year annuity factor |

```
V  =  0.002 · 100 · 2.75073   =   0.5501
```

**Intuition**: positive PV equal to the PV of three years of 20 bp fees. Pure "basis × annuity" pattern — same skeleton as the basis swaps.

**Risk factors**: rate (via annuity DFs — tiny DV01).

---

## 4. Amortising Loan

**What**: you lent money; the borrower repays equal principal each period plus interest on the remaining balance. (Differs from a mortgage's level-payment annuity — this is straight-line principal amortisation.)

**Payoff each period** (year `i` of `n`):  receive `amort + c · outstanding_{i-1}` where `amort = principal/n`.

**Value today** (each cashflow discounted at a SPREAD-ADJUSTED rate — borrower's credit risk):

```
                  n
   V  =        Σ      cf_i  ·  DF(i, spread_bp)
                  i=1
                  PV of declining-balance interest +
                  constant principal amortisation,
                  discounted at credit-adjusted curve
```

**Toy example** (`Loan(100.0, 0.05, 5, 200.0)`):

| Input | Value | Meaning |
|---|---|---|
| `principal` | 100 | $100 lent |
| `coupon` | 0.05 | 5% on remaining balance each year |
| `n` | 5 | 5 annual amortisation payments |
| `spread_bp` | 200 | borrower spread 200 bp (credit risk) |
| `amort` | 20 | $20 principal repayment each year |

```
year 1: cf = 20 + 0.05·100 = 25     (outstanding now 80)
year 2: cf = 20 + 0.05·80  = 24     (outstanding now 60)
year 3: cf = 20 + 0.05·60  = 23     (outstanding now 40)
year 4: cf = 20 + 0.05·40  = 22     (outstanding now 20)
year 5: cf = 20 + 0.05·20  = 21     (outstanding now 0)

V  =  Σ cf_i · DF(i, spread+200bp)   ≈   96.12
```

**Intuition**: PV < $100 principal because the borrower's spread (200 bp) is **higher than the loan coupon** (5% vs ~4.2% risk-free + 2% spread = 6.2% required) — so this loan is "under-coupon" for its credit risk, and trades below par. **If coupon > risk-free + spread** → loan trades above par.

**Risk factors**: rate (DV01), credit spread (CS01 — bump `spread_bp` +1).

---

# Part II — Forwards

## 5. FX Forward

**What**: agree TODAY to swap currencies at a pre-agreed rate `K` on date `T`.

**Payoff at T**: pay `N_for · K` domestic, receive `N_for` foreign.

**Value today** (mark-to-market):

```
V  =  N_for · ( S · fx_factor  −  K ) · DF(T)
       size     today's spot vs contract   PV
                (per unit of foreign)
```

**Toy example** (`FXForward(100.0, 1.25, 1.28, 1.0)`):

| Input | Value | Meaning |
|---|---|---|
| `N_for` | 100 | will buy 100 foreign at maturity |
| `S` | 1.25 | today's spot |
| `K` | 1.28 | contracted rate (locked at trade time) |
| `T` | 1.0 | 1 year |
| `DF(1)` | 0.95600 | one-year discount factor |

```
V  =  100 · (1.25 − 1.28) · 0.95600   =   −2.87
```

**Intuition**: you agreed to BUY foreign at 1.28 but it's worth 1.25 today → ~3 cents loss per unit, PV'd back. Output = what you'd pay today to cancel.

**Par strike**: `K_par = S · fx_factor` (simplified drill form) | `K_par = S · DF_for(T)/DF_dom(T)` (full IRP).
**Risk factors**: FX delta, rate via `DF(T)`.

---

## 6. FRA — Forward Rate Agreement

**What**: agree TODAY on the interest rate `K` to apply to a future deposit period `[T_1, T_2]`.

**Payoff at T_2** (receive-floating, pay-fixed = long FRA): `N · (L − K) · τ` where `L` is the fixing at `T_1`.

**Value today** (replace unknown `L` with the curve's implied forward):

```
                  DF(T_1)
   fwd  =      ( ────────  −  1 )  /  τ                    ← simple-comp forward
                  DF(T_2)


   V    =   N · ( fwd  −  K ) · τ · DF(T_2)
            └─┬┘ └────┬─────┘   ↑       ↑
           size   market vs    accrual  PV
                  contract     factor
```

**Toy example** (`FRA(100.0, 1.0, 2.0, 1.0, 0.043)`):

| Input | Value | Meaning |
|---|---|---|
| `N` | 100 | notional |
| `T_1, T_2` | 1.0, 2.0 | forward window: year 1 to year 2 |
| `τ` | 1.0 | annual |
| `K` | 0.043 | locked rate 4.30% |
| `fwd` | `(0.95600/0.91576 − 1)/1.0 = 0.04394` | market-implied forward |

```
V  =  100 · (0.04394 − 0.043) · 1.0 · 0.91576   =   0.0859
```

**Intuition**: market is pricing 4.39% but you locked in 4.30% — you're 9 bp better off, scaled by notional × τ, discounted.

**Par strike**: `K_par = fwd`.
**Risk factors**: rate only.

> **MTM vs settlement**: the formula above is the curve-based MTM. In real settlement the cash changes hands at `T_1` discounted at the observed fixing `L`: `payment = N·(L−K)·τ / (1 + L·τ)`. The two agree in expectation under the `T_2`-forward measure; the MTM form is simpler because it only needs the curve.

---

# Part III — Bonds

## 7. Vanilla Bond

**What**: a coupon-paying bond — issuer promises a stream of cashflows on fixed dates.

**Payoff each `t_i`**: receive `cf_i` (coupons on most dates, coupon + principal at maturity).

**Value today** (each cashflow discounted at spread-adjusted curve = issuer's credit risk):

```
   V  =  Σ  cf_i · DF(t_i, spread_bp)            DF(t, s) = exp(-(z(t) + s/1e4) · t)
         ↑
         sum over all cashflow dates
```

**Toy example** (`Bond((1,2,3,4,5), (4,4,4,4,104), 150.0)`):

| Input | Value | Meaning |
|---|---|---|
| `times` | (1,2,3,4,5) | annual coupon dates |
| `cashflows` | (4, 4, 4, 4, 104) | $4 coupons + $100 principal at year 5 |
| `spread_bp` | 150 | issuer spread = 150 bp over the risk-free curve |

```
DF(1, +150bp) = exp(-(0.045 + 0.015)·1) = 0.9418     PV_1 = 4 · 0.9418 = 3.767
DF(2, +150bp) = exp(-(0.044 + 0.015)·2) = 0.8884     PV_2 = 4 · 0.8884 = 3.554
DF(3, +150bp) = exp(-(0.043 + 0.015)·3) = 0.8404     PV_3 = 4 · 0.8404 = 3.361
DF(4, +150bp) = exp(-(0.0425+0.015)·4) = 0.7945     PV_4 = 4 · 0.7945 = 3.178
DF(5, +150bp) = exp(-(0.042 + 0.015)·5) = 0.7515     PV_5 = 104 · 0.7515 = 78.211

V  ≈  92.07
```

**Intuition**: standard PV of fixed cashflows. The spread (`150 bp`) is the issuer's credit premium — wider spread → lower DFs → lower price. Below par because the coupon (4%) is lower than the issuer's required yield (~5.7% = risk-free + 150 bp).

**Par coupon** (the coupon that prices the bond at 100): solve `Σ c · DF(t,s) + 100·DF(T,s) = 100` → `c = (1 − DF(T,s)) / Σ DF(t,s)`. Same identity as the IRS par swap rate (§10).

**Risk factors**: rate (DV01), credit spread (CS01).

---

## 8. Foreign-Currency Bond

**What**: a bond denominated in foreign currency. PV is the local-currency PV translated to domestic at spot.

**Value today (domestic currency)**:

```
V  =  ( Σ cf_i · DF(t_i, spread_bp) )  ·  S · fx_factor
       local-currency PV (like §7)         FX conversion
```

**Toy example** (`ForeignBond((1,2,3,4,5), (3,3,3,3,103), 100.0, 1.25)`):

| Input | Value | Meaning |
|---|---|---|
| `times`, `cashflows` | (1..5), (3,3,3,3,103) | 3% coupons + principal in foreign |
| `spread_bp` | 100 | issuer spread 100 bp |
| `S, fx_factor` | 1.25, 1.0 | spot, no FX shock |

```
Local PV  ≈  89.94                            (compute like §7 but with spread=100bp)
V         =  89.94 · 1.25 · 1.0  =  112.43
```

**Intuition**: same as Vanilla Bond but multiplied by current FX rate. Risk = sum of foreign-curve risk AND FX delta.

**Risk factors**: rate (foreign curve via DFs), credit spread, FX (delta = local PV).

---

## 9. Linker — Inflation-Linked Bond

**What**: bond whose every cashflow is uplifted by accumulated inflation between issue and payment.

**Payoff at each `t_i`**: real coupon `c_t` × index ratio `(1 + π_realised)^t`.

**Value today** (replace realised inflation with market breakeven `b`):

```
   V  =  Σ  cf_real_i  ·  (1 + b)^t_i  ·  DF(t_i)
            real          inflation         discount
            cashflow      uplift            (nominal curve)
```

**Toy example** (`Linker((1,2,3,4,5), (1.5,1.5,1.5,1.5,101.5), 0.03)`):

| Input | Value | Meaning |
|---|---|---|
| `times` | (1..5) | annual |
| `real_cfs` | (1.5, 1.5, 1.5, 1.5, 101.5) | 1.5% real coupons + 100 principal at year 5 |
| `b` | 0.03 | breakeven inflation 3% |

```
y1:    1.5 · 1.03^1   · 0.95600  =   1.477
y2:    1.5 · 1.03^2   · 0.91576  =   1.457
y3:    1.5 · 1.03^3   · 0.87897  =   1.441
y4:    1.5 · 1.03^4   · 0.84366  =   1.424
y5:  101.5 · 1.03^5   · 0.81058  =  95.379
                              V ≈  101.178
```

**Intuition**: standard bond PV but cashflows are uplifted by `(1+b)^t` — higher breakeven → higher payments → higher PV. Linker is **long inflation**.

**Risk factors**: rate (DV01 via DFs), inflation breakeven (IE01).

> **Indexation lag**: in production, the inflation index used at payment date is typically the CPI level from 3M earlier (UK index-linked gilts) or 8M earlier (US TIPS) — adds a small basis between true real cashflows and contractual ones. Drill formula ignores this.

---

# Part IV — Swaps

## 10. Interest Rate Swap (IRS)

**What**: agree TODAY to exchange a fixed-rate stream (`c` on `N` per period) for a floating-rate stream (`L_i` on `N` per period) on the same dates.

**Payoff each `t_i`** (receive-fixed): receive `N · c · τ`, pay `N · L_i · τ`.

**Value today** (receive-fixed perspective; uses telescoping identity for the floating leg):

```
   V  =  N  ·  [ c · τ · Σ DF(t_i)  −  ( 1 − DF(T) ) ]
                fixed leg PV               floating leg PV
                (N · c · A(T))             (single-curve telescope)
```

For pay-fixed, flip the sign.

**Toy example** (`InterestRateSwap((1,2,3,4,5), 1.0, 0.042, 100.0, receive_fixed=True)`):

| Input | Value | Meaning |
|---|---|---|
| `N` | 100 | notional |
| `times`, `τ` | (1..5), 1.0 | 5 annual payments |
| `c` | 0.042 | contracted fixed rate 4.20% |
| `Σ DF(1..5)` | 4.40498 | annuity factor |
| `DF(5)` | 0.81058 |  |

```
fixed leg PV  =  100 · 0.042 · 4.40498   =   18.501
float leg PV  =  100 · (1 − 0.81058)      =   18.942
V             =  18.501 − 18.942          =   −0.441
```

**Intuition**: you contracted to receive 4.20% but the curve's par rate today is higher (~4.30%), so you're under-paid by ~10 bp on $100 over 5 years — small negative PV.

**Par swap rate**: `c_par = (1 − DF(T)) / Σ τ·DF(t_i)` — the rate that makes the swap worth zero today.
**Risk factors**: rate only (huge DV01 vs other products — IRS is THE rates instrument).

---

## 11. Zero-Coupon Inflation Swap (ZCIS)

**What**: agree TODAY to exchange a single payment of `N · (1 + π_realised)^T` (floating, inflation) for `N · (1 + K)^T` (fixed) at maturity `T`.

**Payoff at T** (receiver of inflation): `N · [(1+π_realised)^T − (1+K)^T]`.

**Value today** (replace realised inflation with breakeven):

```
   V  =  N  ·  DF(T)  ·  [ (1 + b)^T  −  (1 + K)^T ]
         └┬┘    ↑          └────────┬───────────────┘
        size   PV         market-compound vs contract-compound
```

**Toy example** (`InflationSwap(100.0, 5.0, 0.03, 0.028)`):

| Input | Value | Meaning |
|---|---|---|
| `N` | 100 |  |
| `T` | 5.0 | matures in 5y |
| `b` | 0.03 | breakeven 3% |
| `K` | 0.028 | locked 2.8% |

```
(1+b)^T  =  1.03^5   =  1.15927
(1+K)^T  =  1.028^5  =  1.14803
gap      =  0.01124

V  =  100 · 0.81058 · 0.01124   ≈   0.909
```

**Intuition**: locked in 2.8% inflation, market is now pricing 3.0% — you gain ~20 bp compounded over 5y, discounted.

**Par strike**: `K_par = b` (formula is symmetric — same `(1+·)^T` on both sides).
**Risk factors**: rate (via `DF(T)`), inflation breakeven.

---

## 12. Tenor Basis Swap

**What**: exchange one floating-rate stream (e.g. 3M LIBOR) for another (e.g. 6M LIBOR) plus a spread. The "basis" is the small spread that compensates one side for the tenor mismatch.

**Payoff per period** (receiver of basis): `N · basis · τ` (the floating legs cancel up to the spread).

**Value today**:

```
   V  =  b · N · Σ τ · DF(t_i)             with  b = (basis_bp + tenor_shock_bp) / 1e4
         basis in
         decimal
```

**Toy example** (`TenorBasisSwap(100.0, (1,2,3,4,5), 1.0, 15.0)`):

| Input | Value | Meaning |
|---|---|---|
| `N` | 100 |  |
| `basis_bp` | 15 | receive 15 bp on top of the equivalent leg |
| `Σ τ·DF` | 4.40498 |  |

```
b  =  0.0015
V  =  0.0015 · 100 · 4.40498   =   0.6607
```

**Intuition**: pure stream of basis-spread payments → annuity × basis × notional.

**Quick rule**: every 1 bp of basis on $N over the strip is worth `N · A(T) · 0.0001` per bp.
**Risk factors**: rate (annuity DFs), tenor-basis spread.

---

## 13. XCCY Basis Swap

**What**: cross-currency basis swap — exchange floating in one currency for floating in another. The basis is the spread that compensates for the currency mismatch (always-positive in practice, FX-hedged cost of foreign funding).

**Payoff**: same as tenor basis swap but the basis leg is in **foreign currency** → FX-sensitive.

**Value today (domestic currency)**:

```
   V  =  N_for · S · fx_factor · b · Σ τ · DF(t_i)
        foreign       FX
        notional      conversion
```

(Principal exchange omitted — included in real-world deals; see caveat at end of doc.)

**Toy example** (`XCCYBasisSwap(100.0, 1.25, (1,2,3,4,5), 1.0, -20.0)`):

| Input | Value | Meaning |
|---|---|---|
| `N_for` | 100 | foreign notional |
| `S` | 1.25 |  |
| `basis_bp` | −20 | foreign side **pays** 20 bp (negative basis) |
| `Σ τ·DF` | 4.40498 |  |

```
b  =  −0.002
V  =  100 · 1.25 · (−0.002) · 4.40498   =   −1.1012
```

**Intuition**: same as tenor basis but in foreign currency. Negative basis = paying the spread → negative PV. Sensitive to FX too (the basis leg's PV depends on `S`).

**Risk factors**: rate (annuity DFs), XCCY-basis spread, FX delta.

---

## 14. Total Return Swap (TRS)

**What**: one party (TRS receiver) gets the total return of a reference asset (price changes + coupons); the other gets a funding rate on a notional. Essentially synthetic ownership.

**Payoff**: receiver of total return gets `PV(reference) − funding_notional`.

**Value today**:

```
   V  =  PV(reference asset)  −  funding_notional
         reprice using current        the financing
         market state                 leg's notional
```

**Toy example** (`TotalReturnSwap(reference=Bond((1..5),(4,4,4,4,104),150.0), funding_notional=90.0)`):

| Input | Value | Meaning |
|---|---|---|
| `reference` | a Bond | the underlying being synthetically owned |
| `funding_notional` | 90 | financing leg notional (= what receiver "borrows") |
| `PV(reference)` | 92.07 | from §7 (Bond) |

```
V  =  92.07 − 90.0   =   2.07
```

**Intuition**: you're synthetically long the bond at 90, but it's worth 92 → you're up 2. As the bond rallies, you gain; as it sells off, you lose. **Same risk exposure as owning the bond**, but balance-sheet-light (no funding needed up front).

**Risk factors**: identical to the reference asset (rate, spread, etc.).

---

# Part V — Credit

## 15. Credit Default Swap (CDS)

**What**: insurance against an issuer's default. Buyer pays a periodic premium `K` (the contractual spread); receives `(1 − R)·N` if default occurs.

**Payoff**:
- buyer pays `N · K · τ · Q(t_i)` each period until default or maturity (premium leg)
- buyer receives `(1 − R) · N` at the default time (protection leg)

**Value today** (buyer perspective, hazard rate `λ` implied from current market spread):

```
   λ  =  market_spread / (1 − R)                 ← credit triangle

   premium leg PV     =  Σ τ · DF(t_i) · Q(t_i)             where Q(t) = exp(-λ·t)
   protection leg PV  =  Σ ( Q(t_{i-1}) − Q(t_i) ) · DF(t_i)

   V_buyer  =  N · [ protection · (1 − R)  −  premium · K ]
```

**Toy example** (`CreditDefaultSwap((1..5), 1.0, contract=100bp, market=120bp, N=100)`):

| Input | Value | Meaning |
|---|---|---|
| `contract_K` | 100 bp | you contracted to pay 100 bp/yr |
| `market spread` | 120 bp | market is now pricing 120 bp |
| `R` | 0.4 | assumed 40% recovery |
| `N` | 100 |  |

```
λ        =  0.012 / 0.6  =  0.02
Premium  =  Σ τ · DF · Q          ≈  4.07
Protect  =  Σ (Q_{i-1} − Q) · DF  ≈  0.083

V_buyer  =  100 · [ 0.083 · 0.6  −  4.07 · 0.01 ]
        =  100 · [ 0.050 − 0.0407 ]
        ≈  0.88
```

**Intuition**: you locked in 100 bp protection; the market now charges 120 bp → your contract is "cheap protection" by 20 bp/yr → positive PV. **Spread widens → buyer wins** (protection becomes more valuable).

**Par spread**: `K_par = market_spread` (set V=0 → contract equals market).
**Risk factors**: credit spread (CS01 dominates), rate (small DV01).

---

# Part VI — Options

## 16. FX Option

**What**: right (not obligation) to exchange currencies at strike `K` on expiry `T`. Call = right to BUY foreign; put = right to SELL foreign.

**Payoff at T** (call): `max(S_T − K, 0) · N`.

**Value today** (Black formula — uses the forward `F` directly):

```
   F   =  S · fx_factor  /  DF(T)                         ← forward FX rate
   v   =  σ + vol_shock                                    ← stressed vol

   d1  =  [ ln(F/K) + ½·v²·T ]  /  ( v · √T )
   d2  =  d1 − v · √T

   V_call  =  N · DF(T) · [ F·N(d1)  −  K·N(d2) ]
   V_put   =  N · DF(T) · [ K·N(−d2) − F·N(−d1) ]
```

**Toy example** (`FXOption(expiry=1.0, strike=1.30, vol=0.12, fx0=1.25, N=100, call=True)`):

| Input | Value | Meaning |
|---|---|---|
| `S` | 1.25 | today's spot |
| `K` | 1.30 | option strike |
| `σ` | 12% | implied vol |
| `T` | 1.0 | 1 year |
| `DF(1)` | 0.95600 |  |

```
F   =  1.25 / 0.95600   =  1.3075
d1  =  [ ln(1.3075/1.30) + ½·0.0144 ]  /  (0.12)    ≈   0.108
d2  =  d1 − 0.12                                     ≈  −0.012

N(d1)  ≈  0.543      N(d2)  ≈  0.495

V_call  =  100 · 0.95600 · [ 1.3075 · 0.543  −  1.30 · 0.495 ]
        =  100 · 0.95600 · [ 0.710 − 0.644 ]
        ≈   6.33
```

**Intuition**: out-of-the-money call (K > S), but the forward is essentially ATM (`F = 1.3075` vs `K = 1.30`) due to interest-rate differential, so the option has real time value. Output = today's premium.

**Greeks**:
- **Delta** = `DF(T) · N(d1)` (call) → "shares of spot to hedge"
- **Gamma** = `DF(T) · φ(d1) / (F · v · √T · DF(T))` → curvature
- **Vega** = `N · DF(T) · F · φ(d1) · √T` → vol sensitivity (per 1.0 vol point)
- **Theta** = negative for long options → time decay

**Risk factors**: FX (delta), vol (vega), rate (rho).

---

## 17. Swaption

**What**: option to enter an interest rate swap at expiry `T_expiry` at strike rate `K`. Payer swaption = right to enter as fixed-PAYER; receiver swaption = right to enter as fixed-RECEIVER.

**Value today** (Black model on the forward swap rate):

```
   A    =  Σ τ · DF(t_i)                                 ← swap annuity
   S    =  [ DF(T_expiry) − DF(T_end) ] / A              ← forward swap rate
   v    =  σ + vol_shock                                   ← stressed vol on swap rate

   d1   =  [ ln(S/K) + ½·v²·T ]  /  ( v · √T )
   d2   =  d1 − v · √T

   V_payer    =  N · A · [ S·N(d1)  −  K·N(d2) ]
   V_receiver =  N · A · [ K·N(−d2) − S·N(−d1) ]
```

**Toy example** (`Swaption(expiry=2.0, pay_times=(3,4,5), τ=1.0, K=0.042, σ=0.20, N=100, payer=True)`):

| Input | Value | Meaning |
|---|---|---|
| `T_expiry` | 2.0 | swaption exercise at year 2 |
| `pay_times` | (3,4,5) | swap pays at years 3, 4, 5 |
| `K` | 0.042 | option to enter at fixed rate 4.20% |
| `σ` | 0.20 | implied vol (lognormal Black) |

```
A   =  DF(3) + DF(4) + DF(5)       =  2.53321
S   =  [DF(2) − DF(5)] / A          =  [0.91576 − 0.81058] / 2.53321  =  0.04153
d1  =  [ln(0.04153/0.042) + ½·0.04·2] / (0.20·√2)   ≈  0.101
d2  =  d1 − 0.20·√2                                  ≈ −0.182

V_payer  =  100 · 2.53321 · [0.04153·0.540 − 0.042·0.428]   ≈   1.13
```

**Intuition**: option to PAY 4.20% for 3 years starting in 2 years. Forward swap rate today is 4.15% — so the option is slightly out-of-the-money for the payer, but still has time value due to vol.

**Greeks**: same families as FX option, but **vega** is the dominant exposure for at-the-money swaptions.
**Risk factors**: rate (delta + rho via DFs), vol (vega).

---

# Part VII — Liabilities

## 18. Annuity Liability

**What**: pension-style obligation — you OWE inflation-linked payments every year for `years` years.

**Payoff each year** (negative for you, the holder): `−payment · (1 + π_realised)^t`.

**Value today** (replace realised inflation with breakeven; **negative sign because liability**):

```
                              years
   V  =  −  payment   ·       Σ      (1 + b)^t  ·  DF(t)
                              t=1     └────────┬────────────┘
                                      inflation-uplifted annuity
```

**Toy example** (`AnnuityLiability(5.0, 10, 0.03)`):

| Input | Value | Meaning |
|---|---|---|
| `payment` | 5.0 | $5 per year real |
| `years` | 10 |  |
| `b` | 0.03 |  |

```
y1:  −5 · 1.03^1  · 0.95600   =   −4.923
y2:  −5 · 1.03^2  · 0.91576   =   −4.857
y3:  −5 · 1.03^3  · 0.87897   =   −4.804
y4:  −5 · 1.03^4  · 0.84366   =   −4.747
y5:  −5 · 1.03^5  · 0.81058   =   −4.701
y6:  −5 · 1.03^6  · 0.78198   =   −4.671
y7:  −5 · 1.03^7  · 0.75417   =   −4.640
y8:  −5 · 1.03^8  · 0.72714   =   −4.611
y9:  −5 · 1.03^9  · 0.70088   =   −4.582
y10: −5 · 1.03^10 · 0.66365   =   −4.464
                          V ≈   −46.79
```

**Intuition**: 10 years of $5 inflation-linked obligations. **Rates down → liability more negative**, **inflation up → liability more negative**. The negative sign matters: a portfolio of assets + liability is `V_assets + V_liab` (liability is already negative).

**Risk profile** (THE pension ALM problem):
- DV01 is **opposite-sign** of a long bond: rates down → liability's negative PV gets MORE negative → you OWE more
- IE01 is **negative** (same sign as Linker on PV, but signed the other way): inflation up → liability more negative
- Linker assets PARTIALLY hedge this — match IE01s to neutralise inflation

**Hedge ratio mechanic**: to neutralise IE01,
```
N_linker · IE01_linker_per_unit_face  =  |IE01_liability|

where  IE01_liability =  −payment · Σ t · (1+b)^(t-1) · DF(t)        (∂V/∂b)
       IE01_linker    =  + Σ cf_real_i · t_i · (1+b)^(t_i-1) · DF(t_i)
```

**Worked numerical hedge** for our annuity (payment=5, 10y, b=3%) using a $100-face 5y Linker (§9, IE01 ≈ +0.05 per $100 face — bump `b` by 1bp, reprice):

```
IE01_liab    ≈  −payment · Σ t · (1+b)^(t-1) · DF(t)  · 0.0001
              ≈  −5 · 38.4 · 0.0001                            =  −0.0192 per +1bp
IE01_linker  ≈  +0.045 per $100 face per +1bp
N_hedge      =  |IE01_liab| / IE01_linker_per_$100  · 100      =  $42.7

→ Hold $42.7 of 5y Linker face value for every $5/yr annuity. PARTIAL hedge
  (Linker's 5y duration < liability's 10y duration → uncovered long-end IE01).
```

Same logic for DV01 — match nominal duration with an IRS or nominal bond. But **parallel DV01 alone isn't enough**: a 10y liability hedged with a 5y IRS is parallel-neutral but **key-rate mismatched** at the long end. Real ALM uses **Key Rate Durations** (KRD at 2y/5y/10y/30y buckets) and matches each bucket separately. See `02_duration_convexity_krd.md` for the KRD machinery.

**Sign convention note**: under the convention DV01 = `−∂V/∂r · 0.0001` (positive for long bond), an annuity liability has **negative DV01** (rates ↑ → V less negative → ∂V/∂r > 0, flipped sign → DV01 < 0). The risk-factor table at the end of this doc uses the alternative `∂V/∂r` convention which gives the liability a POSITIVE sign. Pick one across your stack.

**Risk factors**: rate (large DV01 — long duration on long-dated payments), breakeven inflation (large IE01).

---

# Part VIII — Risk Metrics

## 19. Risk Metrics — Duration, Modified Duration, DV01, Convexity

These are sensitivities you compute ON TOP of any rate-bearing instrument (bonds, swaps, linkers, etc.) — they're not products themselves.

### Macaulay Duration  (`D_mac`, units = YEARS)

PV-weighted average time to cashflows. "How long until I get my money back on average."

```
                Σ  t_i · PV_i
   D_mac  =   ─────────────────             where PV_i = cf_i · DF(t_i)
                Σ  PV_i
```

For a 5y bond paying 4% coupons:  `D_mac ≈ 4.6 years`.

### Modified Duration  (`D_mod`, % price sensitivity)

The rate-sensitivity flavour — derivative of price w.r.t. yield, expressed in %:

```
                  D_mac
   D_mod  =  ────────────────                  (semi-annual bond: m = 2)
                1 + y/m


   ΔP / P   ≈   −D_mod · Δy                   ← per unit Δy (e.g. Δy = 0.01 = 1%)
```

For the same bond: `D_mod ≈ 4.45`. A 100 bp yield rise → price drops ~4.45%.

### DV01 / PV01  (Dollar Value of 1 bp, units = $)

Practical desk metric — dollar P&L per 1 bp rate move:

```
   DV01   =   D_mod · P · 0.0001
```

For $100M of the bond: `DV01 ≈ $42,650 per bp`. Bond falls $42k when rates rise 1 bp.

### Convexity  (`C`, second-order correction)

Curvature of the price-yield relationship — duration is only a tangent line; convexity is the next term in the Taylor expansion:

```
                 Σ  t_i · (t_i + 1/m) · PV_i
   C   =   ─────────────────────────────────
                  P · (1 + y/m)²


   ΔP / P   ≈   −D_mod · Δy   +   ½ · C · (Δy)²
                                     ↑
                                  always pushes IN YOUR FAVOR for long bonds
                                  (curvature makes losses smaller, gains bigger)
```

### Why convexity matters

For small moves duration alone is fine; for large moves you need convexity.

```
+100 bp move on a 30y bond:
   duration alone says:  −12% loss
   actual:               −11.5% loss     (convexity saved you 0.5%)
```

### Numerical (FD) versions — when there's no closed form

For exotic products (MC, lattice, basket), bump the rates and revalue:

```
   DV01_numerical  =  V(rate + 1bp)  −  V(rate)          ← forward diff
                   =  [V(rate + 1bp) − V(rate − 1bp)] / 2  ← central diff (more accurate)


   Convexity_FD    =  [V(rate + h)  −  2·V(rate)  +  V(rate − h)]  /  (h² · V(rate))
```

### Greeks (for option-bearing products)

Generic non-rate sensitivities computed the same way (analytical for vanilla, FD for exotics):

| Greek | What | Formula (Black on forward `F`) | Units |
|---|---|---|---|
| Δ delta | dV / dS | `DF(T) · N(d1)` (call) | $/spot unit |
| Γ gamma | d²V / dS² | `DF(T) · φ(d1) / (F · σ · √T)` | $/spot² (note: denominator uses `F`, not `S`) |
| ν vega | dV / dσ | `DF(T) · F · φ(d1) · √T` | $/vol point (e.g. /1% vol) |
| Θ theta | dV / dt | (long expressions; numerical bump is easier) | $/day |
| ρ rho | dV / dr | call-rho > 0 | $/rate unit |

> **Convention**: Gamma in the Black model uses the **forward `F`** in the denominator, not the spot `S`. The two differ by `F/S = 1/DF(T)` for non-dividend assets — small for short tenors, material at 5y+. Many textbooks abuse notation by writing `S` where `F` belongs; verify which model you're using.

---

# Part IX — Extended Treasury Universe (abbreviated entries)

Real treasury desks trade ~50+ products. The 18 above cover all major shapes; this section adds 15 more **variations on those shapes** — anything not here is a recombination of these primitives.

> **Format note**: entries below are **abbreviated** (what / formula / use / risk factors). They share the same patterns as the worked products in Parts I–VII. For an example of the full 6-block template, see any of §1–§18. Each entry is prefixed `E1, E2, …` (NOT continuing §19) so it doesn't collide with the Risk Metrics section.

## E1. T-Bill / Discount paper (short-term zero-coupon govt)

**What**: government short-term debt issued at a DISCOUNT (no coupons). Mature in 4w, 13w, 26w, 52w. Quoted on a discount-yield basis: `dy = (face − price)/face · 360/days`.

```
Value today  =  face · DF(T)                   (zero-coupon bond, no spread for risk-free)
```

**Use**: cash management, collateral, risk-free benchmark. **Risk factors**: rate only (small DV01).

## E2. Commercial Paper (CP) / Banker's Acceptance

**What**: short-term corporate (CP) or bank (BA) zero-coupon debt. Same math as T-bill but with credit spread.

```
Value today  =  face · DF(T, spread_bp)
```

**Use**: corporate funding (CP), trade finance (BA). **Risk factors**: rate, credit spread (CS01).

## E3. Zero-Coupon Bond (long-dated)

**What**: bond with NO coupons, only the face at maturity `T`. Pure interest-rate play.

```
Value today  =  face · DF(T, spread_bp)
Duration     =  T   (Macaulay duration equals the maturity exactly — no PV-weighting needed)
```

**Use**: long-duration hedging (pension liabilities), TIPS strips. **Risk factors**: rate (very large DV01 because all weight at `T`), credit spread.

## E4. Floating-Rate Note (FRN)

**What**: bond paying coupons indexed to a floating rate (LIBOR/SOFR + spread) every period.

```
At each t_i:  pays  (L_i + spread) · τ · face
At maturity:  pays  face

Value today (just after a fixing)  =  face · spread · Σ τ · DF(t_i)  +  face
                                      spread × annuity            ← floating part is par
```

> **Caveat**: the formula above assumes you're valuing AT a fixing date. Between fixings, the just-set coupon `L_last + spread` is already known and accrues to the NEXT payment — the floater PV's to `(L_last + spread)·τ·face·DF(t_next) + face·DF(t_next)` for the residual until the next reset, NOT exactly to par. The bias is small (a few bp) over short reset windows.

**Use**: low-duration credit exposure — duration ≈ time-to-next-fixing (often < 0.25 years). Same idea as IRS floating leg, but you hold the notional.

**Risk factors**: spread (dominant), tiny DV01 (just to next fixing).

## E5. Callable Bond

**What**: bond + embedded short call option (issuer can buy back early at par). Issuer benefits if rates fall (refinance cheaper) — bondholder loses.

```
Value  =  Value(straight bond)  −  Value(call option on the bond)
```

Negative convexity — bondholder bears asymmetric risk. **Use**: corporate issuers (callable munis, sub debt). **Risk factors**: rate, credit, **vol** (the option's vega), call schedule.

## E6. Convertible Bond

**What**: bond + embedded long call option on the issuer's equity (holder can convert into shares).

```
Value  =  Value(straight bond)  +  Value(conversion option)
```

Hybrid risk: bond-like when far-from-conversion, equity-like when in-the-money. **Use**: tech / growth issuers. **Risk factors**: rate, credit, equity, equity vol — multi-factor.

## E7. Overnight Index Swap (OIS)

**What**: IRS where the floating leg is the COMPOUNDED overnight rate (Fed Funds, SOFR, €STR, SONIA) instead of LIBOR/term-rate. The benchmark "risk-free" swap.

```
Same formula as IRS (§10), but floating leg references the overnight curve.
Post-2008: OIS curve is the DISCOUNTING curve in multi-curve frameworks.
```

**Use**: hedge against central-bank rate moves, discount swap collateral, FX swap pricing. **Risk factors**: rate (OIS DV01).

## E8. Constant-Maturity Swap (CMS)

**What**: IRS where the floating leg references a SWAP RATE of fixed tenor (e.g. 10y swap rate, observed every reset) instead of a short rate. Used to bet on curve shape.

```
PV requires a CONVEXITY ADJUSTMENT — the swap rate isn't a martingale under the
forward measure → can't just discount expected forward swap rate.
Practical formula uses replication via swaptions (Hagan).
```

**Use**: curve steepening/flattening bets, hedging principal-protected notes. **Risk factors**: rate (multiple points on curve), vol (the convexity adjustment depends on the **entire swaption SABR vol surface**, not just one σ — change in skew matters).

## E9. Asset Swap (ASW)

**What**: a PACKAGE — buy a bond and simultaneously enter an IRS that swaps the bond's fixed coupons for floating. Net result: synthetic floater earning `LIBOR + ASW_spread`.

```
ASW spread  =  fixed coupon  −  par swap rate (same tenor)        (approx)


PV(package)  =  PV(bond)  +  PV(IRS at bond's coupon)  =  par by construction
```

**Use**: isolate the bond's credit-spread component from rate exposure. **Risk factors**: credit spread of bond, almost zero rate exposure post-swap.

## E10. Year-on-Year (YoY) Inflation Swap

**What**: like ZCIS but pays YOY inflation each year, not a single compounded payment at maturity.

```
At each t_i:  receive  N · (π_{i} − K) · τ        (annual inflation π_i)
```

**Use**: hedge year-by-year inflation exposure (vs lump-sum ZCIS for liability matching). **Risk factors**: breakeven curve, **inflation vol**, **inflation/rate correlation** (the YoY-vs-ZCIS convexity adjustment is `ρ·σ_π·σ_r·t`-flavoured — material at long tenors).

## E11. Cap / Floor / Caplet

**What**: portfolio of interest-rate options. Cap = strip of caplets (each pays when the floating rate exceeds K); floor = puts; collar = cap + sold floor.

```
Caplet payoff at t_i:   τ · N · max(L_i − K, 0)              paid at t_{i+1}
Cap value           =   Σ Caplet_i
Caplet pricing      =   Black formula on the FORWARD rate L_i^fwd
```

**Use**: hedge floating-rate borrowing (cap), hedge a bond portfolio against falling rates (floor). **Risk factors**: rate (delta), vol (vega) — caps are pure vol plays.

## E12. Bermudan Swaption

**What**: swaption with MULTIPLE possible exercise dates (vs European = one). Holder can choose the optimal exercise.

```
Pricing requires LSMC (Longstaff-Schwartz) or lattice — no closed form
Value > European swaption by the "early exercise premium"
```

**Use**: callable bond hedging (the embedded call is a Bermudan), structured products. **Risk factors**: rate, vol, **mean reversion** (which short-rate model you use matters).

## E13. CDS Index (CDX / iTraxx)

**What**: standardised CDS contract on a basket of 100+ reference names (CDX NA IG = 125 names, US investment grade; iTraxx Europe = 125 names, European IG, etc.). Pre-defined coupons (100 bp for IG, 500 bp for HY).

```
Same math as single-name CDS (§15), but each defaulting name pays out a
SLICE of the notional: payout per name = (1/n) · (1−R) · N.
Remaining notional CONTINUES TO ACCRUE PREMIUM after each default.
Index spread quoted in bp (e.g. CDX IG = 65 bp).
```

(Index ≠ first-to-default basket. FTD pays full notional on the FIRST default and terminates; the index pays incrementally and continues.)

**Use**: macro credit hedge, relative value vs single-name CDS, P&L from spread tightening. **Risk factors**: credit spread (CS01 of the whole index, often called "spread DV01"), correlation (for tranches), rate (small).

## E14. Mortgage-Backed Security (MBS) — agency pass-through

**What**: bond backed by a pool of mortgages. Cashflows = scheduled principal + interest + UNCERTAIN prepayments (homeowners refinance when rates fall).

```
PV  =  Σ E[CF_t | prepay model] · DF(t, spread)
                    requires Monte Carlo over interest rate paths
                    with a prepayment model (CPR/PSA)
```

Negative convexity — when rates fall, prepayments accelerate, you receive your money back at par EXACTLY when reinvestment yields drop.

**Use**: high-yield rate exposure with prepay risk (asset-mgr darling). **Risk factors**:
- **rate** (effective DV01 — substantially lower than legal-maturity duration due to negative convexity)
- **prepay risk** (CPR/PSA — separate scenario stress)
- **OAS** (the spread that makes model PV = market PV after stripping the prepay option — NOT the same as vol)
- **vol** (the rate-vol input INTO the prepay model — separately bumped to compute vega)

## E15. Variance Swap

**What**: agree to exchange realised variance over `[0, T]` for a fixed strike `K²`. Pure exposure to realised vol.

```
Payoff at T  =  N_var · ( σ²_realised  −  K² )       per VARIANCE notional N_var

Where:
    N_var       = variance notional ($ per unit of variance)
    N_vega      = vega notional ($ per vol point) = 2K · N_var

For small moves:   payoff ≈ N_vega · (σ_realised − K)        ← intuitive "$ per vol point"
```

Replication = a STATIC portfolio of out-of-the-money calls + puts (weighted as `1/K²`) plus a continuous re-hedge of a delta-one position.

**Use**: vol arbitrage (realised vs implied), tail hedge (long variance = long crashes — non-linear payoff blows up under stress). **Risk factors**: vol (linear in variance, vs vanilla options which are concave in vol), correlation skew, jumps (gap risk).

## E16. Equity Total Return Swap (Equity TRS)

**What**: same structure as bond TRS (§14) but reference is an equity (single stock or index). Receiver gets price changes + dividends; payer gets financing rate.

```
V  =  PV(equity total return)  −  funding_notional
```

**Use**: synthetic equity exposure (no balance sheet), tax-efficient long stocks. **Risk factors**: equity delta, equity vol (small), rate (financing leg), dividends.

## E17. Commodity Forward / Swap

**What**: commodity equivalents of FX forward / IRS. Commodity forward locks in a future price; commodity swap pays floating index vs fixed.

```
Same math skeletons:
  Forward:  V = N · (F_T − K) · DF(T)        where F_T = forward commodity price
  Swap:     fixed leg vs floating commodity index, like an IRS but with comm index
```

**Use**: airlines hedge jet fuel (oil swap), corporates hedge raw materials. **Risk factors**: commodity price, basis (deliverable vs benchmark grade), seasonality, storage / convenience yield.

---

# Universal Patterns to Memorise

## Pattern A: Linear-in-strike

Applies to: **FX Forward (§5), FRA (§6), ZCIS (§11), IRS (§10 ≈ same), CDS (§15)**.

```
V  =  (market_implied  −  contract)  ·  scale  ·  discount


Par strike   =  market_implied   (solves V = 0)
```

| Product | market-implied | scale | discount | par strike |
|---|---|---|---|---|
| FX Forward | `S · fx_factor` | `N_for` | `DF(T)` | `S · DF_for/DF_dom` |
| FRA | `fwd = (D_1/D_2 − 1)/τ` | `N · τ` | `DF(T_2)` | `fwd` |
| ZCIS | `(1+b)^T` | `N` | `DF(T)` | `b` |
| IRS | `c_par = (1−D(T))/A` | `N · A` | bundled | `(1−D(T))/A` |
| CDS | market_spread | `N` (premium/protection) | `Q(t)·DF(t)` | market_spread |

## Pattern B: basis × annuity

Applies to: **Sec Lending (§3), Tenor Basis Swap (§12), XCCY Basis Swap (§13)**.

```
V  =  basis  ·  notional  ·  annuity_factor  [·  FX conversion if foreign]


annuity_factor  =  Σ τ · DF(t)
```

Pure stream of small payments. Drill formula uses `b = bp/10000`.

## Pattern C: Cashflow strip PV

Applies to: **Bond (§7), Foreign Bond (§8), Linker (§9), Loan (§4), Annuity Liability (§18)**.

```
V  =  ±  Σ  cf_i  ·  uplift_i  ·  DF(t_i, spread)


uplift_i  =  (1 + b)^t_i  for inflation-linked
          =  1            for nominal
spread    =  issuer spread (0 for govt bonds, > 0 for corp/credit-risky)
sign      =  + for assets, − for liabilities
```

---

# Risk Factor Quick Reference

Each "01" metric is a one-bp bump-and-revalue (rate factors) or scaled bump (FX, vol).

| Factor | Affects | Risk metric | Bump-and-revalue |
|---|---|---|---|
| **rate** (bp) | every `DF(t)` | DV01 | bump nominal curve +1 bp, reprice |
| **breakeven** (bp) | inflation-linked products (Linker, ZCIS, Annuity) | IE01 | bump `b` +1 bp |
| **credit spread** (bp) | Bond, Loan, Foreign Bond, CDS | CS01 | bump `spread_bp` +1 |
| **FX** (ret, e.g. 1%) | FXSpot, FXForward, ForeignBond, XCCYBasisSwap, FXOption | FX delta | scale FX legs by `(1 + 0.01)` |
| **tenor basis** (bp) | Tenor Basis Swap | tenor-01 | bump that basis +1 bp |
| **XCCY basis** (bp) | XCCY Basis Swap | xccy-01 | bump that basis +1 bp |
| **vol** (vol point) | FX Option, Swaption | vega | bump σ +1% |

Per-instrument sign cheatsheet:

| Long position in… | DV01 sign | IE01 sign | FX delta sign |
|---|---|---|---|
| Bond / Loan / Linker | − (rates ↑ → P ↓) | + for Linker only | 0 (unless foreign) |
| IRS — receive-fixed | + (rates ↑ → V ↑) | 0 | 0 |
| IRS — pay-fixed | − | 0 | 0 |
| ZCIS — receive inflation | + (rates ↑ → V ↑ via DF) | + | 0 |
| FX Spot / Forward (long foreign) | + (via DF) | 0 | + |
| **Annuity Liability** | **+ (V more negative when rates ↑)** | **−** | 0 |
| Long-call FX/Swaption | small | 0 | + (call) / − (put) |

Annuity liability's signs are the OPPOSITE of a long bond's because the cashflows themselves are negative.

---

# Convention Reminders (where people slip up)

1. **bp → decimal** is always `/ 1e4`. (15 bp = 0.0015.)
2. **Inflation uses `(1+b)^t` compounding**, not `1 + b·t` simple — inflation indices compound multiplicatively.
3. **Year fraction τ** must match the contract's day-count (ACT/360, 30/360, ACT/ACT...) — drills use τ = 1.0 annual but real desks rarely have `τ = 1.0` exactly.
4. **Annuity liability NEGATIVE sign** — you OWE these. Forgetting reverses the entire ALM.
5. **Linear-in-strike par rate**: set `V = 0`, divide out common factors → strike = market-implied. Works for FX Forward, FRA, ZCIS, IRS, CDS.
6. **FX-sensitive products** (FX Spot, FX Forward, ForeignBond, XCCY, FX Option) need `S · fx_factor()`; purely domestic products don't.
7. **Forward in FRA** is the SIMPLE-compounded one: `(D_1/D_2 − 1)/τ` — used because FRA payoff `(L−K)·τ` is itself simple. Continuous-compounded forward `−ln(D_2/D_1)/(T_2−T_1)` is for HJM / academic models.
8. **CDS hazard `λ` ≠ spread `s`**: relationship is `s ≈ λ · (1−R)`. Use `λ = s/(1−R)` to build the survival probability `Q(t) = exp(−λt)`.
9. **Swap floating leg telescope**: `Σ payments` simplifies to `1 − DF(T)` in single-curve world. In multi-curve world (LIBOR/OIS dual curve) you need both projection and discount curves — drill ignores this.
10. **Bond spread** in this drill is **continuous-comp** (`exp(−(z+s/1e4)·t)`); production conventions also include Z-spread, OAS, ASW spread — see §7 caveats.
11. **ZCIS quote convention**: `K` quoted as annualised CAGR (matches `(1+K)^T`). Some markets quote on continuous-comp basis — verify before trading.
12. **Option vol** is per the BLACK model — lognormal in `F` (forward FX or forward swap rate), NOT in spot. Strike comparisons should always be vs the FORWARD, not the spot.

---

# Appendix — Real-world ALM Caveats — what the drills gloss over

These are the things experienced desks know that the toy formulas hide. Worth a quick read when refreshing.

### Multi-curve discounting (post-2008)

Drills use a single curve for discounting AND projection. Real desks:

- **Discount** at the **OIS** (or SOFR/€STR) curve — the "risk-free" funding rate
- **Project** floating rates on the LIBOR/SOFR-fixing curve (often different from OIS)
- Difference can be 30–50 bp under stress (e.g. 2008 LIBOR-OIS blowout)
- A real IRS therefore has two curves; the drill's single-curve `1 − DF(T)` collapses this.

### FRA settlement timing

Drill values FRA via the curve-implied forward, discounted at `DF(T_2)`. In real settlement:

- Cash payment occurs at `T_1` (start of period), NOT `T_2`
- Payment = `N · (L − K) · τ / (1 + L · τ)` — discounted at the OBSERVED `L`, not the curve
- The two are equivalent under the `T_2`-forward measure (martingale property)
- Practical importance: settlement discount matters for collateral calls

### FRA vs Eurodollar/SOFR futures — convexity adjustment

A FRA and a STIR future on the same window are *almost* the same, but futures' daily mark-to-market creates a convexity bias of `½σ²T₁T₂`. For 5y+ futures the adjustment is 5–20 bp — material when bootstrapping from futures vs FRAs.

### XCCY swaps — principal exchange and MtM resets

Real XCCY swaps have **principal exchange at start and end** (omitted in drill). The "Mark-to-Market XCCY" variant **resets** the notional on every payment date based on the new spot — this dampens the FX P&L between fixings. Drill formula is the simplest possible version.

### Linker — indexation lag

The CPI used at payment time is typically the index level from 3M earlier (UK index-linked gilts) or 8M earlier (US TIPS). Creates a small basis between contractual cashflows and the breakeven curve. Real IE01 has a lag-adjusted component.

### Breakeven ≠ pure inflation expectation

Market breakeven `b` ≈ expected inflation + **inflation risk premium** + **liquidity premium** (Linkers are less liquid than nominals). For valuation it's `b`; for forecasting actual inflation you should strip out the premia (typically 30–80 bp).

### LDI / pension / insurer regulatory regimes

- **UK FRS102 / IFRS / IAS 19 / US ASC 715**: liabilities discounted on **AA-corporate curve**, not OIS
- → Liability DV01 vs swap DV01 differ by the **swap-vs-AA basis** (a real, persistent P&L source)
- LDI hedges nominal rate + inflation, but the AA-corporate basis is an UNHEDGED residual
- **Solvency II (EU insurers)**: matching adjustment + volatility adjustment let you add a credit-spread bonus when matching cashflows; **SCR Interest Rate Risk** = parallel ±64 bp scenarios on the risk-free curve (per EIOPA tables)
- **Basel III IRRBB (banks)**: EVE and NII sensitivities under 6 prescribed scenarios (parallel ±, short-rate ±, steepener, flattener); HQLA classification affects which bonds count for LCR
- **Accounting**: AFS bonds → P&L volatility goes through OCI (insulates earnings); AC ("hold-to-maturity") → no MTM volatility but limits portfolio flexibility

### Pension de-risking glide path

Most modern DB pension schemes operate a **funded-ratio-triggered glide path**: as the funded ratio (assets/liabilities) crosses thresholds, the hedge ratio steps up. Example: at 90% funded, hedge 60% of liability DV01; at 95% funded, step to 75%; at 100% funded, step to 95%. Mechanically forces de-risking when markets cooperate, leaves upside when they don't.

### LPI caps and floors (UK pensions)

UK pensions often pay LPI (limited-price-indexation): inflation between 0% and 5%, capped at 5%. This makes the liability **non-linear in inflation** — ZCIS doesn't hedge the cap/floor convexity. Real LDI uses inflation caps + floors as well.

### September 2022 gilt crisis — the liquidity risk of LDI itself

UK pension LDI hedges (long-duration rate swaps) required posting collateral when rates rose sharply. Forced selling of the assets being "hedged" cascaded into a gilt crisis. **Hedging strategies have their own liquidity profile** — match `D_mod` AND collateral capacity.

### Multi-tenor basis swaps

Real desks have `3M-vs-6M`, `1M-vs-3M`, `1M-vs-6M` basis swaps — each with its own quoted spread. The drill's "tenor basis swap" is the 3M-vs-6M flavour; production curves have a SURFACE of basis spreads.

### TRS — who's exposed to what

The TRS receiver gets the bond's economic exposure (price + coupon) WITHOUT funding it on balance sheet. The TRS payer (typically a bank) provides synthetic financing and earns the funding spread. Common use: hedge funds use TRS to get bond exposure without using their balance sheet; banks use TRS to lay off counterparty exposure.

### CDS — hazard rate vs market spread

Real CDS curves have a TERM STRUCTURE of spreads (1y, 3y, 5y, 10y), not one flat number. Bootstrap a hazard-rate curve from market spreads exactly like a discount curve. Drill assumes flat λ → 5y CDS only.

### Nominal IRS + nominal bond not in the drill — but always paired with these

For a full hedge book you'd expect:
- **Vanilla IRS** for nominal rate hedging (most-traded rates product on earth)
- **Government bond** for risk-free cashflow projection
- The drill's IRS (§10) is exactly this — you have it. Just flagging that real ALM uses HEAPS of these alongside the others.

### Real vs nominal rate decomposition (Fisher)

For inflation-linked products: `nominal_rate ≈ real_rate + breakeven`. A 1 bp move in NOMINAL rates isn't the same as a 1 bp move in REAL rates — for a Linker, real-rate DV01 is what matters; for an annuity liability discounted on nominal, nominal DV01 matters. Don't conflate them.

---

That's the complete reference. The 6-block template per product means you can refresh ANY of these in 90 seconds: scan the formula, plug into the toy example, check the intuition, look up the closer.
