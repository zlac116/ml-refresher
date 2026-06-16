# Market Quoting Conventions — Bloomberg/Reuters → Model Inputs

A one-page lookup for translating screen quotes into the inputs your exercise
functions expect. For each instrument: how the market quotes it, the
conversion formula, and the exercise that consumes it.

---

## 1. US Treasury bonds + notes

| Aspect | Convention |
|---|---|
| **Price quote** | `XX-YY[+]` in 32nds (+ = extra 1/64) |
| **Yield** | Bond-Equivalent Yield (BEY), semi-annually compounded |
| **Day-count** | ACT/ACT |
| **Settlement** | T+1 |
| **Bloomberg ticker** | e.g. `US10Y`, `US30Y`, `T 4.25 11/15/40` |

**Decode the price format:**

```
"102-16+"   →   102 + 16/32 + 1/64   =  102.515625
"98-08"     →   98  +  8/32          =   98.250000
"100-00"    →  100                   =  100.000000
```

### 📊 Worked example — Quote to fair value

**The quote (Bloomberg DES page):**

```
T 4.25 11/15/40         PX_MID   102-16+  (102.515625)
                        YLD_MID   4.019%
                        DUR_MOD   12.7
```

**Two formulas — one for the quote, one for your model:**

```
Bloomberg's price (YTM, single rate):

                 n
  P(y)  =  sum  c_i  /  (1 + y/m)^i      with y = 4.019%, m = 2, c_n includes face
                 i=1


Your model's price (curve, per-cashflow):

                 n
  P_model = sum  c_i  ×  D(0, t_i)       with D(0, t_i) = exp(-z(t_i) × t_i)
                 i=1                     from YOUR bootstrap (exercises 08, 09)
```

**Numerical check — Bloomberg's yield reprices its own quote:**

```python
>>> ytm_from_dirty(102.515625, face=100, coupon=0.0425, T=14.5, freq=2)
0.04019        ← matches Bloomberg's 4.019% ✓
```

**Your fair value off your curve** (illustrative — substitute your bootstrap):

```python
times = np.arange(1, 30) * 0.5           # 29 semi-annual dates
cf    = np.full(29, 4.25/2); cf[-1] += 100
zeros = your_curve(times)                 # from exercise 08 / 09
D     = np.exp(-zeros * times)
P_model = (cf * D).sum()                  # → 102.40
```

**Gap:**

```
Bloomberg mid     =  102.515625
P_model           =  102.40
gap / $100 face   =   0.1156
DV01 (D_mod×P×1e-4) = 0.1302 / bp
gap in bp         =   0.1156 / 0.1302   ≈   9 bp
on $10M face      =   $11,560
```

**Verdict:** bond is ~9 bp rich on the screen vs your curve.

**Feed to your code:** `01_bond_pricing_ytm.py` (YTM round-trip) + `03_duration_convexity.py` (DV01) + your bootstrap from `08`/`09`.

---

## 2. US T-bills (zero-coupon, < 1 year)

| Aspect | Convention |
|---|---|
| **Price quote** | `% of face` (e.g. 98.50) |
| **Yield quote 1** | **Discount yield** (360 basis, banker's convention) |
| **Yield quote 2** | **Investment yield / BEY** (365 basis, comparable to coupon bonds) |
| **Day-count** | ACT/360 (discount), ACT/365 (BEY) |
| **Settlement** | T+1 |

**The two yields differ — Bloomberg shows BOTH:**

```
discount yield   =  (F - P) / F  *  360/days       ← banker's convention
investment yield =  (F - P) / P  *  365/days       ← BEY, comparable to bonds
```

**Why BEY is higher**: investment yield uses *price* (not face) in the denominator and 365 (not 360) in the numerator. Both effects push it up.

### 📊 Worked example — Quote to fair value

**The quote (Bloomberg DES page):**

```
3M T-bill (91 days)     PX_LAST   98.736
                        DISC_YLD   5.00%
                        BEY        5.13%
```

**Two formulas — one for the quote, one for your model:**

```
Bloomberg's price (discount yield, ACT/360):

  P  =  face × (1 - dy × days/360)        with dy = 5%, days = 91

  BEY (investment yield, ACT/365):

  BEY  =  (face - P) / P × 365 / days


Your model's price (curve):

  P_model  =  face × D(0, T)              with D(0, T) = exp(-z(T) × T)
                                          from YOUR bootstrap
```

**Numerical check — round-trip:**

```
dy=5%, 91d   →   P = 100 × (1 − 0.05 × 91/360) = 98.7361   ✓
P=98.7361    →   BEY = (1.2639/98.7361) × 365/91 = 5.13%   ✓
```

**Your fair value off your curve** (illustrative):

```python
z_91d = your_curve(91/365)            # e.g. 5.10%
D     = np.exp(-z_91d * 91/365)
P_model = 100 * D                       # → 98.7365
```

**Gap:**

```
Bloomberg price   =  98.7361
P_model           =  98.7365
gap / $100 face   =  -0.0004        →  effectively zero
on $10M           =  ~$40
in yield bp       =  ~0.2 bp
```

**Feed to your code:** `15_market_quote_parsing.py` (tasks 2-3).

---

## 3. SOFR / Eurodollar futures (STIRs)

| Aspect | Convention |
|---|---|
| **Price quote** | `100 - rate` |
| **Implied rate** | `100 - quoted_price` |
| **Contract** | 3M deposit equivalent, $1M notional |
| **Tick value** | `$25 per bp` per contract (SOFR/ED) |
| **Day-count** | ACT/360 |
| **Bloomberg ticker** | `SR3` (SOFR), `ED` (legacy Eurodollar) |

**Decode:**

```
quoted_price = 95.50    →   implied 3M rate = 100 - 95.50 = 4.50%
quoted_price = 99.00    →   implied 3M rate = 1.00%

P&L on 100 contracts × +5 bp move:
   100 contracts × 5 bp × $25/bp/contract = $12,500
```

### 📊 Worked example — Quote to fair value

**The quote (Bloomberg page SR3):**

```
SR3 Z6 (Dec 2026 SOFR future)    PX_LAST   95.55
                                  Implied   4.45%
                                  Tick      $25/bp
```

**Two formulas:**

```
Bloomberg's implied rate:

  rate_implied  =  (100 - quoted_price) / 100      = 4.45%


Your model's fair rate (forward from your curve):

  D(0, T_1)             3M discount factor (start of period)
  D(0, T_2)             6M discount factor (end of period)
  δ                     = (T_2 - T_1) in years  (≈ 0.25)

  f_fair  =  (1/δ) × (D(0,T_1) / D(0,T_2) - 1)      ← simple-comp forward
```

**Numerical check (futures price ↔ rate):**

```
quoted 95.55  →  implied  =  (100 − 95.55)/100  =  4.45%   ✓
```

**Your fair rate** (illustrative — your curve gives `f_dec26 = 4.40%`):

```python
D_t1, D_t2 = your_curve(T1=0.25), your_curve(T2=0.50)
f_fair = (D_t1/D_t2 - 1) / (T2-T1)         # → 4.40%
P_model = 100 - f_fair*100                  # → 95.60
```

**Gap:**

```
Quoted price   =  95.55     (rate 4.45%)
P_model        =  95.60     (rate 4.40%)
gap in rate    =   5 bp     (screen is rich in price → low in rate)

P&L on 100 contracts if rate converges to 4.40%:
  100 × 5 bp × $25/bp  =  $12,500
```

**⚠ Convexity caveat**: futures rate differs from the FRA rate by `½σ²T₁T₂`. For 6M out the bias is < 1 bp; for 5y+ futures it's 5–20 bp. **Subtract the bias before comparing**, or you'll mistake the convexity adjustment for a mispricing.

**Feed to your code:** `15_market_quote_parsing.py` (tasks 4-5).

---

## 4. Corporate / agency bonds

| Aspect | Convention |
|---|---|
| **Price quote** | `% of par` (e.g. 99.25) |
| **Yield** | BEY semi-annual (US), annual (most Eurobonds) |
| **Day-count** | 30/360 US corps; ACT/ACT EUR; varies |
| **Spread quotes** | Z-spread, G-spread, ASW spread (see §10) |
| **Settlement** | T+2 (US) or T+3 (some EUR) |

### 📊 Worked example — Quote to fair value

**The quote (Bloomberg DES page):**

```
XYZ 6.000 06/15/2031    PX_MID    98.50 clean
                        ZSPRD    +120 bp
                        GSPRD     +98 bp
                        DUR_MOD    4.19
```

**Two formulas:**

```
Bloomberg's implied price (Z-spread on curve):

  P(s)  =  sum  c_i × exp(-(z(t_i) + s) × t_i)     with s = quoted Z-spread


Your model's Z-spread (back out s from market price):

  Find s such that:    sum  c_i × exp(-(z(t_i) + s) × t_i)  =  market_price
                       (solve via brentq)
```

**Numerical check** (5y, 6% annual coupon, market 98.50, your zero curve from exercises 08/09):

```python
>>> z_spread(market_price=98.50, face=100, coupon_rate=0.06, T=5, zeros=..., tenors=...)
0.00928     ← 92.79 bp on YOUR curve
```

**Gap (dealer's Z=120 bp vs yours = 92.79 bp):**

```
dealer's Z-spread   =  120 bp
your Z-spread       =   92.79 bp
gap in spread       =   27 bp           ← bond is RICH vs your curve

Price impact:    spread duration × spread gap
                  ≈ Mod D × 27 bp
                  =  4.19 × 27/10000
                  =  1.131%   of price

In $/100 face:    1.131% × 98  =  $1.108  per $100
On $5M face:      $5M / 100 × $1.108  =  $55,400
```

**Verdict:** bond is 27 bp / $55k rich on the dealer's screen vs your curve.

**Feed to your code:** `01_bond_pricing_ytm.py` (YTM back-out) → `16_relative_value_spreads.py` (Z, G, ASW spreads).

---

## 5. Sovereign bonds — non-US conventions

| Country | Yield convention | Day-count |
|---|---|---|
| **US Treasury** | BEY semi-annual | ACT/ACT (US Treas) |
| **UK Gilt** | semi-annual | ACT/ACT (ICMA) |
| **German Bund** | **annual** | ACT/ACT |
| **JGB** | semi-annual | ACT/365 |
| **Canadian** | semi-annual | ACT/ACT |
| **Australian** | semi-annual | ACT/ACT |

**Critical**: a "5% Bund yield" pays once per year; a "5% UST yield" pays twice per year as 2.5% semi-annual. Their EARs differ.

```
5% UST (semi):   EAR = (1 + 0.05/2)^2 - 1  =  5.0625%
5% Bund (ann):   EAR = 5.0000%
```

---

## 6. Money-market deposits

| Currency | Day-count | Compounding |
|---|---|---|
| **USD, EUR, CHF, JPY** | ACT/360 | Simple |
| **GBP, AUD, CAD** | ACT/365 | Simple |

Quote format: `r = 5.00%` annualised simple.

### 📊 Worked example — Quote to fair value

**The quote (broker / Bloomberg):**

```
3M USD deposit         5.00%    91 days    notional $1M
```

**Two formulas:**

```
Simple interest (USD, ACT/360):
  interest  =  N × r × days/360
  future value =  N × (1 + r × days/360)

EAR (effective annual rate, for cross-currency comparison):
  EAR  =  (1 + r × days/360)^(360/days)  -  1
```

**Numerical check:**

```
interest_USD  =  1M × 0.05 × 91/360  =  $12,638.89
future value  =  $1,012,638.89
EAR           =  (1.012639)^(3.956) - 1  =  5.094%       ← higher than 5%
```

**Compare conventions** (same 5% headline, different basis):

```
USD (ACT/360):  interest = 1M × 0.05 × 91/360 = $12,638.89
GBP (ACT/365):  interest = 1M × 0.05 × 91/365 = $12,465.75
gap                                            =    $173.14   (USD pays MORE)
```

**Feed to your code:** `05_deposit_fra.py` task 1, or `02_day_count_conventions.py` to compare bases.

---

## 7. Forward Rate Agreements (FRAs)

| Aspect | Convention |
|---|---|
| **Quote name** | `M_start × M_end` (e.g. "3×6", "6×9") |
| **Quoted rate** | Simple-comp ACT/360 forward |
| **Settlement** | Discounted at fixing — see `07_deposit_fra.md` |
| **Bloomberg ticker** | e.g. `USFR0CF Curncy` (3×6 SOFR FRA) |

**Decode the quote name:**

```
"3×6 at 5.04%"     →   F = 0.0504, T_start = 3M, T_end = 6M
"6×9 at 5.20%"     →   F = 0.0520, T_start = 6M, T_end = 9M
```

Note: `M_start × M_end` is in **months from spot**, not years.

### 📊 Worked example — Quote to fair value

**The quote (Bloomberg / broker):**

```
3×6 SOFR FRA           F      5.20%
                       settle T+2
                       period 91 days (3M deposit forward-starting in 3M)
```

**Two formulas:**

```
Bloomberg's quoted rate F  is just the contracted forward rate.


Your model's fair forward rate (from your curve):

  D(0, T_1)        3M discount factor (start of forward window)
  D(0, T_2)        6M discount factor (end of forward window)
  δ                = (T_2 - T_1) in years  (= 91/360 ACT/360)

  F_fair  =  (1/δ) × (D(0,T_1) / D(0,T_2) - 1)
```

**Numerical check** (your curve gives `D_3M=0.98875`, `D_6M=0.97667`):

```python
delta = 91/360
F_fair = (D_3M/D_6M - 1) / delta
       = (0.98875/0.97667 - 1) / 0.252778
       = 0.04893               # → 4.893%
```

**Gap:**

```
Quoted F        =  5.200%
F_fair          =  4.893%
gap             =   +31 bp     (screen is rich → SELL the FRA)
```

**Settlement payoff if market converges to 4.89%** (you sold the FRA at 5.20%):

```
payoff at T_1  =  N × (F - L) × δ / (1 + L·δ)        ← you receive (F − L)
                =  10M × (0.052 - 0.04893) × 0.2528 / (1 + 0.04893×0.2528)
                ≈  $7,660
```

**Feed to your code:** `05_deposit_fra.py` (fair rate) + `06_fra_settlement.py` (cash payment).

---

## 8. Interest Rate Swaps

| Aspect | Convention |
|---|---|
| **Quote** | Par swap rate (% of fixed leg) |
| **USD SOFR swap** | Fixed annual ACT/360, floating quarterly compounded SOFR |
| **EUR ESTR swap** | Fixed annual ACT/360, floating same |
| **GBP SONIA** | Fixed annual ACT/365, floating compounded |
| **Settlement** | T+2 |
| **Bloomberg ticker** | `USSWAP10`, `EURSW10`, `GBPSW10` |

### 📊 Worked example — Quote to fair value

**The quote (Bloomberg page USSWAP):**

```
5y SOFR swap (USSWAP5)    c_par   4.50%
                          BID     4.498%
                          ASK     4.502%
```

**Two formulas:**

```
Bloomberg's quoted par rate c_par makes a fresh swap have zero NPV.


Your model's par rate (from your curve):

  A(0, T_n)  =  sum  δ_i × D(0, T_i)     ← annuity factor
                i=1..n

  c_fair  =  (1 - D(0, T_n))  /  A(0, T_n)
```

**Numerical check** (your curve bootstrapped to par 4.45%):

```python
A_5y    = (deltas * D_curve).sum()     # → 4.396
c_fair  = (1 - D_curve[-1]) / A_5y      # → 0.0445  (4.45%)
```

**Gap:**

```
Quoted par   =  4.500%
c_fair       =  4.450%
gap          =   +5 bp        (screen is rich → receive-fixed at 4.50% wins)
```

**Mark-to-market at trade** ($100M 5y receive-fixed at 4.50%):

```
PV gain  =  N × (c_screen - c_fair) × A(5y)
         =  100M × 0.0005 × 4.396
         =  $219,800                  ← booked as Day-1 P&L
```

**Feed to your code:** bootstrap (08/09) → annuity (10) → `11_swap_pricing.py` with `fixed_rate=0.045`, `freq=1`.

---

## 9. US fixed-rate mortgage

| Aspect | Convention |
|---|---|
| **Note rate** | Nominal annual, monthly compounded |
| **APR** | Effective annual rate INCLUDING fees (Reg Z required) |
| **Day-count** | 30/360 (most consumer products) |

**The two quoted rates differ:**

```
Note rate         =  6.00%   ← what you compute the monthly payment from
APR               =  6.18%   ← effective annual cost (includes points/fees)
EAR (no fees)     =  (1 + 0.06/12)^12 - 1  =  6.168%
EAR (with 1% pt)  ≈  6.18% (varies with loan size and term)
```

### 📊 Worked example — Quote to fair value

**The quote (Bankrate / lender website):**

```
30y fixed conforming    Note rate    6.625%
                        APR          6.85%
                        Points       0.5
                        Principal    $400,000
```

**Two formulas:**

```
Annuity payment (uses NOTE rate, monthly comp):

  r = note_rate/12       n = years × 12
  PMT  =  P × r × (1 + r)^n / ((1 + r)^n - 1)

EAR (to compare with bond yields):

  EAR  =  (1 + note_rate/12)^12 - 1
```

**Numerical check:**

```python
>>> monthly_payment(P=400_000, annual_rate_nominal=0.06625, years=30)
$2,561.24 / month

total payments  =  2,561.24 × 360  =  $922,048
total interest  =  $522,048                ← > principal!
EAR (no fees)   =  (1.005521)^12 - 1  =  6.83%
APR (with fees) =  6.85%                   ← extra 22.5 bp = the 0.5 points + fees
```

**Compare:**

```
Borrower view:   lenders compete on APR (apples-to-apples after fees)
Trader view:     mortgage spread to 10y Treasury
                  6.625% - 4.02%  =  261 bp     ← premium for prepayment + credit risk
```

**Feed to your code:** note rate → `13_loan_amortisation.py`. APR is borrower-side only.

---

## 10. Bond yield spreads (relative value)

| Spread | Definition | When to use |
|---|---|---|
| **G-spread** | Bond YTM − Treasury par rate at same maturity | Quickest "premium over risk-free" measure |
| **Z-spread** | Constant spread added to the **zero curve** such that PV of bond cashflows = market price | Default desk metric for corporate / agency bonds |
| **I-spread** | Bond YTM − interpolated **swap rate** at same maturity | Relative value vs swap curve |
| **ASW spread** | Spread above floating leg in an asset swap | Used when funding the bond off the swap curve |
| **OAS** (option-adjusted) | Z-spread adjusted for embedded options (callable bonds, MBS) | Negative convexity products |

**Definitions in plain math:**

```
G-spread:   s_G  =  YTM_corp  -  YTM_treasury (same maturity)

Z-spread:   solve s_Z such that:
            sum_i cf_i * exp(-(z(t_i) + s_Z) * t_i)  =  Market_price

I-spread:   s_I  =  YTM_corp  -  swap_rate (same maturity)

ASW:        spread above floating that makes the asset swap NPV = 0
```

**Feed to your code:** `16_relative_value_spreads.py`.

---

## 11. Settlement adjustments

| Instrument | Settle |
|---|---|
| US Treasury | T+1 |
| Corporate bonds | T+2 |
| IRS / FRA | T+2 |
| Equity | T+1 (US, post-2024) |

Bloomberg/Reuters shows the **settle price** — already rolled forward to the settlement date. If you reprice from today's date, you'll be off by 1–2 days of accrued + 1–2 days of discount factor. For tight reconciliation: roll your PV to T+settle before comparing.

---

## 12. Bid / Ask / Mid

```
quote:  102.40 / 102.45   ← bid / ask
mid  =  (bid + ask) / 2   =  102.425
```

Always compare your model fair value to the **mid**, not the bid or ask. The bid-ask is the dealer's profit margin, not the fair value.

---

## Quick reverse-lookup table

| You see on screen | Convert by | Feed to exercise |
|---|---|---|
| `"102-16+"` Treasury price | `whole + 32nds/32 + (+/64)` | `01_bond_pricing_ytm.py` |
| `T-bill 5.93% discount` | `BEY = dy × P/F × 365/360`-style | `15_market_quote_parsing.py` task 2 |
| `SOFR future 95.50` | `rate = 100 - price` | `15` task 3 |
| `3×6 FRA at 5.04%` | identical, ACT/360 | `05_deposit_fra.py` / `06_fra_settlement.py` |
| `5y SOFR swap 5.50%` | `freq=1` annual fixed | `11_swap_pricing.py` |
| `Mortgage note rate 6.00%` | monthly compounding | `13_loan_amortisation.py` |
| `Corp price 98 + Z-spread quote` | use `16` | `16_relative_value_spreads.py` |

---

# Multi-instrument reconciliation — closing the book

Each section above has its own worked example showing **quote → decode → fair value → gap**. This final table is the **integration view**: pull these quotes from Bloomberg/Reuters on the same date, plug each through the relevant exercise, and you've reconciled a small fixed-income book against the screen.

| Instrument | Screen quote | Decoded | Where to feed |
|---|---|---|---|
| 2y UST | `99-24      yield 5.13%` | 99.75, freq=2 | `01_bond_pricing_ytm.py` |
| 10y UST | `102-16+    yield 4.07%` | 102.515625, freq=2 | `01_bond_pricing_ytm.py` |
| 3M T-bill | `98.736     dy 5.00%` | 91d, dy=5%, BEY=5.18% | `15_market_quote_parsing.py` tasks 2-3 |
| SR3 Z6 | `95.55      rate 4.45%` | 4.45% future rate | `15` task 4 |
| 3×6 FRA | `5.20%` | F=5.20%, ACT/360 | `05_deposit_fra.py` / `06_fra_settlement.py` |
| 5y swap | `4.50% annual` | par=4.50% | `08` → `10` → `11` |
| 5y XYZ corp | `98.50 / Z+120` | 92.79 bp Z (my calc) | `16_relative_value_spreads.py` |
| Mortgage | `note 6.625% / APR 6.85%` | monthly comp on note | `13_loan_amortisation.py` |

**Calibration rule of thumb**: if each instrument's gap (screen vs model) is **< 1-2 bp**, your model is calibrated. If any single instrument shows **> 5 bp** gap, dig in — wrong day-count, wrong settle, wrong interpolation, or genuinely mispriced.

---

# References

- **ISDA day-count definitions**: <https://www.isda.org/book/2006-isda-definitions-zip-file/>
- **OpenGamma "Interest Rate Instruments and Market Conventions Guide"**: industry standard.
- **Bloomberg DES `<GO>`** on any security shows the security's day-count, freq, and settle. Always check there if uncertain.
- **Treasury.gov par yield curves**: <https://home.treasury.gov/policy-issues/financing-the-government/interest-rate-statistics> (free, daily).
- **FRED**: `DGS2`, `DGS10`, `DGS30` for benchmark Treasury yields (free, daily).
