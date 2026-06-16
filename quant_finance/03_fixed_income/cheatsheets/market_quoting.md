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

**Screen quote:**
```
T 4.25 11/15/40       102-16+    yield 4.071%    settle T+1
bid 102-16   ask 102-17    mid 102-16+
```

**Decoded:**
```
ticker         :  T 4.25 11/15/40  (4.25% coupon, matures Nov 15, 2040)
price          :  102-16+   →  102 + 16/32 + 1/64  =  102.515625
yield (BEY)    :  4.071%    (semi-annual, ACT/ACT)
freq           :  2         face = 100        remaining T ≈ 14.5 years
```

**Self-consistency check:**
```
Take quoted yield 4.071% and reprice:
  PV = sum_{i=1..29}  (4.25/2) / (1 + 0.04071/2)^i  +  100 / (1 + 0.04071/2)^29
     ≈ 102.51                                       ← matches quoted price ✓

Or take quoted price 102.515625 and back out YTM:
  >>> ytm_from_dirty(102.515625, face=100, coupon=0.0425, T=14.5, freq=2)
  0.04071                                           ← matches quoted yield ✓
```

**Compare to your model:**
```
If YOUR model (with your own discount curve) prices it at 102.40 clean:
  gap                =  102.515625 - 102.40        =  +0.1156 per $100 face
  on $10M position   =  $10M / 100 * 0.1156        =  $11,563
  DV01 of 14.5y bond ≈ 12.7 → in bp terms          ≈ 9.1 bp

Verdict: "the bond is 9 bp rich on the screen vs my curve."
```

**Feed to your code:** parsed decimal price → `01_bond_pricing_ytm.py` (`freq=2`, `face=100`) → DV01 + gap calc via `03_duration_convexity.py`.

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

**Screen quote:**
```
3M T-bill (91 days)   discount 5.00%    investment 5.18%    price 98.736
```

**Decoded — two yields for the same bill:**
```
discount yield     =  5.00%     ← ACT/360 banker's convention
investment yield   =  5.18%     ← ACT/365 BEY, comparable to coupon bonds
price              =  100 × (1 - 0.05 × 91/360)  =  98.7361
```

**Self-consistency check (round-trip):**
```
From discount yield → price:
   price  =  face × (1 - dy × d/360)
          =  100  × (1 - 0.05 × 91/360)
          =  98.7361                     ← matches quoted price ✓

From price → BEY:
   BEY    =  (face - price) / price × 365 / days
          =  (1.2639 / 98.7361) × (365/91)
          =  5.135%                       ← matches quoted investment yield to ~bp ✓
```

**Compare to your model:**
```
If your model curve gives D(91d):  fair price = D(91d) × 100 = 98.74
Quoted price:                                                   98.7361
Gap per $100:                                                    0.0039
On $10M:                                                            $390
In yield bp:                                                       ~1.6 bp
```

**Feed to your code:** `15_market_quote_parsing.py` (tasks 2-3) for the yield conversions.

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

**Screen quote:**
```
SR3 Z6     95.55      (Dec 2026 SOFR future)
```

**Decoded:**
```
implied 3M rate    =  100 - 95.55  =  4.45%
contract month     =  Dec 2026 (Z = December IMM)
notional           =  $1M  per contract
tick value         =  $25 per bp per contract
```

**Compare to your model:**
```
Your bootstrapped curve has a 3M forward rate starting Dec 2026 — call it f_dec26.

If your curve says f_dec26 = 4.40%:
   model fair price  =  100 - 4.40 = 95.60
   quoted price      =  95.55  (rate of 4.45%)
   gap in rate       =  5 bp

Trade idea: buy 100 SOFR Dec26 contracts at 95.55.
   PnL if rate converges to 4.40%:
     100 contracts × 5 bp × $25/bp = $12,500
```

**Caveat — convexity adjustment**: the future's *rate* differs from the matched FRA rate by roughly `½σ²T₁T₂`. For 6M out the bias is < 1 bp; for 5y+ futures it's 5–20 bp. **Adjust the future's rate down before comparing it to your FRA-based curve forward**, otherwise you'll see a fake mispricing.

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

**Screen quote:**
```
XYZ 6.000 06/15/2031     98.50 clean    Z-spread +120 bp    G-spread +98 bp
                          bid 98.45      ask 98.55           mid 98.50
```

**Decoded:**
```
coupon       =  6.00%    annual (illustrative for round numbers)
maturity     =  Jun 15, 2031 — about 5 years out
clean price  =  98.50
Z-spread     =  120 bp above the zero curve
G-spread     =   98 bp above the 5y Treasury par rate
```

**Self-consistency check:**
```
Plug 120 bp Z-spread + your zero curve into:
   price_implied  =  sum_i  cf_i × exp(-(z(t_i) + 0.012) × t_i)
                   =  96.82       (from exercise 16 reference impl)

But dealer's quoted clean price is 98.50.  Conflict!  Three options:
   - the dealer's price is stale,
   - the dealer's curve differs from yours (different OIS bootstrap),
   - there's a wedge between bid/ask/mid.

Path forward:
   Compute YOUR Z-spread from the quoted clean price 98.50:
     >>> z_spread(98.50, face=100, coupon_rate=0.06, T=5, ...)
     0.00928       ← 92.79 bp on YOUR curve
```

**Compare to your model:**
```
3 numbers in play:
   dealer's price       :  98.50
   dealer's Z-spread    :  120 bp
   your Z-spread        :  92.79 bp                ← bond is 27 bp RICH vs your curve

Spread duration ≈ Modified D of the bond ≈ 4.5 years.
27 bp × $100k face × 4.5 × 0.0001  =  $122 / $100k face
On $5M:                              $6,075

Verdict: "the bond is 27 bp rich on the dealer's screen vs my curve."
```

**Feed to your code:** parsed price → `01_bond_pricing_ytm.py` (YTM back-out) → `16_relative_value_spreads.py` (all three spreads).

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

**Screen quote:**
```
3M USD deposit          5.00%    91 days    notional $1M
```

**Decoded + cash:**
```
interest_USD  =  N × r × days/360  =  1M × 0.05 × 91/360  =  $12,638.89
future value  =  N + interest      =  $1,012,638.89
EAR equivalent =  (1 + 0.05 × 91/360)^(360/91) - 1  =  5.095%   ← higher than 5%

If the SAME 5% headline were quoted as GBP deposit (ACT/365):
   interest_GBP  =  1M × 0.05 × 91/365  =  $12,465.75    ← $173.14 LESS cash
```

**Feed to your code:** straight into `05_deposit_fra.py` task 1, or `02_day_count_conventions.py` to compare conventions.

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

**Screen quote:**
```
3×6 SOFR FRA          5.20%    settle T+2    91-day deposit period
```

**Decoded:**
```
F (contracted)    =  5.20%
T_1 (start)       =  3M from spot
T_2 (end)         =  6M from spot
δ                 =  91/360  =  0.252778
```

**Compare to your model (FRA fair rate from curve):**
```
Pull D(0, 3M) and D(0, 6M) from your bootstrapped curve:
   D_3M  ≈  0.98875       D_6M  ≈  0.97667

Apply the FRA formula (exercise 05):
   F_fair  =  (1/δ) × (D_3M / D_6M  -  1)
           =  (1 / 0.2528) × (0.98875 / 0.97667 - 1)
           =  4.892%

Quoted F:  5.20%             Your F_fair:  4.892%      Gap:  +31 bp (rich)
```

**Trade decision + settlement payoff:**
```
Quote is rich (31 bp above your fair) → SELL the FRA (pay floating L, receive F).

If market converges to 4.89% by fixing:
   payoff at T_1  =  N × (L - F) × δ / (1 + L·δ)
                  =  -10M × (0.04892 - 0.052) × 0.2528 / (1 + 0.04892×0.2528)
                  ≈  +$7,700 in your favour
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

**Screen quote:**
```
5y SOFR swap          c_par = 4.50%    annual fixed leg ACT/360
                      bid 4.498%       ask 4.502%
```

**Decoded:**
```
par rate     =  4.50%   (the trader receives or pays this for 5 years)
fixed leg    =  annual ACT/360
floating leg =  daily-compounded SOFR
settle       =  T+2
```

**Compare to your model (par rate from your curve):**
```
1. Bootstrap your curve up to 5y (exercises 08 + 09).
2. Apply the par-rate identity (exercise 10):
     c_fair  =  (1 - D(0, 5y))  /  A(0, 5y)
   where A(0, 5y) = sum_{i=1..5}  δ_i × D(0, i)  (=annuity factor).

3. Suppose your curve gives c_fair = 4.45%.   Gap: +5 bp (screen is rich).
```

**Mark-to-market at trade:**
```
For a $100M 5y receive-fixed swap struck at 4.50% (screen) when your
fair par rate is 4.45%, the trader receives 5 bp above fair:

   PV gain at trade  =  N × (c_screen - c_fair) × A(5y)
                      =  100M × 0.0005 × 4.286
                      =  $214,300                ← booked as Day-1 P&L
```

**Feed to your code:** `"5y USD SOFR swap at 4.50%"` → bootstrap → annuity → `11_swap_pricing.py` with `fixed_rate=0.045`, `freq=1`.

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

**Screen quote (Bankrate / lender website):**
```
30y fixed conforming    note rate 6.625%    APR 6.85%    points 0.5
```

**Decoded:**
```
note rate     =  6.625%   ← what your PMT formula uses (nominal, monthly comp)
APR           =  6.85%    ← includes 0.5 points + origination fees (Reg Z)
EAR (no fees) =  (1 + 0.06625/12)^12 - 1  =  6.83%

Gap APR(6.85%) - note(6.625%) = 22.5 bp ≈ what the 0.5 points + fees cost annualised.
```

**Fair value check (cashflow level):**
```
>>> monthly_payment(P=400_000, annual_rate_nominal=0.06625, years=30)
$2,561.07 / month

Total paid over 30y    =  $2,561.07 × 360  =  $921,985
Total interest         =  $921,985 - $400,000  =  $521,985    ← > principal
```

**Compare to your model:**
```
Borrower comparison: lender quotes APR; you compare lenders on APR (apples-to-apples).
Trader comparison:   compare the note rate against 10y Treasury or 10y MBS rate.

E.g.  10y Treasury = 4.07%       30y mortgage rate = 6.625%
      mortgage spread to 10y    = 255 bp  ← premium for prepayment + credit risk
```

**Feed to your code:** the *note rate* goes into `13_loan_amortisation.py`. APR is for borrower comparison only.

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
