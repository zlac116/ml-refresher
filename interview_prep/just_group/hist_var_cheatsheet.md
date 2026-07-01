# Historical-Simulation VaR Stress Engine — Cheatsheet
### For the Just role: extend a Python stress engine to bonds, credit spreads, etc.  ·  *revise out loud*

## ⭐ MUST-KNOW COLD
1. **Historical-simulation VaR** = re-apply **each past day's risk-factor move** to *today's* portfolio, **revalue**, collect the **P&L distribution**, take the tail loss. No distributional assumption — the history *is* the distribution.
2. **99.5% / 1-year is the Solvency II SCR** market-risk calibration → this engine likely feeds **capital**.
3. **VaR(99.5%) = −sorted_pnl[ floor((1−0.995)·N) ]** — the loss at the `0.5%`-worst scenario.
4. **Extending to a new instrument = add (a) a pricing function and (b) its risk factors to the scenario set** — *not* an engine rewrite.
5. **Shock conventions differ by factor:** rates & **credit spreads → absolute (bp)**; equities & FX → **relative (%/log)**. Getting this right is the classic error.
6. **Full revaluation > sensitivity** for a 99.5% tail because of **convexity** (big moves); sensitivity (DV01/CS01) is the fast approximation.
7. **Bond risk factors = the rate curve + the credit spread.** **CS01** = P&L per 1bp spread; **DV01** = per 1bp rate.
8. **ES (Expected Shortfall)** = average loss *beyond* VaR — tail-aware; FRTB & many internal models prefer it.

---

## 1. The algorithm (own these steps)
1. **Lookback window:** `N` historical days (≈ 1–3 yrs; 250–750).
2. **Scenarios:** for each day, the **change in every risk factor** (Δrate, Δspread, Δequity, ΔFX).
3. **Apply to today:** shock *today's* market inputs by that day's move.
4. **Revalue:** `P&L_i = V(shocked_i) − V(base)`.
5. **Distribution:** the `N` P&Ls. **Sort ascending.**
6. **Tail:** `VaR(conf) = −sorted_pnl[ floor((1−conf)·N) ]` (loss as a positive number).
```
VaR(99.5%) = −sorted_pnl[ floor(0.005 · N) ]      ES = −mean( losses beyond VaR )
```
> **Example:** N=250 → `floor(0.005·250)=1` → VaR is the **2nd-worst** P&L. (250 scenarios is thin for 99.5% — real engines use longer windows or scale.)

## 2. Risk factors & shock conventions
- **Rates / credit spreads → ABSOLUTE bp moves:** `s_new = s_today + Δs_hist`.
- **Equities / FX → RELATIVE returns:** `S_new = S_today · (1 + r_hist)` (or log).
- **Why it matters:** applying a 2008 *relative* rate move to today's low rates (or vice-versa) gives nonsense. Match the convention to the factor.

## 3. Revaluation: full reval vs sensitivity
- **Full revaluation:** reprice the instrument under each shocked market. **Accurate, captures convexity & cross-effects.** Cost: pricing × N scenarios.
- **Sensitivity (Taylor):** `ΔP ≈ −DV01·Δy − CS01·Δs + ½·Conv·Δy²`. **Fast**, but misses higher-order terms on big shocks.
- **For 99.5% tail / bonds → prefer full reval** (convexity bites). Use sensitivity for speed or sanity checks.

## 4. Extending to bonds & credit (the actual task)
- **Bond price (full reval):** `P = Σ CFₜ · exp(−(zₜ + s)·t)` — discount on **rate curve `z` + credit spread `s`**.
- **Risk factors to add:** the **rate curve points** *and* the **credit-spread (by rating/sector/issuer)** — both must enter the historical scenario set.
- **CS01** = `P(s) − P(s+1bp)` ≈ spread-duration × price × 1bp. **DV01** = `P(y) − P(y+1bp)`.
- **CDS / hazard-rate instruments:** value via **survival probabilities** `Q(t)=exp(−∫λ)`; spread shocks move the hazard/credit curve.
- **Mapping/proxy:** a bond with no own spread history → **proxy** to a rating/sector spread series.
- **Watch:** FX for non-base-currency bonds; pull-to-par; coupon/accrual; callability (optionality → needs a model, not just discounting).

## 5. Engine architecture / extensibility (they'll probe your design)
- **`Instrument` interface** with `revalue(market)`; each instrument **declares its risk-factor dependencies**.
- **Scenario engine:** historical Δrisk-factors → build `shocked_market` → revalue all instruments → **portfolio P&L vector** → percentile.
- **Add an instrument = new pricer class + its risk-factor series**; the engine stays generic (it never knows instrument internals).
- **Decisions to raise:** full-reval vs sensitivity, absolute-vs-relative per factor, risk-factor **proxying**, curve interpolation, **vectorisation** (numpy: scenarios × factors matrix), caching the base valuation.

## 6. Horizon, scaling, and cousins
- **Holding period:** 1-day VaR → scale to h-day by **√h** (i.i.d. assumption), or use **overlapping h-day returns**. **SCR = 1-year, 99.5%.**
- **Filtered Historical Simulation (FHS):** scale historical moves by **current vs historical volatility** (e.g. EWMA) so the engine is responsive — good depth to mention.
- **Anti-procyclicality (APC):** floors/buffers so capital doesn't collapse in calm and spike in stress.
- **Backtesting:** count exceptions vs expected (e.g. ~2.5 breaches/yr at 99% daily) — regulatory traffic-light.

## 7. Likely questions → answers
- **"Add bonds to the engine?"** → a bond pricer consuming **rate curve + credit spread**; add the spread series to scenarios; **full-reval** each scenario.
- **"How handle credit spreads?"** → CS01 / spread curve, **absolute bp** shocks, **proxy** missing names; CDS via hazard rates.
- **"Write 99.5% historical VaR from a P&L vector."** → sort, `idx=floor(0.005·N)`, `−sorted[idx]` (mind the off-by-one & sign).
- **"Full reval vs sensitivity trade-off?"** → accuracy/convexity vs speed; tail favours full reval.
- **"Design so a new instrument is easy?"** → instrument/pricer abstraction + risk-factor registry; engine generic.
- **"VaR vs ES?"** → ES averages the tail beyond VaR; coherent; FRTB uses ES.

## 8. Gotchas (say these unprompted — signals seniority)
- **Percentile indexing / off-by-one** and **P&L sign** (loss = negative P&L).
- **Absolute vs relative** shocks per factor.
- **Window length vs confidence:** 99.5% needs a long history (or scaling) — 250 days is thin.
- **Missing risk-factor history → proxy**, and **stale/holiday data alignment** across factors.
- **Ghost effect:** a big move drops out of the window after N days → VaR jumps; FHS mitigates.

## 📐 Formula quick-ref
```
Bond (full reval)  P = Σ CFₜ·exp(−(zₜ + s)·t)        s = spread_bp/1e4
DV01               P(y) − P(y+1bp)                   CS01 = P(s) − P(s+1bp)
Scenario P&L_i     V(shock today by day-i moves) − V(base)
Hist VaR(conf)     −sorted_pnl[ floor((1−conf)·N) ]
ES(conf)           −mean( pnls below the VaR threshold )
h-day scaling      VaR_h = VaR_1day · √h             SCR = 99.5%, 1-year
Rates/spreads      absolute bp shock   |  Equity/FX  relative % shock
```

*Practise the engine end-to-end in **`hist_var_drill.py`** (fill-in, graded): bond pricer → DV01/CS01 → P&L vector by full reval → 99.5% VaR. Key `hist_var_drill_SOLUTIONS.py`.*
