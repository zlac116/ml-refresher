# Curve-Building Cheatsheet — Foundations to Forward Swap Rates

**The point**: every curve formula you'll meet on a rates desk is a sentence in the same alphabet. Memorise the foundations (8 formulas), and everything else — including the swap bootstrap, the forward swap rate, DV01 — falls out in 3-4 lines of algebra. This sheet is the alphabet.

---

## The foundational identity

EVERYTHING in curve building starts from one statement:

> $D(0, t) =$ price today of receiving \$1 at time $t$

That's the discount factor. Different compounding conventions express the same $D$ differently:

| Compounding | Formula | Used in |
|---|---|---|
| **Simple** (linear) | $D(0, t) = 1 / (1 + r \cdot \delta)$ | Deposits, FRAs, swap floating fixings |
| **Continuous** | $D(0, t) = \exp(-z(t) \cdot t)$ | Curve storage, modelling |
| **Periodic** ($m$/year) | $D(0, t) = (1 + y/m)^{-m \cdot t}$ | Bond YTM quotes |

All three equal the same $D$. Same information, different units. Convert freely.

---

## 1. Short-end deposit → discount factor

**Given**: a simple deposit rate $r$ at tenor $t$ (e.g., 6M deposit at 4.10%).

**Formula**:
$$D(0, t) = \frac{1}{1 + r \cdot \delta}$$
where $\delta$ is the act/360 day-count fraction.

**Worked example** — 6M deposit at 4.10%, act/360:
$$\delta = 182 / 360 = 0.5056 \\
D(0, 0.5) = 1 / (1 + 0.041 \cdot 0.5056) = 0.9797$$

**Why**: deposit \$1 today, receive $1 + r\delta$ at maturity. For both to have the same value today: $1 = (1 + r\delta) \cdot D(0, t)$. Solve.

---

## 2. Discount factor ↔ zero rate

**Forward direction** (DF → z):
$$z(t) = -\frac{\ln D(0, t)}{t}$$

**Inverse** (z → DF):
$$D(0, t) = e^{-z(t) \cdot t}$$

**Worked example**:
$$D(0, 0.5) = 0.9797 \\
z(0.5) = -\ln(0.9797) / 0.5 = 0.0410 = 4.10\%$$

**Why**: define $z(t)$ implicitly by $D = e^{-z \cdot t}$. Take ln, solve.

**Why bother?** Zero rates are the storage convention. Smooth interpolation. Universal comparison across instruments.

---

## 3. Forward discount factor (between two future dates)

**Formula**:
$$D(T_1, T_2) = \frac{D(0, T_2)}{D(0, T_1)}$$

The discount factor between two **future** dates, observed today.

**Worked example**:
$$D(0, 1) = 0.9614, \quad D(0, 2) = 0.9231 \\
D(1, 2) = 0.9231 / 0.9614 = 0.9602$$

**Why**: by no-arbitrage, holding a $T_2$-bond is equivalent to holding a $T_1$-bond and reinvesting at the forward rate. Ratio of bond prices = forward bond price.

**Where it's used**: forward measures, Hull-White / HJM bond pricing, the analytic ZCB formula in `src/pricers/hw.py`.

---

## 4. Simple-compounded forward rate

**Formula**:
$$F(T_1, T_2) = \frac{1}{\delta} \left( \frac{D(0, T_1)}{D(0, T_2)} - 1 \right), \quad \delta = T_2 - T_1$$

**Worked example** — 1y forward over [1, 2]:
$$D(0, 1) = 0.9614, \quad D(0, 2) = 0.9231 \\
F(1, 2) = (1/1) \cdot (0.9614 / 0.9231 - 1) = 0.0415 = 4.15\%$$

**Why** (no-arb derivation in 4 lines):

1. Forward deposit costs zero today: at $T_1$ deposit \$1, at $T_2$ receive $1 + F\delta$
2. Replicate: sell $T_1$-bond face \$1 (get $D(0,T_1)$ today, owe \$1 at $T_1$) + buy $T_2$-bond face $1 + F\delta$ (pay $(1+F\delta) \cdot D(0,T_2)$ today, receive $1+F\delta$ at $T_2$)
3. Replication net cost today must also be zero: $D(0,T_1) = (1+F\delta) \cdot D(0,T_2)$
4. Solve for $F$ → formula above

**Where it's used**: every floating fixing on a swap, every caplet/floor/FRA payoff, every $L_i$ in the LIBOR Market Model.

---

## 5. Instantaneous forward rate

**Formula**:
$$f(t) = -\frac{\partial \ln D(0, t)}{\partial t}$$

A **single point** in time, not an interval. Limit of $F(T_1, T_2)$ as $T_2 \to T_1$.

**Where it's used**: continuous-time models — Hull-White, Heath-Jarrow-Morton. Doesn't appear in vanilla swap pricing.

**Don't confuse with $F(T_1, T_2)$**: simple forward is a chunky bar over a finite period; instantaneous forward is a single dot at one instant.

---

## 6. The annuity

**Definition**:
$$A = \sum_{i=1}^{n} \delta_i \cdot D(0, T_i)$$
where $\delta_i$ is the year fraction for period $i$ and $T_i$ are the swap's payment dates.

**Worked example** — 3y annual swap:
$$D(0,1) = 0.9614, \; D(0,2) = 0.9231, \; D(0,3) = 0.8853 \\
A = 1 \cdot 0.9614 + 1 \cdot 0.9231 + 1 \cdot 0.8853 = 2.7698$$

**Three interpretations**:

1. **As a price**: PV of receiving \$1 per period over the swap life (real annuity insurance — that's the name)
2. **As a sensitivity (PV01)**: $\text{PV}_{\text{fixed}} = K \cdot A$ ⇒ $\partial \text{PV}/\partial K = A$. Multiply by 1bp = \$ change per bp move
3. **As a numéraire**: under the swap measure, the par swap rate is a martingale → makes Black-76 swaption pricing work

---

## 7. Floating-leg telescoping identity

**Formula**:
$$\text{PV}_{\text{float}} = 1 - D(0, T_n)$$

**Worked example** (3y swap):
$$D(0, 3) = 0.8853 \\
\text{PV}_{\text{float}} = 1 - 0.8853 = 0.1147 \quad \text{(per unit notional)}$$

**Why** — derive once, never forget:

Each floating cash flow $\delta_i \cdot L_i$ paid at $T_i$, where $L_i = (1/\delta_i)(D(0,T_{i-1})/D(0,T_i) - 1)$ is the simple forward.

PV of cash flow $i$:
$$\text{PV}_i = \delta_i \cdot L_i \cdot D(0, T_i) = D(0, T_{i-1}) - D(0, T_i) \quad \text{(} \delta_i \text{ cancels)}$$

Sum over all periods:
$$\text{PV}_{\text{float}} = \sum_{i=1}^{n} \big[D(0, T_{i-1}) - D(0, T_i)\big] = D(0, T_0) - D(0, T_n) = 1 - D(0, T_n)$$

Middle terms cancel pairwise (telescoping); $T_0 = 0$ and $D(0, 0) = 1$.

**This is the trick that makes swap bootstrapping possible.** The whole forward strip is implicitly priced by the endpoint DF.

---

## 8. Par swap rate (spot-starting)

**Formula**:
$$K_{\text{par}} = \frac{1 - D(0, T_n)}{A} = \frac{\text{floating PV}}{\text{annuity}}$$

**Worked example** (3y swap):
$$K_{\text{par}} = (1 - 0.8853) / 2.7698 = 0.0414 = 4.14\%$$

**Why**: par rate = fixed $K$ that makes swap PV = 0:
$$\text{PV}_{\text{payer}} = \text{PV}_{\text{float}} - K \cdot A = 0 \quad \Rightarrow \quad K_{\text{par}} = \text{PV}_{\text{float}} / A$$

In plain English: **par rate = floating PV per unit of annuity**.

---

## 9. Forward swap rate (forward-starting swap)

**Formula** — same as par swap rate but with $T_{\text{start}} > 0$:
$$K_{\text{fwd}}(T_{\text{start}}, T_{\text{end}}) = \frac{D(0, T_{\text{start}}) - D(0, T_{\text{end}})}{A_{\text{fwd}}}$$
where
$$A_{\text{fwd}} = \sum_{i: \, T_{\text{start}} < T_i \leq T_{\text{end}}} \delta_i \cdot D(0, T_i)$$

The annuity sum runs over the swap's payment dates **only after $T_{\text{start}}$**.

**Worked example** — 5y forward swap rate (10y maturity, swap starts in 5y, ends in 15y):
Suppose $D(0, 5) = 0.815$, $D(0, 15) = 0.582$, and the annual annuity over years 6-15 sums to $A_{\text{fwd}} = 6.92$.
$$K_{\text{fwd}}(5, 15) = (0.815 - 0.582) / 6.92 = 0.233 / 6.92 = 0.0337 = 3.37\%$$

**Why** — same derivation as the spot par rate, just with non-zero $T_{\text{start}}$.

The floating leg's telescoping identity for a forward-starting swap:
$$\text{PV}_{\text{float}} = D(0, T_{\text{start}}) - D(0, T_{\text{end}})$$

(Telescoping again — middle DFs cancel; endpoints survive.) Divide by the forward annuity → forward swap rate.

**Where it's used**:
- **Swaption pricing** — the forward swap rate $F$ that goes into Black-76 / SABR for a $T_e \times \text{tail}$ swaption is exactly $K_{\text{fwd}}(T_e, T_e + \text{tail})$
- **Forward starting swaps** — quoted directly in the market for hedging future-period rates
- **CMS pricing** — constant-maturity swap fixings

**Same function as spot**: in `src/curves.py`, calling `par_swap_rate(disc, proj, T_start=5, T_end=15, freq=1)` returns the **forward** swap rate. The label "par" vs "forward" is implicit in whether $T_{\text{start}}$ is 0 or positive.

---

## 10. Bootstrap a swap node

**Given**:
- Prior nodes already bootstrapped: $D(0, T_1), \ldots, D(0, T_{n-1})$
- A new par-rate quote $c_n$ for a swap maturing at $T_n$

**Formula**:
$$D(0, T_n) = \frac{1 - c_n \cdot \sum_{i < n} \delta_i \cdot D(0, T_i)}{1 + c_n \cdot \delta_n}$$

**Worked example** — bootstrap 3y DF given prior 1y, 2y DFs:
$$D(0, 1) = 0.9614, \quad D(0, 2) = 0.9231 \\
c_3 = 0.0414 \quad \text{(market quote, annual freq)}$$

Numerator: $1 - 0.0414 \cdot (1 \cdot 0.9614 + 1 \cdot 0.9231) = 1 - 0.0780 = 0.9220$
Denominator: $1 + 0.0414 \cdot 1 = 1.0414$

$$D(0, 3) = 0.9220 / 1.0414 = 0.8854 \quad \checkmark$$

**Why** — set fixed PV = floating PV at par, isolate the unknown DF:
$$c_n \cdot \sum_{i \leq n} \delta_i \cdot D(0, T_i) = 1 - D(0, T_n) \\
\Rightarrow D(0, T_n) \cdot (1 + c_n \cdot \delta_n) = 1 - c_n \cdot \sum_{i < n} \delta_i \cdot D(0, T_i) \\
\Rightarrow \text{formula above}$$

**Don't memorise this formula** — derive it from the foundations (telescoping + annuity + at-par condition). 4 lines of algebra. Easier to derive on demand than to memorise the formula and worry about index ranges.

---

## 11. Swap DV01

**Formula**:
$$\text{DV01} = A \cdot \text{notional} \cdot 10^{-4}$$
or equivalently via central-difference bumping:
$$\text{DV01} \approx \frac{\text{PV}(z - 1\text{bp}) - \text{PV}(z + 1\text{bp})}{2}$$

**Worked example** — 5y at-par USD swap, \$100M notional, annuity $A = 4.55$:
$$\text{DV01} \approx 4.55 \cdot 10^8 \cdot 10^{-4} = \$45{,}500 \text{ per bp}$$

**Sign convention**: under the central-difference form $(\text{PV}_{-1bp} - \text{PV}_{+1bp})/2$:
- **Receiver** swap → DV01 > 0 (long duration, gains when rates fall)
- **Payer** swap → DV01 < 0 (short duration, loses when rates fall)

Some systems (Bloomberg, MX.3) quote $\partial \text{PV}/\partial y$ with positive = PV rises with +1bp — opposite sign. Check the convention before trusting any number.

---

## 12. Multi-curve discounting (post-2008)

The telescoping identity $\text{PV}_{\text{float}} = 1 - D(0, T_n)$ assumes **the floating index curve equals the discount curve**. Pre-2008 this was true. Post-2008, swaps collateralised under CSA discount on **OIS** but project floating cash flows on the **LIBOR / SOFR projection curve** — and the two differ.

In a true multi-curve world:
$$\text{PV}_{\text{float}} = \sum_i \delta_i \cdot F^{\text{proj}}_i \cdot D^{\text{disc}}(0, T_i)$$

with $F^{\text{proj}}_i$ from the projection curve and $D^{\text{disc}}$ from the OIS curve. **Telescoping no longer applies** — bootstrapping becomes a system of equations across multiple curves, solved jointly.

| World | Discount curve | Projection curve | Telescoping holds? |
|---|---|---|---|
| Single-curve (pre-2008) | LIBOR | LIBOR (same) | yes |
| Multi-curve (post-2008) | OIS / SOFR-OIS / €STR | LIBOR / SOFR / EURIBOR | **no** |

The basis was 30-50bp during the 2008 stress, smaller in normal times — but enough that mixing curves gives wrong PVs.

**In your scaffolding**: single-curve world. `par_swap_rate(disc=Curve, proj=Curve, ...)` works either way; pass `proj=disc` and you get the telescoping fast path.

---

## What to actually memorise — the keep list

These 8 formulas cover the whole curve-building world. Everything else (including swap bootstrap, forward swap rate, DV01) is derived from them in 3-4 lines.

| # | Formula | Why memorise |
|---|---|---|
| 1 | $D = 1 / (1 + r\delta)$ | All deposit pricing |
| 2 | $D = e^{-zt}$ | Convert to/from zero rates |
| 3 | $z = -\ln D / t$ | The inverse |
| 4 | $D(T_1, T_2) = D(0, T_2) / D(0, T_1)$ | Forward bonds, HW pricing |
| 5 | $F(T_1, T_2) = (1/\delta)(D(T_1)/D(T_2) - 1)$ | All simple forwards (swap fixings, caplets) |
| 6 | $f(t) = -\partial \ln D / \partial t$ | Continuous-time models |
| 7 | $A = \sum \delta_i D(0, T_i)$ | Annuity (used everywhere) |
| 8 | $\text{PV}_{\text{float}} = 1 - D(0, T_n)$ | The telescoping trick |

**Composes**:
- Spot par swap rate: $(1 - D(0, T_n)) / A$
- Forward swap rate: $(D(0, T_{\text{start}}) - D(0, T_{\text{end}})) / A_{\text{fwd}}$
- Swap bootstrap: solve fixed PV = floating PV at the par quote
- DV01: $A \cdot \text{notional} \cdot 10^{-4}$ (or central-diff)

---

## The mental algorithm for any curve question

1. **Anything a discount factor can answer** → use $D$ directly
2. **Anything about a forward rate** → use the simple forward formula (swap fixings, caplets)
3. **Anything about a swap PV** → fixed leg = $K \cdot A$, floating leg = $1 - D(0, T_n)$
4. **Anything about par or forward swap rates** → floating PV / annuity, where the annuity sum runs over the swap's payment dates
5. **Anything about bootstrap** → set fixed = floating at the par quote, isolate the unknown DF
6. **Anything about DV01** → $A \cdot \text{notional} \cdot 10^{-4}$ or central-diff bumping
7. **Anything multi-curve** → telescoping fails; project on one curve, discount on another, sum cash flows explicitly

---

## Common gotchas

- **Day-count conventions are silent killers.** USD swap fixed leg act/360, EUR swap fixed leg 30/360. USD govvies act/act, USD corp 30/360. Mixing day counts gives 1-2% systematic errors that look like model bugs.
- **Sign convention on DV01.** Half the world quotes $(P_{\text{down}} - P_{\text{up}})/2$ (receiver positive), the other half quotes $\partial P / \partial y$ (payer positive). Always check.
- **Multi-curve vs single-curve.** Don't accidentally use the OIS curve for floating projection (or vice versa). Telescoping only works in single-curve world.
- **Forward annuity sum range.** When computing a forward swap rate, the annuity sums **only over the future swap period** ($T_{\text{start}} < T_i \leq T_{\text{end}}$), NOT from time zero.
- **Forward swap rate via wrong identity.** $K_{\text{fwd}}$ is **not** $(1 - D(0, T_{\text{end}})) / A$ — that's the spot par rate. Use $(D(0, T_{\text{start}}) - D(0, T_{\text{end}}))/A_{\text{fwd}}$.
- **Continuous vs simple zero rate.** Same DF, different rate values: $z_{\text{cc}} \neq z_{\text{simple}}$ except at $t = 0$. Conversion: $z_{\text{cc}} = \ln(1 + z_{\text{simple}} \cdot \delta) / t$ for finite tenors, or $z_{\text{cc}} = \ln(1 + r/m)^m$ for periodic compounding $m$ times per year.

---

## Cross-references

| Concept | Where covered |
|---|---|
| Full curve bootstrap (deposits + swaps) | `03_fixed_income/03_curve_building.ipynb` |
| Floating-leg telescoping derivation | `03_fixed_income/03_curve_building.ipynb` Step 2 |
| Annuity sidebar | `03_fixed_income/03_curve_building.ipynb` Step 2 |
| Forward rates derivation (simple + instantaneous) | `03_fixed_income/03_curve_building.ipynb` Step 4 |
| Multi-curve world (OIS vs LIBOR/SOFR) | `03_fixed_income/03_curve_building.ipynb` Step 2 caveat |
| Swap PV via $-A(K - K_{\text{par}})$ | `03_fixed_income/04_swaps_swaptions.ipynb` |
| DV01 of a swap, hedge ratios | `03_fixed_income/04_swaps_swaptions.ipynb` |
| Bond duration, convexity, KRDs | `03_fixed_income/02_duration_convexity_krd.ipynb` |
| Forward swap rate as input to SABR | `01_options/02_bs_family_and_asset_classes.ipynb` Part 2 |
| Whiteboard formulas (consolidated) | `quant_finance/CORE_FORMULAS.md` |

---

## Interview-grade summary (one paragraph)

The discount curve is the single underlying object — everything else (zero rates, simple forwards, instantaneous forwards, the forward swap rate, the annuity, the bootstrap recipe) is just a different lens on it. Two market quotes pin two DF nodes (deposit at the short end, swap at the long end), interpolation fills in the middle. The telescoping identity ($\text{PV}_{\text{float}} = 1 - D(0, T_n)$) is the trick that makes the whole thing solvable: each new swap quote pins exactly one new DF node, since the floating leg only depends on the endpoint. Forward swap rates fall out as $(D(0, T_{\text{start}}) - D(0, T_{\text{end}})) / A_{\text{fwd}}$ — generalising the spot par rate to non-zero start dates. Multi-curve discounting (post-2008) breaks the telescoping; in production, OIS discounts everything and per-index projection curves provide forward fixings, bootstrapped jointly across the system.
