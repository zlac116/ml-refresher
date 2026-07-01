# Core Quant Finance Formulas — Memorise These

The whiteboard set: formulas you should be able to reproduce from a blank cell, derive verbally, and explain in plain English. Everything else (SVI, Heston char fn, SABR Hagan, LSMC bases, Carr-Madan FFT, Black-Litterman matrix algebra) is **reach-for** material — recognise the name, look up the formula, never memorise.

**How to drill:** cover the right-hand side of each row, write the formula on a blank notebook cell from memory, then check. Daily 10-min retrieval beats hour-long passive reading. If you can derive it, the formula is just bookkeeping; if you can't, work through the linked notebook.

---

## 01 — Options

### Black-Scholes-Merton

$$d_1 = \frac{\ln(S/K) + (r - q + \tfrac{1}{2}\sigma^2)T}{\sigma\sqrt{T}}, \qquad d_2 = d_1 - \sigma\sqrt{T}$$

$$C = S e^{-qT} N(d_1) - K e^{-rT} N(d_2)$$

$$P = K e^{-rT} N(-d_2) - S e^{-qT} N(-d_1)$$

`N` is the standard-normal CDF; `q` is continuous dividend yield (drop if non-dividend stock).

### Put-call parity

$$C - P = S e^{-qT} - K e^{-rT}$$

Holds **independent of any model** — pure static replication. Use it to spot stale quotes, hard-to-borrow stocks, missed dividends.

### Black-76 (forward-based — swaptions, caps, FX, commodities)

$$C = e^{-rT}\big[F \cdot N(d_1) - K \cdot N(d_2)\big], \qquad d_1 = \frac{\ln(F/K) + \tfrac{1}{2}\sigma^2 T}{\sigma\sqrt{T}}$$

### Risk-neutral pricing identity

$$V_0 = e^{-rT} \, \mathbb{E}^Q\!\big[\Phi(S_T)\big]$$

The price of any European-style derivative is the discounted expectation of its payoff under the risk-neutral measure `Q`.

### Greeks (BS-Merton call; put differs by sign convention)

| Greek | Call formula | What it measures |
|---|---|---|
| **Delta** | $e^{-qT} N(d_1)$ | $\partial C / \partial S$ — hedge ratio |
| **Gamma** | $\dfrac{e^{-qT}\phi(d_1)}{S\sigma\sqrt{T}}$ | $\partial^2 C / \partial S^2$ — convexity (same for call & put) |
| **Vega** | $S e^{-qT} \phi(d_1)\sqrt{T}$ | $\partial C / \partial \sigma$ — vol sensitivity (same for call & put) |
| **Theta** | $-\dfrac{S e^{-qT}\phi(d_1)\sigma}{2\sqrt{T}} - rKe^{-rT}N(d_2) + qSe^{-qT}N(d_1)$ | $\partial C / \partial t$ — time decay |
| **Rho** | $KTe^{-rT}N(d_2)$ | $\partial C / \partial r$ — rate sensitivity |

`φ` is the standard-normal PDF.

---

## 02 — Risk

### Parametric VaR (normal returns)

$$\text{VaR}_\alpha = -\big(\mu + \sigma \cdot z_\alpha\big), \qquad z_\alpha = \Phi^{-1}(\alpha)$$

For 99% VaR: `z = -2.326`. Sign convention: VaR reported as a positive loss number.

### Expected Shortfall (normal)

$$\text{ES}_\alpha = -\mu + \sigma \cdot \frac{\phi(z_\alpha)}{1 - \alpha}$$

ES > VaR always; the gap measures tail thickness beyond VaR.

### Cornish-Fisher VaR (skew/kurtosis adjustment)

$$z_\alpha^{CF} = z_\alpha + \tfrac{1}{6}(z_\alpha^2 - 1)S + \tfrac{1}{24}(z_\alpha^3 - 3z_\alpha)K - \tfrac{1}{36}(2z_\alpha^3 - 5z_\alpha)S^2$$

Use when returns are clearly non-normal but you can estimate `S` (skew) and `K` (excess kurtosis).

### Merton structural credit

$$DD = \frac{\ln(V/D) + (\mu_V - \tfrac{1}{2}\sigma_V^2)T}{\sigma_V \sqrt{T}}, \qquad PD = N(-DD)$$

Default = firm value `V` falls below debt `D` at horizon `T`. Distance to default is just `d2` from BS in disguise.

---

## 03 — Fixed Income

### Bond price (discrete compounding)

$$P = \sum_{t} \frac{CF_t}{(1+y)^t}$$

Continuous compounding: `P = Σ CF_t · exp(-y·t)`.

### Macaulay & modified duration

$$D_{\text{Mac}} = \frac{1}{P}\sum_t t \cdot \frac{CF_t}{(1+y)^t}, \qquad D_{\text{Mod}} = \frac{D_{\text{Mac}}}{1+y}$$

Modified duration is the price elasticity to yield: $\partial P / \partial y \approx -D_{\text{Mod}} \cdot P$.

### Convexity

$$\text{Conv} = \frac{1}{P}\sum_t \frac{t(t+1) \cdot CF_t}{(1+y)^{t+2}}$$

### Price change (second-order)

$$\frac{\Delta P}{P} \approx -D_{\text{Mod}} \cdot \Delta y + \tfrac{1}{2} \cdot \text{Conv} \cdot (\Delta y)^2$$

### DV01 (dollar value of a basis point)

$$\text{DV01} = D_{\text{Mod}} \cdot P \cdot 0.0001$$

### Forward rate (continuous compounding)

$$f(t_1, t_2) = \frac{1}{t_2 - t_1} \ln\frac{P(0, t_1)}{P(0, t_2)}$$

### Bond from forward LIBORs (telescoping product)

$$P(t, T_n) = P(t, T_0) \prod_{i=0}^{n-1} \frac{1}{1 + \delta_i L_i(t)}, \qquad L_i(t) = \frac{1}{\delta_i}\!\left(\frac{P(t, T_i)}{P(t, T_{i+1})} - 1\right)$$

The discrete analogue of $P(t, T) = \exp(-\int_t^T r(u)\,du)$ — replace integration over instantaneous short rates with a product over forward LIBORs. Used everywhere in fixed income: swap pricing, caplet payoffs, the LMM simulator, curve construction.

### Par swap rate (from discount curve)

$$S = \frac{P(0, T_0) - P(0, T_n)}{\sum_{i=1}^n \tau_i \cdot P(0, T_i)}$$

Numerator = floating leg PV, denominator = fixed leg PV01. Sets PV(swap) = 0 at inception.

### LIBOR Market Model (LMM / BGM) — single-line specification

$$dL_i(t) = \sigma_i(t)\, L_i(t)\, dW_i^{T_{i+1}}(t), \qquad i = 0, 1, \dots, N-1$$

Each forward LIBOR is lognormal under its **own** $T_{i+1}$-forward measure. Caplets price by Black-76 with vol $\sigma_i$ — the model is calibrated to the cap-vol surface by construction.

The drift adjustment when switching to a common measure (terminal $Q^{T_N}$ or spot $Q^B$) is **derived on the whiteboard from Girsanov + numeraire change** — don't memorise the index ranges and signs, derive them. See `03_fixed_income/05_libor_market_model.ipynb`.

---

## 04 — Portfolio

### Sharpe ratio

$$\text{SR} = \frac{\bar R - R_f}{\sigma_R}$$

Annualise: multiply by `√252` for daily returns, `√52` for weekly, `√12` for monthly.

### Markowitz mean-variance

Minimise $\tfrac{1}{2} w^\top \Sigma w$ subject to $w^\top \mu = \mu^*, \, \mathbf{1}^\top w = 1$.

Closed-form for unconstrained tangency portfolio (after subtracting `R_f`):

$$w_{\text{tan}} \propto \Sigma^{-1} (\mu - R_f \mathbf{1})$$

Normalise so weights sum to 1.

### CAPM

$$\mathbb{E}[R_i] = R_f + \beta_i \big(\mathbb{E}[R_m] - R_f\big), \qquad \beta_i = \frac{\text{Cov}(R_i, R_m)}{\text{Var}(R_m)}$$

### Risk parity (equal risk contribution)

$$w_i \cdot (\Sigma w)_i = w_j \cdot (\Sigma w)_j \quad \forall i, j$$

Each asset contributes the same amount to portfolio variance. No closed-form; solve numerically.

---

## 05 — Volatility

### Realised vol (close-to-close)

$$\hat\sigma = \sqrt{\frac{252}{N-1} \sum_{t=1}^{N} (r_t - \bar r)^2}, \qquad r_t = \ln(S_t / S_{t-1})$$

### Parkinson (high-low range)

$$\hat\sigma^2_{\text{Park}} = \frac{1}{4 \ln 2} \cdot \frac{1}{N} \sum_{t=1}^{N} \ln^2(H_t / L_t)$$

5× more efficient than close-to-close when intraday range is observable.

### GARCH(1,1)

$$\sigma_t^2 = \omega + \alpha \cdot r_{t-1}^2 + \beta \cdot \sigma_{t-1}^2$$

Long-run variance: $\sigma_\infty^2 = \omega / (1 - \alpha - \beta)$. Stationarity requires $\alpha + \beta < 1$.

### Dupire local volatility

$$\sigma_{\text{loc}}^2(K, T) = \frac{\partial C / \partial T + (r - q) K \, \partial C / \partial K + qC}{\tfrac{1}{2} K^2 \, \partial^2 C / \partial K^2}$$

The unique local-vol surface that exactly reproduces today's call-price surface.

---

## 06 — Stochastic Calculus

### Geometric Brownian Motion

$$dS_t = \mu S_t \, dt + \sigma S_t \, dW_t$$

Solution: $S_t = S_0 \exp\!\left((\mu - \tfrac{1}{2}\sigma^2) t + \sigma W_t\right)$. Note the `−σ²/2` drag — the median grows slower than the mean.

### Itô's lemma (1D)

For $f(t, S_t)$ with $dS = a \, dt + b \, dW$:

$$df = \left(\partial_t f + a \, \partial_S f + \tfrac{1}{2} b^2 \, \partial_{SS} f\right) dt + b \, \partial_S f \, dW$$

The `½ b² ∂²/∂S²` term is what distinguishes stochastic from ordinary calculus.

### The one stochastic-calc rule

$$(dW)^2 = dt, \qquad dW \cdot dt = 0, \qquad (dt)^2 = 0$$

Everything in BS, Heston, and any SDE manipulation flows from this.

### Itô isometry

$$\mathbb{E}\!\left[\left(\int_0^T f_s \, dW_s\right)^2\right] = \mathbb{E}\!\left[\int_0^T f_s^2 \, ds\right]$$

The variance of an Itô integral is the integral of the variance.

### Girsanov change of measure

Define $L_t = \exp\!\big(-\int_0^t \theta_s \, dW_s - \tfrac{1}{2}\int_0^t \theta_s^2 \, ds\big)$ and $dQ/dP = L_T$.

Then $\tilde W_t = W_t + \int_0^t \theta_s \, ds$ is a `Q`-Brownian motion.

This is how you switch between physical (`P`) and risk-neutral (`Q`) measures. For BS: `θ = (μ - r) / σ`.

---

## What NOT to memorise

These are reach-for material — know the name, know when they apply, look up the formula:

| Topic | Where to look |
|---|---|
| SVI parameterisation (Gatheral 2004) | `01_options/06_implied_vol_surface.ipynb` |
| Heston characteristic function (Lewis 2001 form) | `01_options/07_heston.ipynb` |
| Carr-Madan FFT pricing | `01_options/07_heston.ipynb` |
| SABR Hagan formula | `01_options/02_bs_family_and_asset_classes.ipynb` |
| Bachelier closed form | `01_options/02_bs_family_and_asset_classes.ipynb` |
| Brenner-Subrahmanyam IV approximation | `01_options/01_black_scholes.ipynb` |
| Longstaff-Schwartz (LSMC) regression bases | `06_stoch_calc/03_lsmc_american.ipynb` |
| Black-Litterman views matrix | `04_portfolio/02_black_litterman.ipynb` |
| Nelson-Siegel curve fit | `03_fixed_income/03_curve_building.ipynb` |
| Heston, SABR, local vol calibration loss functions | their respective notebooks |
| Specific quasi-MC scrambling schemes | `01_options/05_monte_carlo_pricing.ipynb` |
| LMM drift formulas (terminal / spot LIBOR measure) — derive from Girsanov, don't memorise index ranges and signs | `03_fixed_income/05_libor_market_model.ipynb` |
| Rebonato correlation parameterisation, predictor-corrector LMM scheme, LMM cap-vol bootstrap | `03_fixed_income/05_libor_market_model.ipynb` |

---

## Daily drill protocol

1. Open a blank notebook cell.
2. Pick one section (rotate through).
3. Write each formula from memory.
4. Check against this file.
5. For ones you missed, also implement them in code (e.g., write `bs_call(S, K, T, r, sigma)` from scratch — no peeking).
6. 10-15 minutes per session. The whole sheet drilled in a week.

Retrieval is the only thing that cements; re-reading is theatre.
