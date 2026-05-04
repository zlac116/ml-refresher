# Bachelier (Normal Model) Cheatsheet — Black-76 With the Lognormality Removed

**The point:** when the forward `F` can be near zero or negative (rates regime, oil futures crash), Black-76 dies — you can't take $\ln F$ on a non-positive number. Bachelier swaps geometric Brownian for arithmetic Brownian — same risk-neutral pricing argument, no positivity constraint.

---

## The single specification

$$\boxed{\;dF_t = \sigma_n \, dW_t\;}$$

The vol $\sigma_n$ is in **absolute units** — bps per √year for rates, dollars per √year for prices — not a percentage. The forward `F` can drift to $\pm\infty$.

Closed-form prices, with $d = (F - K) / (\sigma_n \sqrt{T})$:

$$\boxed{\;\text{Call} = e^{-rT}\big[(F - K)\,N(d) + \sigma_n\sqrt{T}\,\phi(d)\big]\;}$$

$$\boxed{\;\text{Put}  = e^{-rT}\big[(K - F)\,N(-d) + \sigma_n\sqrt{T}\,\phi(d)\big]\;}$$

$\phi$ is the standard-normal PDF (not CDF).

**Bachelier put-call parity:** $C - P = e^{-rT}(F - K)$ — no exponential of $F$ (vs. Black-76 which has $e^{-qT}S - e^{-rT}K$ with the spot/forward link).

---

## Bachelier vs Black-76 — when each applies

| Regime | $F$ value | Use | Why |
|---|---|---|---|
| Standard rates | $F$ well above zero, low vol | **Black-76** | Industry default; lognormal fits empirically |
| Distressed / negative rates | $F$ near zero or negative | **Bachelier** | Lognormal blows up; normal handles $F \le 0$ natively |
| Borderline (positive but low) | $F$ small, positive | **Shifted-lognormal** | Black-76 on $F + s$ with shift $s > 0$ |
| Oil futures (April 2020 crash) | $F$ went to $-37$/barrel | **Bachelier** | Only normal model can quote it |
| EUR rates 2014–2022 | ECB deposit at $-0.5\%$ | **Bachelier** (or shifted) | Lognormal can't represent negative rates |
| SOFR caps/floors (post-2022) | $F$ low but positive | **Bachelier** is desk default | Quoting convention switched |

---

## Closed forms worth memorising (ATM)

When $F = K$ (ATM), $d = 0$, $N(0) = 0.5$, $\phi(0) = 1/\sqrt{2\pi}$. Three identities fall out:

| Quantity | ATM closed form | Notes |
|---|---|---|
| **Call price** | $\sigma_n \sqrt{T / (2\pi)}$ | Ignore $r$ in the limit |
| **Vega** | $e^{-rT} \sqrt{T / (2\pi)}$ | **Constant in $\sigma_n$** at ATM (vs. Black-76 vega which depends on $\sigma$) |
| **Delta** | **Exactly 0.5** | Unlike Black-76, where ATM delta > 0.5 (the famous "ATM is not 0.5" trap) |

---

## Bachelier ↔ Black-76 vol conversion

Given a Bachelier vol $\sigma_n$, what lognormal vol $\sigma_{LN}$ reproduces the same option price?

| Approximation | Where it works | Where it breaks |
|---|---|---|
| Crude ATM: $\sigma_n \approx \sigma_{LN} \cdot F$ | Right at the money | Wings |
| Hagan-Kennedy: $\sigma_n \approx \sigma_{LN} \cdot \sqrt{F K}$ | Off-ATM (geometric mean) | Far OTM, large $\sigma\sqrt{T}$ |

For exact conversion, invert numerically (Brent on the difference of the two models' prices).

---

## How to remember

Three sentences, drilled to muscle memory:

1. **"Arithmetic, not geometric — $dF = \sigma_n \, dW$."** $F$ can be anywhere on the real line.
2. **"ATM call $= \sigma_n \sqrt{T/(2\pi)}$, vega $= \sqrt{T/(2\pi)}$, delta $= 0.5$ exactly."** Three ATM closed forms that come up at every interview.
3. **"Bachelier ↔ Black-76 via $\sigma_n \approx \sigma_{LN}\sqrt{FK}$."** The one conversion to remember.

---

## Bonus — limit behaviour and the "name the model" trick

- **$F \to \infty$ with fixed $K$:** Bachelier and Black-76 converge (the lognormal looks normal far from zero).
- **$F$ near zero or negative:** Black-76 fails entirely; Bachelier carries on.

Interview test: "Which model would you use to price a 1y option on a SOFR forward of 75 bps with ATM vol of 80 bps?" Answer: Bachelier (or shifted-lognormal). Forward-to-vol ratio of ~1 means a 1-σ down-move would put $F$ at zero — lognormal dynamics are wildly inappropriate.
