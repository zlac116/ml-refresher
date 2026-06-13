# Black-Scholes Cheatsheet — One Formula, Four Cases

**The point:** the BS price is the *cost of running a delta-hedge* — not a prediction of the underlying. Once you see this, all four "variants" (spot call, spot put, forward call, forward put — and FX, futures, etc.) collapse to **one formula** parameterised by the forward `F`.

---

## The unified form

For any European vanilla, define the forward of the underlying:

$$F = S \cdot e^{(r - q)T}$$

(For options *quoted on a forward* — futures, swaps, FX, commodities — `F` is observed directly. The conversion isn't needed.)

Then:

$$d_1 = \frac{\ln(F/K) + \tfrac{1}{2}\sigma^2 T}{\sigma\sqrt{T}}, \qquad d_2 = d_1 - \sigma\sqrt{T}$$

$$\boxed{\;\text{Call} = e^{-rT} \big[F \, N(d_1) - K \, N(d_2)\big]\;}$$

$$\boxed{\;\text{Put} = e^{-rT} \big[K \, N(-d_2) - F \, N(-d_1)\big]\;}$$

**That's it. Four cases, one formula.**

---

## Why the four "different" formulas are the same

| Quoted on | Stock leg | Cash leg | Comment |
|---|---|---|---|
| **Spot, BS-Merton** | $S \cdot e^{-qT} \cdot N(d_1)$ | $K \cdot e^{-rT} \cdot N(d_2)$ | Substitute $F = S \cdot e^{(r-q)T}$ and factor out $e^{-rT}$ → identical |
| **Spot, BS (no dividend)** | $S \cdot N(d_1)$ | $K \cdot e^{-rT} \cdot N(d_2)$ | Same with $q = 0$ |
| **Forward, Black-76** | $F \cdot e^{-rT} \cdot N(d_1)$ | $K \cdot e^{-rT} \cdot N(d_2)$ | Already in this form |
| **FX, Garman-Kohlhagen** | $S \cdot e^{-r_f T} \cdot N(d_1)$ | $K \cdot e^{-r_d T} \cdot N(d_2)$ | $F = S \cdot e^{(r_d - r_f)T}$, same identity |

The common form: **forward leg − strike leg, both discounted at the domestic rate `r`, each weighted by its measure's exercise probability.**

---

## The hedging interpretation, in this unified form

For the **call**, your hedge today consists of:

- $N(d_1)$ units of forward exposure (long)
- Short cash equal to $K \cdot e^{-rT} \cdot N(d_2)$ — the PV of what you'll owe if exercised

For the **put**, mirror image:

- $N(-d_1)$ units of forward exposure (short)
- Long cash equal to $K \cdot e^{-rT} \cdot N(-d_2)$

$N(d_1)$ is your **hedge ratio**; $N(d_2)$ is the **risk-neutral exercise probability**. They're related but live in different measures.

---

## How to remember

Three sentences, drilled to muscle memory:

1. **"Convert spot to forward."** $F = S \cdot e^{(r-q)T}$. If quoted on a forward, skip this step.
2. **"d1, then d2 = d1 − σ√T."** $d_1$ has the $+\tfrac{1}{2}\sigma^2$ drift inside; $d_2$ is just $d_1$ shifted by one vol-time.
3. **"Call is forward minus strike, both discounted, each weighted by N(d). Put is the same with negative d's, sign-flipped."**

If you can compute $F$ and $d_1$ for any product, the rest is one division ($\sigma\sqrt{T}$) and four $N(\cdot)$ lookups. **Memorise one formula, derive four.**

---

## Bonus — put-call parity falls out for free

From the unified form:

$$\text{Call} - \text{Put} = e^{-rT}(F - K)$$

Substituting $F = S \cdot e^{(r-q)T}$ recovers the standard parity statement:

$$C - P = S \cdot e^{-qT} - K \cdot e^{-rT}$$

If asked in interview to verify parity, derive it on the unified form in one line.
