# LIBOR Market Model (LMM / BGM) Cheatsheet — One Model, Three Measures

**The point:** each forward LIBOR is lognormal under its **own** $T_{i+1}$-forward measure (so caplet pricing reduces to Black-76 *for free*). To price any product that touches multiple forwards, you must switch them all to a common measure — the drift correction this introduces **is** the entire technical content of the model.

---

## The single specification

For a tenor structure $0 = T_0 < T_1 < \dots < T_N$ with $\delta_i = T_{i+1} - T_i$, define the forward LIBOR:

$$L_i(t) = \frac{1}{\delta_i}\!\left(\frac{P(t, T_i)}{P(t, T_{i+1})} - 1\right)$$

Under its **own** $T_{i+1}$-forward measure $Q^{T_{i+1}}$ (numeraire = $P(\cdot, T_{i+1})$), each $L_i$ is a martingale with lognormal dynamics:

$$\boxed{\;dL_i(t) = \sigma_i(t)\, L_i(t)\, dW_i^{T_{i+1}}(t), \qquad i = 0, 1, \dots, N-1\;}$$

with correlated Brownian motions: $d\langle W_i^{T_{i+1}}, W_j^{T_{j+1}}\rangle_t = \rho_{ij}\, dt$.

**Consequence:** caplet on $L_i$ struck at $K$, paying $\delta_i \max(L_i(T_i) - K, 0)$ at $T_{i+1}$:

$$\text{Caplet}_i = \delta_i \cdot P(0, T_{i+1}) \cdot \text{Black-76}\big(L_i(0), K, T_i, \sigma_i\big)$$

The model is **market-consistent by construction** — calibration to the cap-vol surface is just Black-76 inversion per tenor.

---

## Pick a measure, get a drift

To price a multi-forward product (swaption, Bermudan, TARN), all $L_i$'s must be evolved under the **same** measure. Three standard choices:

| Measure | Numeraire | Drift of $L_i$ | When to use |
|---|---|---|---|
| **Natural** $Q^{T_{i+1}}$ | $P(\cdot, T_{i+1})$ | $0$ (martingale) | Caplet pricing — gives Black-76 directly |
| **Terminal** $Q^{T_N}$ | $P(\cdot, T_N)$ | $-\sigma_i \sum_{j=i+1}^{N-1} \dfrac{\delta_j \rho_{ij}\sigma_j L_j}{1 + \delta_j L_j}$ | European products; cleanest formula |
| **Spot LIBOR** $Q^B$ | rolling bank account | $+\sigma_i \sum_{j=\eta(t)+1}^{i} \dfrac{\delta_j \rho_{ij}\sigma_j L_j}{1 + \delta_j L_j}$ | Bermudan / callable products |

Drift sign / sum direction is **derived from Girsanov + numeraire ratio**, not memorised. The pattern:

- **Terminal**: sum runs **forward** ($j = i+1, \dots, N-1$). Drift is **negative** (rates ↑ ⇒ numeraire $P(\cdot, T_N)$ ↓). $L_{N-1}$ is a martingale (its natural measure equals the terminal).
- **Spot**: sum runs **backward** ($j = \eta+1, \dots, i$). Drift is **positive**. The first-alive forward $L_{\eta}$ is a martingale.

---

## What each piece means

- **$\sigma_i(t)$** — instantaneous vol of forward $i$. Calibrated from caps via Black-76 inversion (bootstrapping per tenor).
- **$\rho_{ij}$** — correlation between forwards. Parameterised, e.g. Rebonato: $\rho_{ij} = \rho_\infty + (1 - \rho_\infty)\exp(-\beta|T_i - T_j|)$.
- **Number of factors** — full rank is $N$, but PCA shows 2-3 factors capture >99% of variance. Production reduces $\rho$ to rank 2-3 for speed.
- **Drift terms** — bookkeeping for change of measure, NOT new market information. They're what make the model self-consistent.

---

## How to remember

Three sentences, drilled to muscle memory:

1. **"Each forward is lognormal under its own measure — caplets are Black-76 for free."** $dL_i = \sigma_i L_i \, dW_i^{T_{i+1}}$. This is the model.
2. **"Switching measures introduces a drift via Girsanov."** Don't memorise the formula — derive it from the numeraire ratio. Sign and index range follow.
3. **"Terminal sums forward, spot sums backward."** Pattern of the drift sum. Plus: terminal-measure $L_{N-1}$ has zero drift; spot-measure $L_\eta$ has zero drift.

---

## Bonus — the bond-from-forwards identity

Everything in LMM rests on:

$$P(t, T_n) = P(t, T_0) \prod_{i=0}^{n-1} \frac{1}{1 + \delta_i L_i(t)}$$

Used in: change-of-numeraire computations, deriving the drift, computing discount factors path-by-path in the simulator, and converting between LMM forwards and the bond curve.

---

## What NOT to memorise (reach-for material)

- Full drift formulas under each measure (derive from Girsanov)
- Predictor-corrector simulation scheme
- Cap-vol bootstrap algebra
- Rebonato correlation parameterisation in detail
- Longstaff-Schwartz basis-function choice for Bermudan LSMC
- Shifted-LMM modification for negative rates
- Rank reduction via PCA

All of the above live in `05_libor_market_model.ipynb` — recognise the names, look up the implementation when needed.
