# 4-Week Study Guide — IB & AM Quant Interview Prep

A compressed plan to get from "I have this repo" to "I can answer 80% of likely
interview questions" in **4 weeks at 20 hrs/week (~80 hours total)**.

This is the **interview-prep** plan, not the deep-mastery plan. Trade-offs are
explicit at the end.

---

## TL;DR

| | Default depth | This plan |
|---|---|---|
| Total hours | 200–290 | **70–85** |
| Calendar time | 4–6 months evenings | **3–4 weeks** |
| Outcome | Implementation fluency | Recognition + recall |
| What you can do | Code from scratch on any topic | Code from scratch on T1 core; defend any topic in interview |

Suitable for: first-job / graduate quant interviews, mid-level rotations into
quant from a related field, or pre-screening for senior roles. **Not** suitable
for senior IC role specialisation — that needs 4–6 more weeks on chosen depth
areas.

---

## Time budget — what's in and what's out

### IN (the must-haves)

| Section | Hours | Status |
|---|---|---|
| EDA playbook + decisions wall-poster | 2 | full read |
| Toolkit cheatsheets (pandas, numpy, sklearn, statsmodels) | 3 | bookmark + skim |
| **Quant finance T1** (11 notebooks) | 35–40 | full read, math + code |
| **Quant finance T2 critical** (Heston, BL, Merton, factors) | 10–12 | math + interview Q&A |
| Quant finance T2 remaining (6 notebooks) | 3–4 | markdown only |
| Quant finance T3 (5 notebooks) | 2–3 | concept + Q&A only |
| Drilling + whiteboard rehearsal | 15–20 | active recall |
| **TOTAL** | **70–85 hrs** | |

### OUT (cut entirely)

| Section | Why |
|---|---|
| Toolkit exercises (9 practice projects) | Excellent for fluency, but slow. Cut for time |
| ML pipeline notebooks (classification, regression, time-series) | Not asked in pure quant-finance interviews |
| REST API notebook | Useful for ML-eng, not quant |
| LLM pipeline cheatsheet | Skip unless interviewing at LLM-using fund |
| Toolkit cheatsheet exercises beyond pandas/numpy/sklearn skim | Reference, not study material |

If you have 100+ hours, restore ML pipelines + 3-4 toolkit exercises. If you
have <60 hours, see the "60-hour squeeze" at the end.

---

## Week 1 — Foundations + Options (20 hrs)

Goal: by Friday you can derive Black-Scholes, explain all the Greeks, price
American puts via binomial, and price exotics via Monte Carlo.

| Day | Hours | What | How |
|---|---|---|---|
| **Mon** | 4 | EDA playbook + decisions wall-poster (`toolkit/eda_decisions.md`, `toolkit/eda_playbook.ipynb`); skim toolkit pandas + numpy + sklearn | Run the playbook on `crypto_hourly.parquet`. Bookmark the cheatsheets as references; don't memorise |
| **Tue** | 4 | `quant_finance/01_options/01_black_scholes.ipynb` | Read the math twice. Try to re-derive d₁, d₂ on paper before reading. Run all cells |
| **Wed** | 4 | `quant_finance/01_options/02_bs_family_and_asset_classes.ipynb` | Bachelier, SABR, FX (Garman-Kohlhagen), FI options. Heavy at first — focus on *when each model applies*, not all the math |
| **Thu** | 4 | `quant_finance/01_options/03_greeks.ipynb` | Closed-forms for δ, γ, ν, θ, ρ. Re-derive gamma from delta on paper. Run the gamma-theta P&L simulation |
| **Fri** | 4 | `quant_finance/01_options/04_binomial_trees.ipynb` + `05_monte_carlo_pricing.ipynb` | Implement CRR from scratch. Run the antithetic/control-variate comparisons |

**End-of-week checkpoint** — without notes, can you:
- [ ] Derive the BS PDE in 5 minutes on a whiteboard?
- [ ] State delta, gamma, vega, theta, rho closed forms?
- [ ] Explain why ATM call delta isn't 0.5?
- [ ] Price an American put via CRR backward induction?
- [ ] State Merton's no-early-exercise theorem?
- [ ] Explain when antithetic variates fail?

If 5/6 correct → continue. If <4 → re-do Wed/Thu material.

---

## Week 2 — Risk + Fixed Income + Portfolio (20 hrs)

Goal: by Friday you can price bonds and swaps, hedge with DV01, run mean-variance optimisation, and explain VaR vs ES.

| Day | Hours | What |
|---|---|---|
| **Mon** | 4 | `01_options/06_implied_vol_surface.ipynb` (SVI, arbitrage constraints) + `07_heston.ipynb` (stochastic vol) |
| **Tue** | 4 | `02_risk/01_var_methods.ipynb` + `02_expected_shortfall.ipynb` + `03_credit_merton.ipynb` |
| **Wed** | 4 | `03_fixed_income/01_bond_pricing.ipynb` + `02_duration_convexity_krd.ipynb` |
| **Thu** | 4 | `03_fixed_income/03_curve_building.ipynb` + `04_swaps_swaptions.ipynb` |
| **Fri** | 4 | `04_portfolio/01_markowitz.ipynb` + `02_black_litterman.ipynb` |

**End-of-week checkpoint**:
- [ ] Compute parametric, historical, MC VaR — and explain when each fails?
- [ ] Define ES and explain why it's coherent (and VaR isn't)?
- [ ] Bootstrap a SOFR curve from market quotes?
- [ ] Price a swap and compute its DV01?
- [ ] Derive the Markowitz tangency portfolio formula?
- [ ] State the Black-Litterman master formula?
- [ ] Explain Merton's structural credit model?

If 5/7 correct → continue.

---

## Week 3 — Breadth + ML refresher (20 hrs)

Goal: by Friday you've at least skimmed every notebook in the repo. Recognition-level coverage.

| Day | Hours | What |
|---|---|---|
| **Mon** | 4 | `04_portfolio/04_factor_models.ipynb` (Fama-French, attribution) + `05_performance_attribution.ipynb` |
| **Tue** | 4 | **Skim only** `06_stoch_calc/01_brownian_motion.ipynb` + `02_ito_and_gbm.ipynb`. Read the math + interview Q&A. Don't run code |
| **Wed** | 4 | **Skim only** `05_volatility/01_garch.ipynb` + `02_realized_vol.ipynb`. Concepts + Q&A only |
| **Thu** | 4 | ML pipeline skim: read Stage 0 + main concept of each stage in `classification.ipynb`, `regression.ipynb`, `time_series.ipynb`. **Skip exercises** |
| **Fri** | 4 | T3 skim: `04_portfolio/03_risk_parity.ipynb`, `05_volatility/03_local_vol_dupire.ipynb`, `02_risk/04_cva_intro.ipynb`, `06_stoch_calc/03_lsmc_american.ipynb`. Read interview Q&A on each (1 hr each) |

**End-of-week checkpoint**:
- [ ] State the Fama-French 3-factor model?
- [ ] Apply Itô's lemma to ln(S) for a GBM?
- [ ] Explain what GARCH(1,1) does?
- [ ] State Dupire's formula for local vol?
- [ ] Explain CVA in one sentence?

---

## Week 4 — Drilling + interview rehearsal (20 hrs)

Goal: by Friday everything is recall-level (no peeking) on T1, plus implementation fluency on 5-6 core functions.

| Day | Hours | Drill |
|---|---|---|
| **Mon** | 4 | **Whiteboard re-derive** (no notes): BS PDE; Greek closed forms; put-call parity via static replication; Itô's lemma; CRR up/down/p formulas |
| **Tue** | 4 | **Whiteboard re-derive**: parametric VaR; ES under normal; bond duration; Markowitz tangency; Black-Litterman master formula |
| **Wed** | 4 | **Mock interview Q&A**: pick 5 random questions per T1 notebook (50 questions total). Speak the answer out loud, then check. Re-read where you stumble |
| **Thu** | 4 | **Code from scratch** on a blank file (no library imports beyond `numpy`, `scipy.stats`, `scipy.optimize`): `black_scholes`, `implied_vol`, `bs_delta`/`gamma`/`vega`, `crr_tree`, `mc_european_call`. No peeking |
| **Fri** | 4 | **Pitfalls cards** across every notebook (60+ cards). Read each, identify which you actually internalised. Cover gaps. End-of-day mock interview with a friend / peer |

**End-of-week — final checkpoint**:
- [ ] Can derive BS in 5 min on a whiteboard
- [ ] Can code BS + Greeks + IV solver from scratch in 15 min
- [ ] Can explain N(d₁) vs N(d₂) without notes
- [ ] Can verbalise pitfalls for: VaR, Markowitz, IV surface, swaps
- [ ] Can sketch the structure of any T2 notebook from the title alone

You're ready.

---

## How to read fast — the 25-minute skim

For T2 remaining and T3 notebooks (the "skim only" entries above), use this
fixed protocol:

1. **5 min** — Read "Why this matters" + the 30-second concept
2. **10 min** — Read the math derivation OR the closed-form formula table (whichever is shorter)
3. **5 min** — Read the Interview Q&A *out loud*. If you can't, you don't know it
4. **5 min** — Write 2 flashcards: the headline formula + the most-asked question

Hard rule: **don't run any code, don't do exercises, don't deep-dive**.

By Week 4 you have a 100-200 card deck. Spend 30 min/day reviewing it. Active
recall beats re-reading 10×.

---

## How to read deep — the 90-minute full read

For T1 + T2-critical (the "full read" entries), use this:

1. **15 min** — Why this matters + 30-second concept + math overview
2. **15 min** — Try to derive the math on paper *before* reading the derivation. You won't fully succeed; that's fine. The struggle is the learning
3. **20 min** — Read the implementation cells. Then close them and try to write the function from scratch
4. **30 min** — Do the exercises with hidden solutions. No peeking until you've genuinely tried
5. **5 min** — Read the Interview Q&A out loud
6. **5 min** — Write 3-4 flashcards covering the formula, the key insight, the pitfall

90 minutes per notebook × ~22 must-do notebooks ≈ 33 hours. Matches the budget.

---

## What you give up

Be honest with yourself: this plan **will not** make you fluent in:

| You'll struggle with… | Because we cut… |
|---|---|
| Implementing barrier options with BGK correction | MC + barriers exercises |
| Deriving Heston char-fn step-by-step | Heston deep-dive |
| Calibrating SABR with arbitrage repair | T2 SABR detail |
| Setting up CVA with wrong-way risk simulation | T3 depth |
| Coding GARCH MLE from scratch | T2 detail |
| Implementing a production risk-parity backtest | T3 portfolio depth |
| Vanna-volga FX exotic pricing | Greeks vanna depth |
| Multi-curve OIS-discounted swap MTM with collateral | Curve detail |

For graduate / first-job / mid-level interviews: this is fine. For senior IC
specialisation: extend by 4–6 weeks on your chosen depth areas.

---

## Alternate compressions

### The 60-hour squeeze

Cut Week 3 in half. Specifically:

- Skip ML pipelines entirely (−3 hrs)
- Skim toolkit cheatsheets to bookmarks only (−2 hrs)
- Skim BM + Itô into one 2-hr session (−2 hrs)

Redirect 7 hrs into T1 drilling. **You'll have rock-solid T1 + 4 T2-critical, with recognition on T2/T3 and zero ML refresh.** For most pure-quant interviews this is the right trade.

### The 100-hour version

Add back:
- 3-4 toolkit exercises (pandas, sklearn, optuna projects) — adds 12-16 hrs
- One T2 deep-dive (your favourite topic — Heston / BL / SABR) — adds 4-6 hrs
- ML pipeline classification deep-read — adds 4-6 hrs

Use this if you have ~5 weeks instead of 4.

---

## Daily habits that compound

1. **Morning, 30 min** — flashcard review of yesterday's cards (active recall, no peeking until you commit to an answer)
2. **Evening, 4 hrs** — the day's notebook + new flashcards
3. **Weekend, 1 hr per day** — re-derive a random T1 derivation on a whiteboard

The flashcard review is the highest-leverage activity in the plan. Don't skip
it.

---

## After Week 4 — what's next?

If you have more time before interviews:

1. **Weeks 5–6** — Specialisation: pick 2 favourite areas (e.g. options + portfolio) and go deep. Run all exercises, do the toolkit projects, re-implement everything from scratch.
2. **Weeks 7+** — Build a portfolio project. Take BTC/ETH data, build an end-to-end strategy using factor models + risk parity + ML predictions. This becomes "tell me about a project" content.
3. **Ongoing** — Interview practice with peers, rotating through the Q&A sections.

---

## Reading shortcuts — when in doubt, prioritise

| If you have… | Read in this order |
|---|---|
| 1 hour | EDA decisions wall-poster + BS notebook math section + Greeks Q&A |
| 4 hours | Above + VaR + Markowitz |
| 1 day | Above + Black-Litterman + bond pricing + duration |
| 1 week | All of T1 |
| 2 weeks | All of T1 + T2-critical (Heston, BL, Merton, factors) |
| 4 weeks | This guide |

---

## One last reality check

Interviews are 80% **recall + verbal articulation**, 20% **whiteboard math/code**. The
flashcard work is what makes the verbal articulation effortless. **Skipping it
is the most common mistake.**

Read the Interview Q&A sections out loud. Out loud. Not in your head. The act
of speaking the answer is what makes you fluent on the day.

Good luck.
