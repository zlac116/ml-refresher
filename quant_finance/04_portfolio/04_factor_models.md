# Factor Models — CAPM, Fama-French, Carhart

## Why this matters

A stock's return isn't all "alpha" — most of it comes from **systematic exposures** to common risk factors. Decomposing returns into factor exposures + residual alpha is the single most useful AM analytical workflow.

Three landmark models:

| Model | Year | Factors |
|---|---|---|
| **CAPM** | 1964 | Market |
| **Fama-French 3F** | 1993 | Market, Size (SMB), Value (HML) |
| **Carhart 4F** | 1997 | + Momentum (UMD/MOM) |
| **Fama-French 5F** | 2015 | + Profitability (RMW), Investment (CMA) |

You will be asked, in any AM interview:
1. State CAPM. What's β?
2. **Run a factor regression** — what does the alpha mean?
3. Why was the size and value premium added in FF93?
4. **Performance attribution** — explain a portfolio's return as factor × exposure + alpha.
5. Multi-factor risk model — sources of risk decomposed by factor.
6. Why is alpha so hard to find post-2010? (Factor crowding.)

This notebook covers all six on real Fama-French daily data (Ken French's library) regressed against AAPL stock returns.

## The math — CAPM

Sharpe-Lintner CAPM (1964):

$$\mathbb{E}[R_i] - r_f = \beta_i \cdot (\mathbb{E}[R_M] - r_f)$$

$$\beta_i = \frac{\text{Cov}(R_i, R_M)}{\text{Var}(R_M)}$$

In words: a stock's excess return is proportional to the market's excess return, with proportionality constant β. Empirically: regress stock excess returns on market excess returns. Slope = β. Intercept = "α", the unexplained return.

## Fama-French 3-Factor (1993)

CAPM was empirically inadequate — small stocks and "value" stocks (high book-to-market) consistently outperformed CAPM predictions. Fama-French fixed this by adding two factors:

$$R_i - r_f = \alpha + \beta_M (R_M - r_f) + \beta_{SMB} \cdot SMB + \beta_{HML} \cdot HML + \epsilon$$

- **SMB** (Small Minus Big): return on small-cap minus big-cap. Captures the size premium.
- **HML** (High Minus Low book-to-market): return on value minus growth stocks. Captures the value premium.

## Carhart (1997)

Add momentum:

$$+ \beta_{MOM} \cdot MOM$$

- **MOM**: long winners, short losers (12-month look-back, skip last month).

## Fama-French 5-Factor (2015)

Add profitability and investment:

$$+ \beta_{RMW} \cdot RMW + \beta_{CMA} \cdot CMA$$

- **RMW**: Robust Minus Weak — high-profitability stocks vs low.
- **CMA**: Conservative Minus Aggressive — low investment vs high.

## Setup


```python
import warnings; warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas_datareader.data as web
import yfinance as yf

# Pull Ken French's daily 3-factor + momentum factors (2024-2026)
ff3   = web.DataReader('F-F_Research_Data_Factors_daily',         'famafrench', start='2024-01-01', end='2026-04-30')[0]
ff_mom = web.DataReader('F-F_Momentum_Factor_daily',               'famafrench', start='2024-01-01', end='2026-04-30')[0]

# All in percent — convert to decimal
ff3 = ff3 / 100
ff_mom = ff_mom / 100
ff_mom.columns = ['MOM']
factors = pd.concat([ff3, ff_mom], axis=1).dropna()

print(f'Factor data: {factors.shape[0]} days from {factors.index[0].date()} to {factors.index[-1].date()}')
print(f'Columns: {factors.columns.tolist()}')
print(factors.tail())
```

    Factor data: 541 days from 2024-01-02 to 2026-02-27
    Columns: ['Mkt-RF', 'SMB', 'HML', 'RF', 'MOM']
                Mkt-RF     SMB     HML      RF     MOM
    Date                                              
    2026-02-23 -0.0118 -0.0030 -0.0131  0.0001  0.0135
    2026-02-24  0.0083  0.0051 -0.0066  0.0001  0.0033
    2026-02-25  0.0079 -0.0037  0.0049  0.0001  0.0046
    2026-02-26 -0.0047  0.0063  0.0032  0.0001 -0.0193
    2026-02-27 -0.0051 -0.0044 -0.0125  0.0001 -0.0104



```python
# AAPL daily returns from yfinance
aapl = yf.Ticker('AAPL').history(start='2024-01-01', end='2026-04-30')[['Close']]
aapl = aapl.assign(ret=lambda d: np.log(d['Close']).diff()).dropna()
aapl.index = aapl.index.tz_localize(None).normalize()
factors.index = pd.to_datetime(factors.index).normalize()

# Align dates
common = factors.index.intersection(aapl.index)
factors_a = factors.loc[common]
aapl_a    = aapl.loc[common]

# Excess returns: AAPL - RF
excess_aapl = aapl_a['ret'] - factors_a['RF']
excess_aapl.name = 'AAPL_excess'

print(f'\nMerged dataset: {len(common)} days')
print(f'AAPL annualised excess return: {excess_aapl.mean()*252:.3%}')
print(f'AAPL annualised vol:           {excess_aapl.std()*np.sqrt(252):.3%}')
```

    
    Merged dataset: 540 days
    AAPL annualised excess return: 11.996%
    AAPL annualised vol:           27.697%


## CAPM regression


```python
# CAPM: AAPL_excess = α + β * (Mkt - RF) + ε
X = sm.add_constant(factors_a['Mkt-RF'])
y = excess_aapl
model_capm = sm.OLS(y, X).fit()

print('CAPM regression:')
print(model_capm.summary().tables[1])
print(f'\nR² = {model_capm.rsquared:.4f}')
print(f'Annualised α: {model_capm.params["const"]*252:.3%}')
print(f'Annualised α (t-stat): {model_capm.tvalues["const"]:.2f}')
print(f'Market β: {model_capm.params["Mkt-RF"]:.4f}')
```

    CAPM regression:
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const         -0.0002      0.001     -0.343      0.732      -0.001       0.001
    Mkt-RF         1.1121      0.056     20.002      0.000       1.003       1.221
    ==============================================================================
    
    R² = 0.4265
    Annualised α: -4.923%
    Annualised α (t-stat): -0.34
    Market β: 1.1121


## Fama-French 3-Factor regression


```python
# FF3: AAPL_excess = α + β_M (Mkt - RF) + β_SMB SMB + β_HML HML + ε
X = sm.add_constant(factors_a[['Mkt-RF', 'SMB', 'HML']])
model_ff3 = sm.OLS(y, X).fit()

print('Fama-French 3-Factor:')
print(model_ff3.summary().tables[1])
print(f'\nR² = {model_ff3.rsquared:.4f}  (vs CAPM R² = {model_capm.rsquared:.4f})')
print(f'Annualised α: {model_ff3.params["const"]*252:.3%}')

# Compare
comparison = pd.DataFrame({
    'CAPM': [model_capm.params['const']*252, model_capm.params['Mkt-RF'], np.nan, np.nan, np.nan, model_capm.rsquared],
    'FF3':  [model_ff3.params['const']*252, model_ff3.params['Mkt-RF'], model_ff3.params['SMB'], model_ff3.params['HML'], np.nan, model_ff3.rsquared],
}, index=['α (annualised)', 'β_Mkt', 'β_SMB', 'β_HML', 'β_MOM', 'R²']).round(4)
print('\nComparison:')
print(comparison.to_string())
```

    Fama-French 3-Factor:
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const         -0.0003      0.001     -0.514      0.608      -0.001       0.001
    Mkt-RF         1.1699      0.063     18.690      0.000       1.047       1.293
    SMB           -0.2511      0.086     -2.924      0.004      -0.420      -0.082
    HML           -0.0099      0.082     -0.121      0.903      -0.171       0.151
    ==============================================================================
    
    R² = 0.4359  (vs CAPM R² = 0.4265)
    Annualised α: -7.351%
    
    Comparison:
                      CAPM     FF3
    α (annualised) -0.0492 -0.0735
    β_Mkt           1.1121  1.1699
    β_SMB              NaN -0.2511
    β_HML              NaN -0.0099
    β_MOM              NaN     NaN
    R²              0.4265  0.4359


## Carhart 4-Factor (+ momentum)


```python
# Carhart 4F: + MOM factor
X = sm.add_constant(factors_a[['Mkt-RF', 'SMB', 'HML', 'MOM']])
model_c4 = sm.OLS(y, X).fit()

print('Carhart 4-Factor:')
print(model_c4.summary().tables[1])
print(f'\nR² = {model_c4.rsquared:.4f}')

# Update comparison
comparison['Carhart 4F'] = [model_c4.params['const']*252, model_c4.params['Mkt-RF'],
                             model_c4.params['SMB'], model_c4.params['HML'], model_c4.params['MOM'],
                             model_c4.rsquared]
print('\nFinal comparison:')
print(comparison.round(4).to_string())
print('\n→ As we add factors, the unexplained α typically shrinks. R² rises modestly.')
print('→ AAPL\'s β_Mkt > 1 (more sensitive than market). β_SMB negative (it\'s a mega-cap).')
```

    Carhart 4-Factor:
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const      -8.091e-05      0.001     -0.147      0.883      -0.001       0.001
    Mkt-RF         1.2283      0.061     20.129      0.000       1.108       1.348
    SMB           -0.3942      0.086     -4.599      0.000      -0.563      -0.226
    HML           -0.1686      0.083     -2.041      0.042      -0.331      -0.006
    MOM           -0.4343      0.067     -6.455      0.000      -0.566      -0.302
    ==============================================================================
    
    R² = 0.4767
    
    Final comparison:
                      CAPM     FF3  Carhart 4F
    α (annualised) -0.0492 -0.0735     -0.0204
    β_Mkt           1.1121  1.1699      1.2283
    β_SMB              NaN -0.2511     -0.3942
    β_HML              NaN -0.0099     -0.1686
    β_MOM              NaN     NaN     -0.4343
    R²              0.4265  0.4359      0.4767
    
    → As we add factors, the unexplained α typically shrinks. R² rises modestly.
    → AAPL's β_Mkt > 1 (more sensitive than market). β_SMB negative (it's a mega-cap).


## Performance attribution

Decompose the portfolio's total return into:
- **Factor contributions**: $\beta_k \times \bar F_k$ for each factor $k$
- **Residual alpha**: $\bar \alpha$
- **Idiosyncratic noise**: $\bar \epsilon$ (zero by construction in expectation)

For AAPL over the sample, attribute the realised return.


```python
# Daily factor contribution = beta * factor_return
attribution = pd.Series({
    'α':       model_c4.params['const'] * 252,
    'Market':  model_c4.params['Mkt-RF']  * factors_a['Mkt-RF'].mean() * 252,
    'SMB':     model_c4.params['SMB']     * factors_a['SMB'].mean() * 252,
    'HML':     model_c4.params['HML']     * factors_a['HML'].mean() * 252,
    'MOM':     model_c4.params['MOM']     * factors_a['MOM'].mean() * 252,
})
attribution['Total (sum)'] = attribution.sum()
attribution['Realized (mean × 252)'] = excess_aapl.mean() * 252

print('Annualised return attribution:')
print(attribution.round(4).to_string())
print()
print('→ Most of AAPL\'s excess return comes from market + size/style exposures.')
print('→ The "α" line is what\'s left unexplained.')
```

    Annualised return attribution:
    α                       -0.0204
    Market                   0.1869
    SMB                      0.0247
    HML                     -0.0039
    MOM                     -0.0673
    Total (sum)              0.1200
    Realized (mean × 252)    0.1200
    
    → Most of AAPL's excess return comes from market + size/style exposures.
    → The "α" line is what's left unexplained.


## Why factor crowding ate alpha post-2010

Pre-2000s: factor strategies were rare. Discovering "value works" was real alpha. Post-2010: every quant fund runs Fama-French-Carhart-style factor strategies. So the factor *premia* themselves get arbitraged away — the SMB and HML returns since 2010 have been **near zero or negative**. Genuine alpha is harder. Modern AM looks for:
- **New factors** (quality, low-vol, ESG)
- **Non-linear exposures** (machine learning on factor inputs)
- **Alternative data** (satellite, credit-card, social) — where most don't have access yet
- **Capacity-limited niches** (small-cap, EM, cross-asset)

## Exercises

### Exercise 1 — Run CAPM on a different stock

Pull META (2024-2026) and run CAPM regression. Compare β to AAPL.


```python
# Your answer here

```

<details>
<summary><b>Reveal solution</b></summary>

```python
meta = yf.Ticker('META').history(start='2024-01-01', end='2026-04-30')[['Close']]
meta = meta.assign(ret=lambda d: np.log(d['Close']).diff()).dropna()
meta.index = meta.index.tz_localize(None).normalize()

common_meta = factors.index.intersection(meta.index)
y_meta = meta.loc[common_meta, 'ret'] - factors.loc[common_meta, 'RF']

X = sm.add_constant(factors.loc[common_meta, 'Mkt-RF'])
model_meta = sm.OLS(y_meta, X).fit()
print(f'META CAPM β: {model_meta.params["Mkt-RF"]:.4f}')
print(f'AAPL CAPM β: {model_capm.params["Mkt-RF"]:.4f}')
```

_META typically has higher β than AAPL — more market-sensitive._

</details>

### Exercise 2 — Significance of factor exposures

Look at FF3 t-stats for AAPL. Which factor exposures are statistically significant at 5%?


```python
# Your answer here

```

<details>
<summary><b>Reveal solution</b></summary>

```python
sig_factors = []
for factor in ['Mkt-RF', 'SMB', 'HML']:
    t = model_ff3.tvalues[factor]
    p = model_ff3.pvalues[factor]
    sig = '✓' if p < 0.05 else '✗'
    sig_factors.append([factor, model_ff3.params[factor], t, p, sig])
    print(f'{factor:>8}: β = {model_ff3.params[factor]:+.4f}, t = {t:+.2f}, p = {p:.4f}  {sig}')
print()
print('→ Significant factors define the stock\'s risk profile.')
```

_Mkt-RF nearly always significant. SMB/HML/MOM depend on the stock._

</details>

### Exercise 3 — Information ratio

Compute AAPL's information ratio (IR) = α / SE(α) for each model. Higher IR = more reliable alpha.


```python
# Your answer here

```

<details>
<summary><b>Reveal solution</b></summary>

```python
ir_data = []
for name, m in [('CAPM', model_capm), ('FF3', model_ff3), ('Carhart 4F', model_c4)]:
    alpha_ann = m.params['const'] * 252
    se_alpha = m.bse['const'] * np.sqrt(252)
    ir = alpha_ann / se_alpha
    ir_data.append([name, alpha_ann, se_alpha, ir])

print(f'{"Model":>10}  {"α":>10}  {"SE(α)":>10}  {"IR":>10}')
for row in ir_data:
    print(f'{row[0]:>10}  {row[1]:>+10.4f}  {row[2]:>10.4f}  {row[3]:>+10.2f}')
print('\n→ |IR| > 0.5 is considered "real" alpha. Below that, hard to distinguish from noise.')
```

_AAPL IR varies; for individual stocks rarely > 1.0._

</details>

## Interview Q&A

**Q: State CAPM.**

A: $\mathbb{E}[R_i] - r_f = \beta_i (\mathbb{E}[R_M] - r_f)$, where $\beta_i = \text{Cov}(R_i, R_M)/\text{Var}(R_M)$. Empirically: regress stock excess returns on market excess returns. Slope = β. Intercept = α, unexplained.

**Q: Why is FF3 better than CAPM?**

A: Empirically, small stocks and value (high book-to-market) stocks **consistently outperform** what CAPM predicts. FF93 added size (SMB) and value (HML) factors to capture these well-documented anomalies. R² jumps from ~30-50% to ~70-85% on individual stocks.

**Q: SMB negative — what does that mean?**

A: A negative β_SMB means the stock behaves like a **large-cap** (loads negatively on the small-minus-big factor). Mega-caps like AAPL, MSFT, NVDA all have negative β_SMB.

**Q: What's a positive HML loading?**

A: Loading on the value factor — the stock behaves like a **value stock** (high book-to-market). Consistent with mature, slow-growth firms with high tangible book. Tech stocks generally have negative HML (growth tilt).

**Q: Information ratio — what's a good number?**

A: IR = α / SE(α). Annualised. IR > 0.5 is "real". IR > 1 is "exceptional". For mutual funds, the population median is near 0. Hedge funds report higher IRs but often inflate via leverage, illiquidity premia, or selection bias.

**Q: How does factor regression distinguish skill from luck?**

A: The t-stat on α. If t > 2 (one-sided), the alpha is statistically significant at 5%. **But**: with thousands of stocks, multiple-testing means some "significant" alphas are flukes. Apply False Discovery Rate (FDR) corrections (Benjamini-Hochberg) for cross-sectional studies.

**Q: Why has alpha decayed since the 1990s?**

A: **Factor crowding**. Once everyone runs the same model and trades the same factors, the premia get arbitraged. Original Fama-French (1993): SMB returned ~3%/yr historically. Post-2010: ~0%. Same for value (HML). Smart-beta ETFs (~$1T AUM) accelerate the crowding. Fund managers have moved to: ML-augmented factors, alternative data, tactical timing of factors, exotic premia (vol, carry, illiquidity).

**Q: Multi-factor risk vs return models — different things?**

A: Yes. **Risk models** (e.g., Barra) focus on $\Sigma$ — explaining covariance via shared factor exposures. **Return models** (Fama-French, Carhart) focus on expected return premia. Practitioners often use both: risk model for portfolio construction (covariance), return model for return forecasting (expected returns).

**Q: Adjusted R² and overfitting**

A: Adjusted R² penalises extra factors that don't add explanatory power. With many factors and short samples, naïve R² inflates. Always report adjusted R² and cross-validate (out-of-sample) before trusting a factor model.

## Pitfalls reference card

| Pitfall | Issue | Fix |
|---|---|---|
| Reporting raw α without t-stat | Single number doesn't show significance | Always report α / SE(α) |
| Long-only proxy regressions | LO portfolios load on Mkt + size + value automatically | Use long-short factors (SMB, HML) which are zero-cost |
| Time-varying betas | Sample-period β isn't predictive | Use rolling β or condition on regime (recession indicator) |
| Heteroscedastic / autocorrelated errors | OLS SE wrong | Newey-West for time-series, White for heterosk |
| Survivorship bias in factor data | Ken French data already corrected; CRSP raw isn't | Use Ken French's curated factors |
| Multi-collinear factors | SMB and Value sometimes correlate | Check VIF; orthogonalise if needed |
| In-sample fitting | Significant α in-sample isn't predictive OOS | Walk-forward / cross-validation |
| Wrong factor universe for the asset | US factors on EM stocks → spurious | Use region-appropriate factors (Asia ex-Japan etc.) |

## What you've earned

After this notebook you can:

1. **State and run** CAPM, FF3, Carhart 4F regressions on real data.
2. **Interpret** factor loadings (β_Mkt > 1, β_SMB < 0, etc.) for an individual stock.
3. **Decompose** returns into factor contributions + alpha + noise.
4. **Compute** information ratio and interpret as skill measure.
5. **Defend** factor model choices in interview — why FF3 over CAPM, why factor crowding eats alpha, when to use Carhart's momentum factor.

This completes the **interview-critical T2** notebook batch (Heston, BL, Merton, factor models). Together with T1, that's the full IB/AM quant interview foundation.
