# RiskPlus Streamlit MVP - Phase 2: Risk Budgeting & Factor Contribution

## What's Implemented

### Four New Report Tabs

1. **RISK BUDGETING (ETL)**
   - Compares each asset's actual/expected return to its *implied return*
   - Implied Return: the return an asset must deliver to justify its contribution to portfolio ETL
   - Formula: `implied_return_i = (portfolio_excess_return / portfolio_etl) * marginal_contribution_etl_i + rf_rate`
   - Status column indicates whether actual return exceeds implied return (↑ Increase weight, ↓ Decrease weight, → Neutral)
   - Interactive scatter chart: Mean Return vs. Marginal ETL Contribution with implied return reference line

2. **RISK BUDGETING (StDev)**
   - Similar to ETL version but based on portfolio standard deviation
   - Helps identify assets whose volatility contribution is justified by their expected returns
   - Same chart and table layout as ETL tab

3. **FACTOR CONTRIBUTION**
   - Decomposes portfolio risk into systematic (factor-explained) and specific (idiosyncratic) components
   - Shows percentage contribution to StDev and ETL by:
     - Systematic vs. Specific Risk (high-level split)
     - Factor bucket aggregation (Equity Risk, Commodity Risk, FX Risk, etc.)
     - Individual factor exposures
   - Automatic factor-to-bucket mapping using keyword rules (e.g., "S&P 500" → Equity Risk)
   - Horizontal bar charts for visual risk decomposition

4. **EXPOSURE BY FACTOR BUCKET**
   - Shows portfolio sensitivity to each market segment (factor bucket)
   - Weighted exposure = sum of (asset weight × asset beta for each factor in bucket)
   - Displays net exposure to Equity Risk, Commodity Risk, FX Risk, Interest Rate Risk, Sovereign Risk, Corporate Risk, Volatility Risk, Fixed Income Risk
   - Useful for understanding portfolio positioning and diversification across risk dimensions

---

## Methodology Notes

### Implied Returns (MVP Approximation)

**RiskPlus approach (proprietary):**
- Uses STARR (Sharpe Tail Risk Adjusted Return) optimal portfolio framework
- Solves a constrained optimization problem to derive optimal allocation and implied returns

**RiskPlus Streamlit MVP approach (practical approximation):**
```
performance_ratio = (portfolio_mean - rf_rate) / portfolio_risk_measure
implied_return_i = performance_ratio × marginal_contribution_i + rf_rate
```

This simple ratio-based approach is transparent and interpretable:
- If an asset's expected return > implied return, it may justify higher allocation (better risk-adjusted return)
- If expected return < implied return, it may justify lower allocation (insufficient return for its risk)
- Simpler than proprietary RiskPlus but economically sound and useful for portfolio optimization discussions

**Important caveat:** This is not the exact RiskPlus methodology but a practical approximation suitable for analytical prototyping and portfolio discussions. For official risk reporting, validate against actual RiskPlus software.

---

### Factor Contribution Decomposition

**Systematic Risk:**
- Calculated as: `sqrt(beta_vector.T @ factor_cov @ beta_vector)`
- Represents portfolio volatility explained by factor model
- Marginal contribution computed from factor sensitivity and covariance

**Specific (Idiosyncratic) Risk:**
- Calculated as: `sqrt(residual_variance from OLS)`
- Represents unexplained portfolio volatility
- Not attributable to the selected factor model

**Allocation to ETL:**
- Uses approximation: scale factor's StDev contribution by ratio of simulated portfolio StDev to estimated tail loss
- More sophisticated approximation than simple ratio, but still MVP-level
- Not matching proprietary RiskPlus ETL factor decomposition

---

### Factor-to-Bucket Mapping

Automatic keyword-based inference:

| Bucket | Example Keywords |
|--------|------------------|
| Equity Risk | MSCI, Russell, S&P, equity, value, growth, small, large, EAFE, EM |
| Commodity Risk | GSCI, crude, oil, commodity, gold, copper, agriculture, metals |
| FX Risk | dollar, euro, yen, sterling, pound, franc, FX, exchange |
| Interest Rate Risk | rate, treasury, LIBOR, SOFR, yield, spread, maturity, duration |
| Sovereign Risk | sovereign, government bond, treasury master |
| Corporate Risk | corporate, high yield, credit, BBB, AAA |
| Volatility Risk | VIX, CBOE, volatility |
| Fixed Income Risk | bond, mortgage, ABS, MBS, CMBS, fixed income |

Users can override mappings in future enhancements by providing uploaded metadata.

---

### Exposure Calculation

**Weighted Portfolio Exposure (default):**
```
exposure_to_bucket_i = sum_over_assets(weight_asset × beta_asset_to_factors_in_bucket)
```

**Standalone Beta Exposure (alternative toggle):**
```
exposure_to_bucket_i = beta_asset_to_factors_in_bucket (weight = 1 conceptually)
```

---

## Limitations vs. Proprietary RiskPlus

| Feature | RiskPlus | RiskPlus Streamlit MVP | Status |
|---------|----------|----------------------|--------|
| Skewed Student-t Copula | ✓ | Student-t with Gaussian copula | MVP Approximation |
| STARR Optimal Portfolio | ✓ | Simple performance ratio | MVP Approximation |
| Stepwise AIC Factor Selection | ✓ | User selects factors manually | Placeholder |
| Multi-asset/fund analytics | ✓ | Single portfolio (extensible) | MVP Scope |
| Stress Testing | ✓ | Not implemented | Phase 3+ |
| Scenario Analysis | ✓ | Not implemented | Phase 3+ |
| live Factor Feeds | ✓ | User-uploaded data only | Not in scope |
| Regulatory Reporting Compliance | ✓ | Not validated | Not intended |

---

## Backend Functions Added

### Risk Budgeting

```python
compute_implied_returns(portfolio_mean, portfolio_risk, marginal_contributions, rf_rate, af)
```
Returns array of implied returns per asset based on risk contribution and portfolio performance ratio.

```python
compute_risk_budgeting_table(weights, returns_matrix, marginal_contribs, risk_measure, rf_rate, af)
```
Returns DataFrame with Weight, Mean Return, MC, Implied Return, Difference, Status for all assets.

### Factor Contribution

```python
compute_factor_contribution(ols_model, factor_returns, portfolio_returns, simulated_portfolio_returns, confidence)
```
Returns dict with systematic/specific risk split, factor-level marginal contributions, and bucket mapping.

```python
get_factor_bucket_mapping(factor_names)
```
Infers bucket membership from factor names using keyword rules.

```python
compute_factor_bucket_exposures(weights, betas_matrix, bucket_mapping)
```
Computes weighted portfolio exposure to each factor bucket.
```python
compute_factor_bucket_exposures(betas, bucket_mapping)
```
Aggregates fitted OLS factor betas into factor buckets (e.g., Equity Risk, Commodity Risk).
---

## UI/UX Enhancements

- **Four new tabs** after SIMULATED RISK
- **Sidebar settings:**
  - Toggle factor bucket analysis on/off
  - Select "Weighted portfolio exposure" vs. "Standalone beta" for EXPOSURE BY FACTOR BUCKET
- **RiskPlus-inspired formatting:**
  - Dense worksheet-style tables
  - Descriptive captions above each tab
  - Status indicators (↑↓→ arrows) for risk budgeting guidance
  - Interactive Plotly charts with hover details
- **Color/styling:**
  - Charts use professional blue/gray palette
  - Charts include reference lines and dual-series (actual vs. implied)

---

## Excel Export (Future Enhancement)

When Excel export is enhanced for Phase 2, the following sheets will be added:
- RISK BUDGETING (ETL)
- RISK BUDGETING (StDev)
- FACTOR CONTRIBUTION
- EXPOSURE BY FACTOR BUCKET

Each sheet will include:
- Dark blue title band with subtitle
- Condensed worksheet tables
- Bold portfolio and strategy rows
- Conditional formatting (green/yellow/red for Difference columns)
- Percent/bps number formats
- Frozen panes for scrolling

---

## Intended Use Cases

1. **Portfolio Optimization:** Identify which holdings are earning sufficient return for their risk contribution
2. **Risk Attribution:** Understand which factors and buckets drive portfolio risk
3. **Diversification Analysis:** Confirm portfolio exposure across major risk dimensions
4. **Investor Communication:** Explain portfolio construction to clients using risk-based metrics
5. **Analytical Prototyping:** Test portfolio changes in a lightweight, interpretable tool before running formal RiskPlus

---

## Caveats for Users

- **MVP Quality:** These calculations are practical approximations intended for analytical insight, not official risk reporting
- **Not Proprietary RiskPlus:** Results may differ from official BISAM RiskPlus software
- **Single Portfolio:** Current app supports one uploaded portfolio; multi-asset/fund-level analysis is a future enhancement
- **Factor Model Reliance:** All systematic risk calculations depend on the fitted OLS factor model; results only as good as factor selection and model fit
- **Simulation Variance:** Risk budgeting and factor contribution metrics depend on Monte Carlo simulations; rerun for updated outputs
- **Data Quality:** Garbage in, garbage out; ensure uploaded data is clean, aligned in time, and free of errors

---

## Next Steps (Phase 3+)

- [ ] Multi-asset/fund-level analytics (asset weights in upload)
- [ ] Strategy/group-level subtotals and hierarchical reporting
- [ ] Stress testing and scenario analysis tabs
- [ ] Factor bucket metadata upload and override UI
- [ ] Robust covariance matrix estimation (Ledoit-Wolf shrinkage)
- [ ] Dynamic factor selection / stepwise AIC
- [ ] Live factor price feeds (Yahoo Finance, FRED, Bloomberg-style connector)
- [ ] RiskPlus-formatted Excel export with conditional formatting
- [ ] PDF report generation
- [ ] Backtesting framework to compare RiskPlus Streamlit vs. actual BISAM outputs
