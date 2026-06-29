# Post-Refactor Stabilization Audit

## Current Module Responsibilities

- `app.py`: Streamlit UI orchestration, sidebar inputs, tab rendering, and display calls.
- `riskplus_core/engine.py`: reusable analysis pipeline that runs the current single-portfolio workflow.
- `riskplus_core/data.py`: file loading, validation, stream preparation, and compatibility wrappers.
- `riskplus_core/models.py`: dataclasses for structured results and metadata.
- `riskplus_core/reporting.py`: pure DataFrame and Plotly builders for tables and charts.
- `riskplus_core/risk.py`: frequency detection and historical risk statistics.
- `riskplus_core/portfolio.py`: portfolio-weight normalization and construction helpers.
- `riskplus_core/factors.py`: OLS, VIF, and factor-bucket mapping helpers.
- `riskplus_core/simulation.py`: Student-t fit and Monte Carlo simulation helpers.
- `riskplus_core/contribution.py`: marginal contribution, implied return, and budgeting helpers.
- `riskplus_core/attribution.py`: systematic/specific risk and factor contribution helpers.
- `riskplus_core/analytics.py`: compatibility facade that re-exports the public API.

## Sidebar Settings Collected But Not Used In Core Calculations

- `dist_type`: collected and shown in the settings table, but not used in analysis.
- `corr_method`: collected and shown in the settings table, but not used in analysis.
- `ewma_decay`: collected and shown only when `corr_method == "EWMA"`, but not used.
- `display_mode`: collected in the sidebar but not used yet.
- `portfolio_value`: collected and shown in the settings table, but not used in calculations.
- `report_name`: used in the settings table only.

## Likely Data Scaling Problems

- `values_in_percent` is a manual switch only; the app does not auto-detect decimal versus percent input.
- `prepare_return_stream` and `prepare_factor_stream` apply a fixed divide-by-100 conversion when the checkbox is enabled, so mixed-scale uploads can be misread.
- The current smoke path simulates a single portfolio series and then reuses that series for risk budgeting, so the contribution outputs are not yet multi-fund aware.

## Multi-Fund Inputs Collapsed Too Early

- `app.py` merges all fund uploads into one `merged_data` frame before any deeper analysis.
- `run_core_analysis` then builds a single aggregated portfolio series from those assets.
- Risk budgeting and factor contribution still operate on the aggregated portfolio rather than keeping fund-level separation.

## Hardcoded Frequency Usage

- `riskplus_core/engine.py` computes historical stats using detected frequency, but simulated stats are still passed `'monthly'` explicitly.
- This means simulated annualization is not fully aligned with the detected sampling frequency.

## Upload Modes No Longer Supported Compared With Earlier Versions

- The app no longer supports a single combined fund-and-factor upload path; fund files and factor files are now separate.
- The current workflow does not support multi-sheet workbook selection or alternate upload layouts beyond the current CSV/XLSX file-per-fund flow.
- The `display_mode` sidebar option is visible but does not yet change exposure logic.

## Prioritized Fix List

1. Replace the hardcoded simulated-stat frequency with the detected frequency path.
2. Add explicit fund-level handling before portfolio aggregation so multi-fund analysis is not collapsed too early.
3. Auto-detect or validate percent-versus-decimal scaling more robustly.
4. Either wire `display_mode` into exposure logic or remove it from the sidebar.
5. Revisit the upload flow if a combined workbook or sheet-based mode is still required.

## Notes

- No mathematical formulas were changed in this stabilization pass.
- The current refactor compiles and the smoke tests pass.