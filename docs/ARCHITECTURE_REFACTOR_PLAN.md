# Architecture Refactor Plan

## Current Responsibilities

### `app.py`
- Owns the Streamlit UI lifecycle.
- Handles sidebar upload controls, column mapping, parameter widgets, and tab rendering.
- Orchestrates the analysis pipeline by calling data-loading and analytics helpers.
- Stores analysis state in `st.session_state` for the current interactive run.
- Displays tables, charts, warnings, and summaries.

### `riskplus_core/data.py`
- Reads uploaded CSV/XLSX files.
- Infers or normalizes fund names.
- Prepares individual fund return streams and factor streams.
- Merges multiple fund files and factor files into one analysis frame.
- Validates inputs for missing values, duplicates, and minimum observations.
- Detects frequency and computes annualization factors.
- Builds the weighted portfolio return series.
- Runs OLS setup and VIF calculations.

### `riskplus_core/analytics.py`
- Computes historical risk statistics.
- Fits Student-t distributions.
- Simulates fat-tailed return paths.
- Computes risk metrics from distributions.
- Computes marginal and percent risk contributions.
- Computes systematic vs specific risk.
- Computes implied returns and risk budgeting tables.
- Maps factors into risk buckets.
- Computes factor contribution and factor bucket exposure outputs.

## Target Architecture

The target structure is to split responsibilities into smaller, purpose-built modules while keeping the Streamlit app as a thin orchestration layer.

### Proposed modules
- `riskplus_core/data.py` - file reading, normalization, validation, joining, portfolio construction, frequency detection.
- `riskplus_core/portfolio.py` - portfolio weight handling, return aggregation, portfolio metadata helpers.
- `riskplus_core/factors.py` - factor file parsing, factor naming, factor selection, factor normalization.
- `riskplus_core/simulation.py` - Monte Carlo, Student-t fitting, scenario generation.
- `riskplus_core/risk.py` - historical risk metrics, VaR, ETL, ETR, Sharpe, STARR, Rachev, drawdown.
- `riskplus_core/contribution.py` - marginal and percent risk contributions.
- `riskplus_core/attribution.py` - factor attribution, systematic/specific decomposition.
- `riskplus_core/scenarios.py` - scenario aggregation helpers, stress/scenario composition.
- `riskplus_core/export.py` - Excel export and workbook formatting.
- `riskplus_core/reporting.py` - tab/table/chart payload preparation and display-friendly summaries.
- `riskplus_core/engine.py` - top-level pipeline coordination and shared analysis flow.
- `riskplus_core/constants.py` - default parameters and shared configuration constants.
- `riskplus_core/models.py` - typed data containers, dataclasses, and shared schemas.

## Function Placement Plan

### Keep in `riskplus_core/data.py`
- `read_uploaded_file`
- `infer_fund_name_from_file`
- `prepare_return_stream`
- `prepare_factor_stream`
- `merge_analysis_frames`
- `validate_raw_data`
- `prepare_analysis_data`
- `build_portfolio_series`
- `detect_frequency`
- `annualization_factor`
- `run_ols`
- `compute_vif`

### Move to `riskplus_core/portfolio.py`
- `build_portfolio_series` if the portfolio module is separated from pure data loading.
- Any future helpers for weighted portfolio composition.

### Move to `riskplus_core/factors.py`
- Factor parsing helpers when factor-specific metadata grows beyond simple frame preparation.
- Any factor selection or factor-normalization helpers that are UI-independent.

### Move to `riskplus_core/simulation.py`
- `fit_student_t_distribution`
- `simulate_fat_tailed_returns`

### Move to `riskplus_core/risk.py`
- `compute_historical_stats`
- `compute_risk_metrics_from_dist`
- `compute_implied_returns`

### Move to `riskplus_core/contribution.py`
- `compute_marginal_risk_contributions`
- `compute_percent_risk_contributions`

### Move to `riskplus_core/attribution.py`
- `compute_systematic_specific_risk`
- `compute_factor_contribution`
- `compute_factor_bucket_exposures`
- `get_factor_bucket_mapping`

### Move to `riskplus_core/export.py`
- Workbook creation helpers.
- Any Excel formatting or multi-sheet export utilities.

### Move to `riskplus_core/reporting.py`
- Functions that convert calculations into display-ready tables, labels, and summary payloads.

### Move to `riskplus_core/engine.py`
- A top-level `run_analysis(...)` coordinator that accepts inputs, calls the backend modules, and returns a structured result object for the UI.

### Move to `riskplus_core/constants.py`
- `MIN_OBSERVATIONS`
- `DEFAULT_CONFIDENCE`
- `DEFAULT_RF_RATE`
- `DEFAULT_EWMA_DECAY`
- `DEFAULT_NUM_SIMULATIONS`
- `DEFAULT_STUDENT_T_DF`

### Move to `riskplus_core/models.py`
- Typed dataclasses for uploaded file metadata.
- Analysis result containers.
- Portfolio/factor/fund input schemas.

## Backward Compatibility Strategy

1. Keep existing imports working by re-exporting moved functions from the original modules during transition.
2. Add lightweight wrapper imports in the old files so callers can keep using the same module paths.
3. Introduce new modules incrementally and move one function group at a time.
4. Avoid renaming public functions until the new module paths are stable.
5. Keep `riskplus.py` as a launcher and `app.py` as the UI entrypoint so the run command does not change.

## Safe Refactor Sequence

### Step 1
- Create `constants.py` and `models.py`.
- Move only constants and dataclasses first.

### Step 2
- Create `simulation.py` and move Student-t simulation helpers there.
- Keep compatibility re-exports in `analytics.py`.

### Step 3
- Create `risk.py` and move historical risk metrics there.

### Step 4
- Create `contribution.py` and move risk contribution functions there.

### Step 5
- Create `attribution.py` and move factor attribution and bucket helpers there.

### Step 6
- Create `portfolio.py` and `factors.py` if their responsibilities start growing beyond file prep.

### Step 7
- Create `engine.py` and move the orchestration pipeline into it.

### Step 8
- Move export/reporting helpers into `export.py` and `reporting.py`.

### Step 9
- Trim `app.py` down to UI-only orchestration.

## Smoke Checks After Each Step

1. Run `python -m py_compile` on the touched files.
2. Run the Streamlit app and verify it starts.
3. Upload the sample single-fund CSV and confirm the main tabs still render.
4. Upload the multi-fund sample and confirm the merged portfolio still analyzes correctly.
5. Verify the factor bucket tab still renders with the same output shape.
6. Confirm the Run Analysis flow still gates computation correctly.

## Notes

- This plan does not change application behavior yet.
- The current codebase already has a partial split, so this refactor should be done incrementally rather than as a single rewrite.
- The safest path is to preserve public APIs and add new modules behind compatibility wrappers until the UI and tests are stable.