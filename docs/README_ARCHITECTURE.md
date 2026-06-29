# RiskPlus Streamlit Architecture

This repository has been refactored into a small set of purpose-specific modules so the Streamlit UI stays thin and the backend remains reusable.

## Module Layout

- `riskplus.py` - Streamlit entrypoint that launches the app.
- `app.py` - UI orchestration only: collects inputs, calls backend functions, and renders tabs.
- `riskplus_core/data.py` - file loading, validation, preparation, and compatibility wrappers.
- `riskplus_core/portfolio.py` - portfolio construction helpers.
- `riskplus_core/factors.py` - factor model helpers, VIF, and factor bucket mapping.
- `riskplus_core/risk.py` - frequency detection and historical risk metrics.
- `riskplus_core/simulation.py` - Student-t fitting and Monte Carlo simulation.
- `riskplus_core/contribution.py` - marginal risk contribution and budgeting helpers.
- `riskplus_core/attribution.py` - systematic/specific risk and factor attribution helpers.
- `riskplus_core/reporting.py` - pure DataFrame and Plotly builders for UI tables and charts.
- `riskplus_core/engine.py` - reusable core analysis pipeline.
- `riskplus_core/constants.py` - shared defaults and thresholds.
- `riskplus_core/models.py` - dataclasses for structured results and metadata.

## Compatibility Strategy

The refactor preserves older import paths while modules are being split:

- `riskplus_core.analytics` is now a compatibility facade that re-exports the public API.
- `riskplus_core.data` still exposes `run_ols`, `compute_vif`, `build_portfolio_series`, `detect_frequency`, and `annualization_factor`.
- The Streamlit app continues to use the same eight tabs and the same reporting content.

## Current Design Goals

- Keep business logic out of `app.py`.
- Keep UI table and chart construction in `reporting.py`.
- Keep analysis orchestration in `engine.py`.
- Keep backend helpers pure whenever possible.
- Preserve behavior first, then refine methodology in later passes.
