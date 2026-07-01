"""Core analysis pipeline orchestration for the Streamlit app."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .analytics import (
    compute_percent_risk_contributions,
    compute_factor_bucket_exposures,
    compute_factor_contribution,
    compute_historical_stats,
    compute_marginal_risk_contributions,
    compute_risk_budgeting_table,
    compute_systematic_specific_risk,
    simulate_fat_tailed_returns,
)
from .data import annualization_factor, build_portfolio_series, detect_frequency, run_ols
from .factors import get_factor_bucket_mapping
from .models import CoreAnalysisResults, NormalizedDataSource


def _coerce_bundle(
    analysis_input: pd.DataFrame | NormalizedDataSource | dict[str, object],
    asset_cols: list[str] | None,
    factor_cols: list[str] | None,
    asset_weight_input: pd.Series | dict[str, float] | list[float] | np.ndarray | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], list[str], pd.Series | dict[str, float] | list[float] | np.ndarray | None, dict[str, object]]:
    if isinstance(analysis_input, NormalizedDataSource):
        return (
            analysis_input.merged_fund_returns,
            analysis_input.factor_returns,
            analysis_input.analysis_data,
            analysis_input.asset_cols,
            analysis_input.factor_cols,
            analysis_input.asset_weight_input,
            analysis_input.data_source_metadata,
        )

    if isinstance(analysis_input, dict) and 'merged_fund_returns' in analysis_input and 'analysis_data' in analysis_input:
        return (
            analysis_input['merged_fund_returns'],
            analysis_input['factor_returns'],
            analysis_input['analysis_data'],
            list(analysis_input['asset_cols']),
            list(analysis_input['factor_cols']),
            analysis_input['asset_weight_input'],
            dict(analysis_input.get('data_source_metadata', {})),
        )

    if asset_cols is None or factor_cols is None:
        raise ValueError('asset_cols and factor_cols are required when passing a dataframe to run_core_analysis.')

    metadata = {
        'mode': 'legacy_dataframe',
        'warnings': [],
    }
    if isinstance(analysis_input, pd.DataFrame):
        return analysis_input, analysis_input[factor_cols], analysis_input, asset_cols, factor_cols, asset_weight_input, metadata

    raise TypeError('Unsupported analysis input type.')


def run_core_analysis(
    filtered_data: pd.DataFrame | NormalizedDataSource | dict[str, object],
    asset_cols: list[str] | None = None,
    factor_cols: list[str] | None = None,
    asset_weight_input: pd.Series | dict[str, float] | list[float] | np.ndarray | None = None,
    rf_rate: float = 0.02,
    confidence: float = 0.95,
    num_sims: int = 50000,
    random_seed: int = 42,
) -> CoreAnalysisResults:
    """Run the existing single-portfolio analysis pipeline and return structured outputs."""
    merged_fund_returns, factor_returns, analysis_data, asset_cols, factor_cols, asset_weight_input, metadata = _coerce_bundle(
        filtered_data,
        asset_cols,
        factor_cols,
        asset_weight_input,
    )

    portfolio_history = merged_fund_returns[asset_cols].dropna()
    if portfolio_history.empty:
        raise ValueError('No usable fund return rows were found after alignment.')

    frequency = detect_frequency(portfolio_history.index)
    portfolio, asset_weights = build_portfolio_series(portfolio_history, asset_cols, asset_weight_input)
    hist_stats = compute_historical_stats(portfolio, frequency, rf_rate)
    hist_stats_by_fund = {
        fund: compute_historical_stats(portfolio_history[fund], frequency, rf_rate)
        for fund in asset_cols
    }

    simulated_portfolio_returns = simulate_fat_tailed_returns(portfolio, n_sims=int(num_sims), random_seed=random_seed)['Portfolio']
    simulated_fund_returns = simulate_fat_tailed_returns(portfolio_history, n_sims=int(num_sims), random_seed=random_seed)
    sim_stats = compute_historical_stats(simulated_portfolio_returns, frequency, rf_rate)
    sim_stats_by_fund = {
        fund: compute_historical_stats(simulated_fund_returns[fund], frequency, rf_rate)
        for fund in asset_cols
    }

    required_cols = asset_cols + factor_cols
    if set(required_cols).issubset(analysis_data.columns):
        combined = analysis_data[required_cols].dropna()
    else:
        combined = pd.concat([portfolio_history, factor_returns[factor_cols]], axis=1, join='inner').dropna()
    if combined.empty:
        raise ValueError(
            'No overlapping usable rows were found between fund returns and factor returns. '
            'Check the upload dates, selected columns, and scaling.'
        )
    if len(combined) < max(2, len(factor_cols) + 1):
        raise ValueError(
            f'Only {len(combined)} overlapping rows remain after alignment; '
            f'need at least {max(2, len(factor_cols) + 1)} rows for factor regression.'
        )

    portfolio_input = combined[asset_cols]
    portfolio_for_factor_model, _ = build_portfolio_series(portfolio_input, asset_cols, asset_weights)
    factors = combined[factor_cols]

    ols_results = run_ols(portfolio_for_factor_model, factors)

    mc_contribs = compute_marginal_risk_contributions(asset_weights.values, simulated_fund_returns, confidence)
    pc_contribs = compute_percent_risk_contributions(asset_weights.values, mc_contribs)
    sys_spec = compute_systematic_specific_risk(portfolio_for_factor_model, factors, ols_results['model'])
    factor_contrib = compute_factor_contribution(
        ols_results['model'],
        factors,
        portfolio_for_factor_model,
        simulated_portfolio_returns,
        confidence,
    )

    rb_etl = compute_risk_budgeting_table(
        asset_weights.values,
        simulated_fund_returns,
        mc_contribs['mc_etl'],
        'ETL',
        rf_rate,
        annualization_factor(frequency),
    )
    rb_stdev = compute_risk_budgeting_table(
        asset_weights.values,
        simulated_fund_returns,
        mc_contribs['mc_vol'],
        'StDev',
        rf_rate,
        annualization_factor(frequency),
    )

    bucket_mapping = get_factor_bucket_mapping(factor_cols)
    bucket_exposures = compute_factor_bucket_exposures(ols_results['coef_table']['Coefficient'], bucket_mapping)

    return CoreAnalysisResults(
        frequency=frequency,
        portfolio=portfolio,
        asset_weights=asset_weights,
        factors=factors,
        simulated_fund_returns=simulated_fund_returns,
        simulated_portfolio_returns=simulated_portfolio_returns,
        simulated_returns=simulated_portfolio_returns.to_frame(name='Portfolio'),
        hist_stats=hist_stats,
        sim_stats=sim_stats,
        hist_stats_by_fund=hist_stats_by_fund,
        sim_stats_by_fund=sim_stats_by_fund,
        ols_results=ols_results,
        mc_contribs=mc_contribs,
        pc_contribs=pc_contribs,
        sys_spec=sys_spec,
        factor_contrib=factor_contrib,
        rb_etl=rb_etl,
        rb_stdev=rb_stdev,
        bucket_exposures=bucket_exposures,
        data_source_metadata=metadata,
        fund_returns_full=merged_fund_returns,
        factor_returns_full=factor_returns,
        fund_factor_overlap=analysis_data,
    )