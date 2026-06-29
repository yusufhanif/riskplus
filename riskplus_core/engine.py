"""Core analysis pipeline orchestration for the Streamlit app."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .analytics import (
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
from .models import CoreAnalysisResults


def run_core_analysis(
    filtered_data: pd.DataFrame,
    asset_cols: list[str],
    factor_cols: list[str],
    asset_weight_input: pd.Series | dict[str, float] | list[float] | np.ndarray | None,
    rf_rate: float,
    confidence: float,
    num_sims: int,
    random_seed: int = 42,
) -> CoreAnalysisResults:
    """Run the existing single-portfolio analysis pipeline and return structured outputs."""
    frequency = detect_frequency(filtered_data.index)
    portfolio, asset_weights = build_portfolio_series(filtered_data, asset_cols, asset_weight_input)
    factors = filtered_data[factor_cols]

    ols_results = run_ols(portfolio, factors)
    hist_stats = compute_historical_stats(portfolio, frequency, rf_rate)

    simulated_returns = simulate_fat_tailed_returns(portfolio, n_sims=int(num_sims), random_seed=random_seed)
    sim_stats = compute_historical_stats(simulated_returns['Portfolio'], 'monthly', rf_rate)

    mc_contribs = compute_marginal_risk_contributions(np.array([1.0]), simulated_returns, confidence)
    sys_spec = compute_systematic_specific_risk(portfolio, factors, ols_results['model'])
    factor_contrib = compute_factor_contribution(
        ols_results['model'],
        factors,
        portfolio,
        simulated_returns['Portfolio'],
        confidence,
    )

    rb_etl = compute_risk_budgeting_table(
        np.array([1.0]),
        simulated_returns,
        mc_contribs['mc_etl'],
        'ETL',
        rf_rate,
        annualization_factor(frequency),
    )
    rb_stdev = compute_risk_budgeting_table(
        np.array([1.0]),
        simulated_returns,
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
        simulated_returns=simulated_returns,
        hist_stats=hist_stats,
        sim_stats=sim_stats,
        ols_results=ols_results,
        mc_contribs=mc_contribs,
        sys_spec=sys_spec,
        factor_contrib=factor_contrib,
        rb_etl=rb_etl,
        rb_stdev=rb_stdev,
        bucket_exposures=bucket_exposures,
    )