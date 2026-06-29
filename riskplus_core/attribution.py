"""Systematic and factor attribution helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .constants import DEFAULT_CONFIDENCE
from .factors import get_factor_bucket_mapping


def compute_systematic_specific_risk(
    returns: pd.Series,
    factor_returns: pd.DataFrame,
    ols_model: Any | None = None,
) -> dict[str, float]:
    """Split total risk into systematic and specific variance components."""
    if ols_model is None:
        from .data import run_ols
        ols_model = run_ols(returns, factor_returns)['model']

    total_var = float(returns.var(ddof=1))

    betas = ols_model.params.drop(labels=['const'], errors='ignore')
    factor_cov = factor_returns.cov()
    systematic_var = float(np.dot(betas.values, np.dot(factor_cov.values, betas.values.T)))
    specific_var = float(ols_model.resid.var(ddof=1))

    total = systematic_var + specific_var
    systematic_pct = systematic_var / total if total > 0 else 0.0
    specific_pct = specific_var / total if total > 0 else 0.0

    return {
        'systematic_var': systematic_var,
        'specific_var': specific_var,
        'total_var': total_var,
        'systematic_pct': systematic_pct,
        'specific_pct': specific_pct,
    }


def compute_factor_contribution(
    ols_model: Any,
    factor_returns: pd.DataFrame,
    portfolio_returns: pd.Series,
    simulated_portfolio_returns: pd.Series,
    confidence: float = DEFAULT_CONFIDENCE,
) -> dict[str, Any]:
    """Return the current factor contribution bundle used by the UI."""
    betas = ols_model.params.drop(labels=['const'], errors='ignore')
    factor_names = betas.index.tolist()

    factor_cov = factor_returns.cov()
    residuals = ols_model.resid
    total_var = portfolio_returns.var(ddof=1)

    systematic_var = float(np.dot(betas.values, np.dot(factor_cov.values, betas.values.T)))
    specific_var = float(residuals.var(ddof=1))

    systematic_stdev = np.sqrt(max(systematic_var, 0))
    specific_stdev = np.sqrt(max(specific_var, 0))
    total_stdev = np.sqrt(total_var)

    systematic_pct = systematic_stdev / total_stdev if total_stdev > 1e-10 else 0.0
    specific_pct = specific_stdev / total_stdev if total_stdev > 1e-10 else 0.0

    factor_mc_stdev = np.zeros(len(factor_names))
    for i, _factor in enumerate(factor_names):
        factor_mc_stdev[i] = betas.iloc[i] * factor_cov.iloc[i, i] / (systematic_stdev + 1e-10) if systematic_stdev > 1e-10 else 0.0

    factor_contrib_etl = np.abs(factor_mc_stdev) * (np.std(simulated_portfolio_returns) / (total_stdev + 1e-10))
    bucket_mapping = get_factor_bucket_mapping(factor_names)

    return {
        'systematic_stdev_pct': systematic_pct,
        'specific_stdev_pct': specific_pct,
        'betas': betas,
        'factor_names': factor_names,
        'factor_mc_stdev': factor_mc_stdev,
        'factor_contrib_etl': factor_contrib_etl,
        'bucket_mapping': bucket_mapping,
    }