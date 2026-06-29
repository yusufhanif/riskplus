"""Factor model helpers, bucket mapping, and compatibility wrappers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


def run_ols(portfolio_returns: pd.Series, factor_returns: pd.DataFrame) -> dict[str, Any]:
    """Fit OLS with HC1 covariance and return the coefficient summary bundle."""
    x = sm.add_constant(factor_returns, has_constant='add')
    model = sm.OLS(portfolio_returns, x)
    fitted = model.fit(cov_type='HC1')

    coef_table = pd.DataFrame(
        {
            'Coefficient': fitted.params,
            'StdError': fitted.bse,
            'tStat': fitted.tvalues,
            'pValue': fitted.pvalues,
        }
    )
    coef_table.index.name = 'Term'
    coef_table['Significant'] = coef_table['pValue'] < 0.05

    return {
        'model': fitted,
        'coef_table': coef_table,
        'residuals': fitted.resid,
        'fitted_values': fitted.fittedvalues,
        'r2': float(fitted.rsquared),
        'adj_r2': float(fitted.rsquared_adj),
    }


def compute_vif(factor_returns: pd.DataFrame) -> pd.DataFrame:
    """Compute VIF for each factor while excluding the constant term from output."""
    x = sm.add_constant(factor_returns, has_constant='add')
    vif_rows: list[dict[str, Any]] = []
    for i, col in enumerate(x.columns):
        if col == 'const':
            continue
        vif_rows.append({'Factor': col, 'VIF': variance_inflation_factor(x.values, i)})
    return pd.DataFrame(vif_rows).sort_values('VIF', ascending=False)


def get_factor_bucket_mapping(factor_names: list[str]) -> dict[str, list[str]]:
    """Group factor names into RiskPlus-style buckets using keyword matching."""
    buckets = {
        'Equity Risk': [],
        'Commodity Risk': [],
        'FX Risk': [],
        'Interest Rate Risk': [],
        'Sovereign Risk': [],
        'Corporate Risk': [],
        'Volatility Risk': [],
        'Fixed Income Risk': [],
    }

    keywords = {
        'Equity Risk': ['msci', 'russell', 's&p', 'equity', 'value', 'growth', 'small', 'large', 'world', 'eafe', 'em'],
        'Commodity Risk': ['gsci', 'crude', 'oil', 'commodity', 'gold', 'copper', 'agriculture', 'metal'],
        'FX Risk': ['dollar', 'euro', 'yen', 'sterling', 'pound', 'franc', 'krona', 'renminbi', 'fx', 'exchange'],
        'Interest Rate Risk': ['rate', 'libor', 'sofr', 'yield', 'spread', 'maturity', 'treasury', 'duration'],
        'Sovereign Risk': ['sovereign', 'government bond', 'treasury', 'master'],
        'Corporate Risk': ['corporate', 'high yield', 'crossover', 'credit', 'bbb', 'aaa'],
        'Volatility Risk': ['vix', 'cboe', 'volatility'],
        'Fixed Income Risk': ['bond', 'mortgage', 'abs', 'mbs', 'cmbs', 'fixed income'],
    }

    for factor in factor_names:
        factor_lower = factor.lower()
        matched = False
        for bucket, kws in keywords.items():
            if any(kw in factor_lower for kw in kws):
                buckets[bucket].append(factor)
                matched = True
                break
        if not matched:
            buckets.setdefault('Other Risk', []).append(factor)

    return {k: v for k, v in buckets.items() if v}


def compute_factor_bucket_exposures(
    betas: pd.Series,
    bucket_mapping: dict[str, list[str]],
) -> pd.DataFrame:
    """Aggregate fitted factor betas into bucket-level exposures."""
    rows = []

    for bucket, factors in bucket_mapping.items():
        matching_betas = [betas.get(f, 0.0) for f in factors if f in betas.index]
        if not matching_betas:
            continue

        rows.append(
            {
                'Bucket': bucket,
                'Exposure': float(sum(matching_betas)),
                'Factor Count': len(matching_betas),
            }
        )

    return pd.DataFrame(rows) if rows else pd.DataFrame()