"""Student-t fitting and Monte Carlo simulation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.stats as stats

from .constants import DEFAULT_NUM_SIMULATIONS


def fit_student_t_distribution(returns: pd.Series) -> dict[str, float]:
    """Fit a Student-t distribution, falling back to simple moments on failure."""
    returns_clean = returns.dropna().values
    if len(returns_clean) < 10:
        return {'df': np.inf, 'loc': float(returns_clean.mean()), 'scale': float(returns_clean.std())}

    try:
        params = stats.t.fit(returns_clean)
        df, loc, scale = params
        df = max(2.0, float(df))
        return {'df': df, 'loc': float(loc), 'scale': float(scale)}
    except Exception:
        return {'df': 5.0, 'loc': float(returns_clean.mean()), 'scale': float(returns_clean.std())}


def simulate_fat_tailed_returns(
    returns_or_cov: pd.DataFrame | pd.Series,
    n_sims: int = DEFAULT_NUM_SIMULATIONS,
    random_seed: int | None = None,
) -> pd.DataFrame:
    """Simulate fat-tailed returns for a series or a correlated dataframe."""
    if random_seed is not None:
        np.random.seed(random_seed)

    if isinstance(returns_or_cov, pd.Series):
        t_params = fit_student_t_distribution(returns_or_cov)
        df, loc, scale = t_params['df'], t_params['loc'], t_params['scale']

        if np.isinf(df):
            sims = np.random.normal(loc, scale, n_sims)
        else:
            sims = stats.t.rvs(df, loc=loc, scale=scale, size=n_sims)

        return pd.DataFrame({'Portfolio': sims})

    returns_df = returns_or_cov.copy()
    n_assets = returns_df.shape[1]

    t_params_list = [fit_student_t_distribution(returns_df[col]) for col in returns_df.columns]

    corr = returns_df.corr().fillna(0)
    try:
        L = np.linalg.cholesky(corr.values)
    except np.linalg.LinAlgError:
        L = np.linalg.cholesky(np.eye(n_assets) * 0.99 + corr.values * 0.01)

    normal_sims = np.random.normal(0, 1, (n_sims, n_assets))
    correlated_normal = normal_sims @ L.T

    simulated_returns = np.zeros((n_sims, n_assets))
    for i, t_param in enumerate(t_params_list):
        df, loc, scale = t_param['df'], t_param['loc'], t_param['scale']
        if np.isinf(df):
            simulated_returns[:, i] = correlated_normal[:, i] * scale + loc
        else:
            u = stats.norm.cdf(correlated_normal[:, i])
            simulated_returns[:, i] = stats.t.ppf(u, df, loc=loc, scale=scale)

    return pd.DataFrame(simulated_returns, columns=returns_df.columns)