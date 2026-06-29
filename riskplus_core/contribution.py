"""Risk contribution and risk budgeting helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import DEFAULT_CONFIDENCE
from .risk import compute_risk_metrics_from_dist


def compute_marginal_risk_contributions(
    weights: np.ndarray,
    simulated_returns: pd.DataFrame,
    confidence: float = DEFAULT_CONFIDENCE,
    epsilon: float = 0.01,
) -> dict[str, np.ndarray]:
    """Estimate marginal volatility, ETL, and ETR contributions via epsilon perturbation."""
    n_assets = simulated_returns.shape[1]
    weights = np.asarray(weights).flatten()

    base_portfolio = simulated_returns.values @ weights
    base_vol = np.std(base_portfolio, ddof=1)
    base_metrics = compute_risk_metrics_from_dist(base_portfolio, confidence)
    base_etl = base_metrics['etl']
    base_etr = base_metrics['etr']

    mc_vol = np.zeros(n_assets)
    mc_etl = np.zeros(n_assets)
    mc_etr = np.zeros(n_assets)

    for i in range(n_assets):
        weights_perturbed = weights.copy()
        weights_perturbed[i] += epsilon
        weights_perturbed = weights_perturbed / weights_perturbed.sum()

        perturbed_portfolio = simulated_returns.values @ weights_perturbed
        new_vol = np.std(perturbed_portfolio, ddof=1)
        new_metrics = compute_risk_metrics_from_dist(perturbed_portfolio, confidence)
        new_etl = new_metrics['etl']
        new_etr = new_metrics['etr']

        mc_vol[i] = (new_vol - base_vol) / epsilon if base_vol > 1e-10 else 0.0
        mc_etl[i] = (new_etl - base_etl) / epsilon if base_etl > 1e-10 else 0.0
        mc_etr[i] = (new_etr - base_etr) / epsilon if base_etr > 1e-10 else 0.0

    return {'mc_vol': mc_vol, 'mc_etl': mc_etl, 'mc_etr': mc_etr}


def compute_percent_risk_contributions(
    weights: np.ndarray,
    marginal_contribs: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Convert marginal contributions into normalized percentage contributions."""
    weights = np.asarray(weights).flatten()

    pc_vol = np.abs(weights * marginal_contribs['mc_vol'])
    pc_vol = pc_vol / (pc_vol.sum() + 1e-10)

    pc_etl = np.abs(weights * marginal_contribs['mc_etl'])
    pc_etl = pc_etl / (pc_etl.sum() + 1e-10)

    pc_etr = np.abs(weights * marginal_contribs['mc_etr'])
    pc_etr = pc_etr / (pc_etr.sum() + 1e-10)

    return {'pc_vol': pc_vol, 'pc_etl': pc_etl, 'pc_etr': pc_etr}


def compute_implied_returns(
    portfolio_mean: float,
    portfolio_risk: float,
    marginal_contributions: np.ndarray,
    rf_rate: float,
    af: int,
) -> np.ndarray:
    """Compute implied returns using the current ratio-based approximation."""
    periodic_rf = rf_rate / af
    excess_return = portfolio_mean - periodic_rf

    if portfolio_risk < 1e-10:
        return np.full_like(marginal_contributions, portfolio_mean, dtype=float)

    performance_ratio = excess_return / portfolio_risk
    implied_excess = performance_ratio * marginal_contributions
    return implied_excess + periodic_rf


def compute_risk_budgeting_table(
    weights: np.ndarray,
    returns_matrix: pd.DataFrame,
    marginal_contribs: np.ndarray,
    risk_measure: str,
    rf_rate: float,
    af: int,
) -> pd.DataFrame:
    """Build the current risk budgeting table for ETL or standard deviation."""
    asset_names = returns_matrix.columns

    portfolio_rets = returns_matrix.values @ weights
    portfolio_mean = np.mean(portfolio_rets)
    if risk_measure == 'ETL':
        left_q = 0.05
        tail_mask = portfolio_rets <= np.quantile(portfolio_rets, left_q)
        portfolio_risk = np.abs(np.mean(portfolio_rets[tail_mask]))
    else:
        portfolio_risk = np.std(portfolio_rets, ddof=1)

    asset_means = returns_matrix.mean().values
    implied_rets = compute_implied_returns(portfolio_mean, portfolio_risk, marginal_contribs, rf_rate, af)
    differences = asset_means - implied_rets
    status = np.where(differences > 1e-4, '↑ Increase', np.where(differences < -1e-4, '↓ Decrease', '→ Neutral'))

    return pd.DataFrame(
        {
            'Asset': asset_names,
            'Weight (%)': weights * 100,
            'Mean Return (%)': asset_means * 100,
            'MC to Risk': marginal_contribs,
            'Implied Return (%)': implied_rets * 100,
            'Difference (%)': differences * 100,
            'Status': status,
        }
    )