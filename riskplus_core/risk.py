"""Frequency detection and historical risk statistics helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.stats as stats

from .constants import DEFAULT_RF_RATE


def annualization_factor(freq: str) -> int:
    """Map a frequency label to its annualization factor."""
    return {'daily': 252, 'monthly': 12, 'quarterly': 4}.get(freq, 12)


def detect_frequency(index: pd.DatetimeIndex) -> str:
    """Infer a rough sampling frequency from a datetime index."""
    if len(index) < 3:
        return 'unknown'
    day_gaps = pd.Series(index).sort_values().diff().dropna().dt.days
    median_gap = float(day_gaps.median())
    if median_gap <= 3:
        return 'daily'
    if median_gap <= 40:
        return 'monthly'
    if median_gap <= 120:
        return 'quarterly'
    return 'unknown'


def compute_historical_stats(
    returns: pd.Series,
    freq: str,
    rf_rate: float = DEFAULT_RF_RATE,
) -> dict[str, float]:
    """Compute the existing historical risk summary without changing the formulas."""
    af = annualization_factor(freq)
    periodic_rf = rf_rate / af

    returns_clean = returns.dropna()
    obs = len(returns_clean)

    if obs < 2:
        return {f: 0.0 for f in ['mean', 'ann_mean', 'vol', 'ann_vol', 'skew', 'xkurt', 'var', 'etl', 'etr', 'sharpe', 'starr', 'rachev', 'max_dd', 'best_period', 'worst_period', 'obs_count']}

    mean_ret = float(returns_clean.mean())
    ann_mean = mean_ret * af
    vol = float(returns_clean.std(ddof=1))
    ann_vol = vol * np.sqrt(af)

    skew = float(stats.skew(returns_clean, bias=False))
    xkurt = float(stats.kurtosis(returns_clean, fisher=True, bias=False))

    left_q = 0.05
    right_q = 0.95
    var_val = float(-np.quantile(returns_clean, left_q))

    tail_returns = returns_clean[returns_clean <= np.quantile(returns_clean, left_q)]
    etl_val = float(-tail_returns.mean()) if len(tail_returns) > 0 else var_val

    right_tail_returns = returns_clean[returns_clean >= np.quantile(returns_clean, right_q)]
    etr_val = float(right_tail_returns.mean()) if len(right_tail_returns) > 0 else mean_ret

    sharpe = (mean_ret - periodic_rf) / vol if vol > 1e-10 else np.nan
    starr = (mean_ret - periodic_rf) / etl_val if etl_val > 1e-10 else np.nan
    rachev = etr_val / etl_val if etl_val > 1e-10 else np.nan

    cumsum = (1 + returns_clean).cumprod()
    running_max = cumsum.expanding().max()
    drawdown = cumsum / running_max - 1
    max_dd = float(drawdown.min()) if not drawdown.empty else np.nan

    best_period = float(returns_clean.max())
    worst_period = float(returns_clean.min())

    return {
        'mean': mean_ret,
        'ann_mean': ann_mean,
        'vol': vol,
        'ann_vol': ann_vol,
        'skew': skew,
        'xkurt': xkurt,
        'var': var_val,
        'etl': etl_val,
        'etr': etr_val,
        'sharpe': sharpe,
        'starr': starr,
        'rachev': rachev,
        'max_dd': max_dd,
        'best_period': best_period,
        'worst_period': worst_period,
        'obs_count': int(obs),
    }


def compute_risk_metrics_from_dist(
    returns: np.ndarray | pd.Series,
    confidence: float,
) -> dict[str, float]:
    """Compute VaR, ETL, and ETR from a simulated return distribution."""
    returns_clean = np.asarray(returns).flatten()
    returns_clean = returns_clean[~np.isnan(returns_clean)]

    left_q = 1 - confidence
    right_q = confidence

    var_val = float(-np.quantile(returns_clean, left_q))
    tail_mask = returns_clean <= np.quantile(returns_clean, left_q)
    etl_val = float(-returns_clean[tail_mask].mean()) if tail_mask.sum() > 0 else var_val

    right_tail_mask = returns_clean >= np.quantile(returns_clean, right_q)
    etr_val = float(returns_clean[right_tail_mask].mean()) if right_tail_mask.sum() > 0 else returns_clean.mean()

    return {'var': var_val, 'etl': etl_val, 'etr': etr_val}