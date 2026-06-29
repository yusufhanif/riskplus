"""Portfolio construction and analysis-period helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def normalize_weights(
    asset_cols: list[str],
    asset_weights: pd.Series | dict[str, float] | list[float] | np.ndarray | None = None,
) -> pd.Series:
    if asset_weights is None:
        weights = np.repeat(1.0 / len(asset_cols), len(asset_cols))
        return pd.Series(weights, index=asset_cols, dtype=float)

    if not isinstance(asset_weights, pd.Series):
        asset_weights = pd.Series(asset_weights, index=asset_cols, dtype=float)

    asset_weights = asset_weights.astype(float).reindex(asset_cols).fillna(0.0)
    if float(asset_weights.sum()) <= 0:
        weights = np.repeat(1.0 / len(asset_cols), len(asset_cols))
        return pd.Series(weights, index=asset_cols, dtype=float)

    return asset_weights / float(asset_weights.sum())


def build_portfolio_series(
    data: pd.DataFrame,
    asset_cols: list[str],
    asset_weights: pd.Series | dict[str, float] | list[float] | np.ndarray | None = None,
) -> tuple[pd.Series, pd.Series]:
    normalized_weights = normalize_weights(asset_cols, asset_weights)
    portfolio_values = data[asset_cols].values @ normalized_weights.values
    portfolio = pd.Series(portfolio_values, index=data.index, name='Portfolio')
    return portfolio, normalized_weights


def filter_analysis_period(
    data: pd.DataFrame,
    start_date: Any | None,
    end_date: Any | None,
) -> pd.DataFrame:
    filtered = data
    if start_date is not None:
        filtered = filtered.loc[filtered.index >= pd.to_datetime(start_date)]
    if end_date is not None:
        filtered = filtered.loc[filtered.index <= pd.to_datetime(end_date)]
    return filtered


def build_portfolio_composition_table(
    asset_cols: list[str],
    factor_cols: list[str],
    asset_weights: pd.Series | dict[str, float] | list[float] | np.ndarray | None,
) -> pd.DataFrame:
    normalized_weights = normalize_weights(asset_cols, asset_weights)
    composition = pd.DataFrame(
        {
            'Asset': asset_cols + factor_cols,
            'Weight': list(normalized_weights.values) + [0.0] * len(factor_cols),
        }
    )
    return composition