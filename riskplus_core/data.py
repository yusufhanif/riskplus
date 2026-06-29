"""Data loading, validation, and compatibility helpers for RiskPlus Streamlit."""

from __future__ import annotations

from io import BytesIO
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from .constants import MIN_OBSERVATIONS
from .risk import annualization_factor as _annualization_factor, detect_frequency as _detect_frequency
from .factors import compute_vif as _compute_vif, run_ols as _run_ols
from .portfolio import (
    build_portfolio_composition_table,
    build_portfolio_series as _build_portfolio_series,
    filter_analysis_period,
    normalize_weights,
)


@st.cache_data(show_spinner=False)
def read_uploaded_file(file_name: str, file_bytes: bytes) -> pd.DataFrame:
    if file_name.lower().endswith('.csv'):
        return pd.read_csv(BytesIO(file_bytes))
    if file_name.lower().endswith(('.xlsx', '.xls')):
        return pd.read_excel(BytesIO(file_bytes))
    raise ValueError('Unsupported file type. Upload CSV or Excel.')


def infer_fund_name_from_file(file_name: str) -> str:
    name = file_name.rsplit('.', 1)[0]
    name = name.replace('_', ' ').replace('-', ' ')
    return name.strip() or 'Fund'


def prepare_return_stream(
    df: pd.DataFrame,
    date_col: str,
    return_col: str,
    stream_name: str,
    values_in_percent: bool,
) -> pd.DataFrame:
    frame = df[[date_col, return_col]].copy()
    frame[date_col] = pd.to_datetime(frame[date_col], errors='coerce')
    frame[return_col] = pd.to_numeric(frame[return_col], errors='coerce')
    frame = frame.dropna().sort_values(date_col)
    frame = frame.drop_duplicates(subset=[date_col], keep='first')
    frame = frame.set_index(date_col)
    if values_in_percent:
        frame[return_col] = frame[return_col] / 100.0
    return frame.rename(columns={return_col: stream_name})


def prepare_factor_stream(
    df: pd.DataFrame,
    date_col: str,
    factor_cols: list[str],
    values_in_percent: bool,
) -> pd.DataFrame:
    frame = df[[date_col, *factor_cols]].copy()
    frame[date_col] = pd.to_datetime(frame[date_col], errors='coerce')
    for col in factor_cols:
        frame[col] = pd.to_numeric(frame[col], errors='coerce')
    frame = frame.dropna().sort_values(date_col)
    frame = frame.drop_duplicates(subset=[date_col], keep='first')
    frame = frame.set_index(date_col)
    if values_in_percent:
        frame[factor_cols] = frame[factor_cols] / 100.0
    return frame


def validate_raw_data(
    df: pd.DataFrame,
    date_col: str,
    asset_cols: list[str],
    factor_cols: list[str],
    min_observations: int,
    max_missing_pct: float,
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    selected_cols = [date_col, *asset_cols, *factor_cols]
    if len(asset_cols) < 1:
        errors.append('Select at least one fund/portfolio return column.')
        return errors, warnings

    if len(factor_cols) < 1:
        errors.append('Select at least one factor column.')
        return errors, warnings

    if len(set(selected_cols)) != len(selected_cols):
        errors.append('Date, fund, and factor columns must be unique.')

    missing_cols = [col for col in selected_cols if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing selected columns: {', '.join(missing_cols)}")
        return errors, warnings

    date_parsed = pd.to_datetime(df[date_col], errors='coerce')
    if date_parsed.isna().all():
        errors.append('Date column could not be parsed into valid dates.')
        return errors, warnings

    if date_parsed.duplicated().any():
        errors.append('Duplicate dates found.')

    numeric_cols = [*asset_cols, *factor_cols]
    numeric_frame = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    for col in numeric_cols:
        col_missing_pct = float(numeric_frame[col].isna().mean())
        if col_missing_pct > max_missing_pct:
            errors.append(f'{col}: {col_missing_pct:.1%} missing (limit: {max_missing_pct:.1%}).')

    candidate = pd.concat([date_parsed.rename('Date'), numeric_frame], axis=1).dropna()
    if len(candidate) < min_observations:
        errors.append(f'Only {len(candidate)} complete rows; need {min_observations}.')

    extreme_mask = numeric_frame.abs() > 5
    if extreme_mask.any().any():
        warnings.append('Extreme returns (>500%) detected. Check if data is scaled correctly.')

    if len(factor_cols) > 1:
        corr_matrix = numeric_frame[factor_cols].corr()
        high_corr = (corr_matrix.abs() > 0.75) & (corr_matrix != 1.0)
        if high_corr.any().any():
            warnings.append('Some factors are highly correlated (>0.75). Consider removing redundant factors.')

    return errors, warnings


def prepare_analysis_data(
    df: pd.DataFrame,
    date_col: str,
    asset_cols: list[str],
    factor_cols: list[str],
    values_in_percent: bool,
) -> pd.DataFrame:
    frame = df[[date_col, *asset_cols, *factor_cols]].copy()
    frame[date_col] = pd.to_datetime(frame[date_col], errors='coerce')
    for col in [*asset_cols, *factor_cols]:
        frame[col] = pd.to_numeric(frame[col], errors='coerce')

    frame = frame.dropna().sort_values(date_col)
    frame = frame.drop_duplicates(subset=[date_col], keep='first')
    frame = frame.set_index(date_col)

    if values_in_percent:
        frame[[*asset_cols, *factor_cols]] = frame[[*asset_cols, *factor_cols]] / 100.0

    return frame


def merge_analysis_frames(
    fund_frames: dict[str, pd.DataFrame],
    factor_frame: pd.DataFrame | None,
    join_type: str = 'inner',
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = list(fund_frames.values())
    if factor_frame is not None:
        frames.append(factor_frame)

    if not frames:
        return pd.DataFrame()

    merged = pd.concat(frames, axis=1, join=join_type).sort_index()
    merged = merged.loc[:, ~merged.columns.duplicated()]
    return merged


def build_portfolio_series(
    data: pd.DataFrame,
    asset_cols: list[str],
    asset_weights: pd.Series | dict[str, float] | list[float] | np.ndarray | None = None,
) -> tuple[pd.Series, pd.Series]:
    return _build_portfolio_series(data, asset_cols, asset_weights)


def detect_frequency(index: pd.DatetimeIndex) -> str:
    return _detect_frequency(index)


def annualization_factor(freq: str) -> int:
    return _annualization_factor(freq)


def run_ols(portfolio_returns: pd.Series, factor_returns: pd.DataFrame) -> dict[str, Any]:
    return _run_ols(portfolio_returns, factor_returns)


def compute_vif(factor_returns: pd.DataFrame) -> pd.DataFrame:
    return _compute_vif(factor_returns)
