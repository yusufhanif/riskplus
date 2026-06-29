"""Normalized upload-mode helpers for RiskPlus Streamlit."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .data import detect_frequency, prepare_factor_stream, prepare_return_stream
from .models import NormalizedDataSource
from .portfolio import normalize_weights


def _prepare_wide_panel(
    df: pd.DataFrame,
    date_col: str,
    selected_cols: list[str],
    values_in_percent: bool,
) -> pd.DataFrame:
    if not selected_cols:
        return pd.DataFrame()

    frame = df[[date_col, *selected_cols]].copy()
    frame[date_col] = pd.to_datetime(frame[date_col], errors='coerce')
    frame[date_col] = frame[date_col].dt.normalize()
    for col in selected_cols:
        frame[col] = pd.to_numeric(frame[col], errors='coerce')
    frame = frame.dropna(subset=[date_col]).sort_values(date_col)
    frame = frame.drop_duplicates(subset=[date_col], keep='first')
    frame = frame.set_index(date_col)
    if values_in_percent:
        frame[selected_cols] = frame[selected_cols] / 100.0
    return frame[selected_cols]


def _format_range(index: pd.DatetimeIndex) -> str:
    if index.empty:
        return 'unknown'
    return f'{index.min():%Y-%m} to {index.max():%Y-%m}'


def _build_quality_rows(raw_df: pd.DataFrame, date_col: str, value_cols: list[str], label_type: str) -> tuple[list[dict[str, Any]], int, int]:
    if raw_df.empty or date_col not in raw_df.columns or not value_cols:
        return [], 0, 0

    parsed_dates = pd.to_datetime(raw_df[date_col], errors='coerce').dt.normalize()
    duplicate_date_count = int(parsed_dates.duplicated().sum())
    raw_row_count = int(len(raw_df))
    rows: list[dict[str, Any]] = []

    for column in value_cols:
        numeric = pd.to_numeric(raw_df[column], errors='coerce')
        panel = pd.DataFrame({'Date': parsed_dates, column: numeric})
        cleaned = panel.dropna(subset=['Date', column]).sort_values('Date')
        cleaned = cleaned.drop_duplicates(subset=['Date'], keep='first')
        cleaned_row_count = int(len(cleaned))
        missing_pct = float(0.0 if raw_row_count == 0 else numeric.isna().mean())
        rows.append(
            {
                'label_type': label_type,
                'name': column,
                'raw_rows': raw_row_count,
                'cleaned_rows': cleaned_row_count,
                'first_date': cleaned['Date'].min() if not cleaned.empty else pd.NaT,
                'last_date': cleaned['Date'].max() if not cleaned.empty else pd.NaT,
                'missing_pct': missing_pct,
                'duplicate_date_count': duplicate_date_count,
                'extreme_return_count': int((numeric.abs() > 5).sum()),
            }
        )

    extreme_count = int(sum(row['extreme_return_count'] for row in rows))
    return rows, duplicate_date_count, extreme_count


def align_fund_and_factor_returns(
    fund_returns_full: pd.DataFrame,
    factor_returns_full: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    exact_overlap = pd.concat([fund_returns_full, factor_returns_full], axis=1, join='inner').sort_index()
    exact_overlap = exact_overlap.loc[:, ~exact_overlap.columns.duplicated()]
    if not exact_overlap.empty:
        return exact_overlap, 'exact_date'

    fund_freq = detect_frequency(fund_returns_full.index)
    factor_freq = detect_frequency(factor_returns_full.index)
    if fund_freq == factor_freq and fund_freq in {'monthly', 'quarterly'}:
        freq_code = {'monthly': 'M', 'quarterly': 'Q'}[fund_freq]
        fund_period = fund_returns_full.copy()
        factor_period = factor_returns_full.copy()
        fund_period.index = fund_period.index.to_period(freq_code).to_timestamp(how='end').normalize()
        factor_period.index = factor_period.index.to_period(freq_code).to_timestamp(how='end').normalize()
        period_overlap = pd.concat([fund_period, factor_period], axis=1, join='inner').sort_index()
        period_overlap = period_overlap.loc[:, ~period_overlap.columns.duplicated()]
        if not period_overlap.empty:
            return period_overlap, f'period_{fund_freq}'

    return exact_overlap, 'none'


def build_data_source_metadata(
    mode: str,
    fund_returns_full: pd.DataFrame,
    factor_returns_full: pd.DataFrame,
    fund_factor_overlap: pd.DataFrame,
    alignment_method: str,
) -> dict[str, Any]:
    warnings: list[str] = []

    fund_start = fund_returns_full.index.min() if not fund_returns_full.empty else None
    fund_end = fund_returns_full.index.max() if not fund_returns_full.empty else None
    factor_start = factor_returns_full.index.min() if not factor_returns_full.empty else None
    factor_end = factor_returns_full.index.max() if not factor_returns_full.empty else None
    overlap_start = fund_factor_overlap.index.min() if not fund_factor_overlap.empty else None
    overlap_end = fund_factor_overlap.index.max() if not fund_factor_overlap.empty else None

    if fund_start is not None and factor_start is not None and factor_start > fund_start:
        warnings.append(
            f'Fund return history starts in {fund_start:%Y-%m}, but factor returns start in {factor_start:%Y-%m}. '
            f'Historical portfolio risk can use the full fund history. Factor model analytics will use the overlapping period from {factor_start:%Y-%m} onward.'
        )

    if fund_start is not None and factor_start is not None and overlap_start is None:
        warnings.append('No overlapping dates were found between fund returns and factor returns.')

    if alignment_method.startswith('period_'):
        warnings.append(f'No exact date overlap was found, so the upload was aligned using {alignment_method.replace("period_", "")}-period ends.')

    return {
        'mode': mode,
        'alignment_method': alignment_method,
        'fund_history_start': fund_start,
        'fund_history_end': fund_end,
        'factor_history_start': factor_start,
        'factor_history_end': factor_end,
        'overlap_start': overlap_start,
        'overlap_end': overlap_end,
        'fund_history_label': _format_range(fund_returns_full.index),
        'factor_history_label': _format_range(factor_returns_full.index),
        'overlap_label': _format_range(fund_factor_overlap.index),
        'warnings': warnings,
    }


def prepare_combined_wide_file(
    raw_df: pd.DataFrame,
    date_col: str,
    fund_cols: list[str],
    factor_cols: list[str],
    values_in_percent: bool,
    asset_weight_input: pd.Series | dict[str, float] | list[float] | None = None,
) -> NormalizedDataSource:
    fund_returns_full = _prepare_wide_panel(raw_df, date_col, fund_cols, values_in_percent)
    factor_returns_full = _prepare_wide_panel(raw_df, date_col, factor_cols, values_in_percent)
    overlap, alignment_method = align_fund_and_factor_returns(fund_returns_full, factor_returns_full)
    metadata = build_data_source_metadata('combined_wide', fund_returns_full, factor_returns_full, overlap, alignment_method)
    fund_quality_rows, fund_duplicate_date_count, fund_extreme_count = _build_quality_rows(raw_df, date_col, fund_cols, 'fund')
    factor_quality_rows, factor_duplicate_date_count, factor_extreme_count = _build_quality_rows(raw_df, date_col, factor_cols, 'factor')
    metadata.update(
        {
            'fund_quality_rows': fund_quality_rows,
            'factor_quality_rows': factor_quality_rows,
            'fund_duplicate_date_count': fund_duplicate_date_count,
            'factor_duplicate_date_count': factor_duplicate_date_count,
            'fund_extreme_return_count': fund_extreme_count,
            'factor_extreme_return_count': factor_extreme_count,
        }
    )

    return NormalizedDataSource(
        mode='combined_wide',
        merged_fund_returns=fund_returns_full.copy(),
        factor_returns=factor_returns_full.copy(),
        analysis_data=overlap.copy(),
        asset_cols=list(fund_cols),
        factor_cols=list(factor_cols),
        asset_weight_input=normalize_weights(list(fund_cols), asset_weight_input),
        data_source_metadata=metadata,
        weight_source_metadata={},
        fund_returns_full=fund_returns_full,
        factor_returns_full=factor_returns_full,
        fund_factor_overlap=overlap,
    )


def prepare_wide_fund_file_plus_factor_file(
    fund_df: pd.DataFrame,
    fund_date_col: str,
    fund_cols: list[str],
    factor_df: pd.DataFrame,
    factor_date_col: str,
    factor_cols: list[str],
    values_in_percent: bool,
    asset_weight_input: pd.Series | dict[str, float] | list[float] | None = None,
) -> NormalizedDataSource:
    fund_returns_full = _prepare_wide_panel(fund_df, fund_date_col, fund_cols, values_in_percent)
    factor_returns_full = _prepare_wide_panel(factor_df, factor_date_col, factor_cols, values_in_percent)
    overlap, alignment_method = align_fund_and_factor_returns(fund_returns_full, factor_returns_full)
    metadata = build_data_source_metadata('wide_fund_plus_factor', fund_returns_full, factor_returns_full, overlap, alignment_method)
    fund_quality_rows, fund_duplicate_date_count, fund_extreme_count = _build_quality_rows(fund_df, fund_date_col, fund_cols, 'fund')
    factor_quality_rows, factor_duplicate_date_count, factor_extreme_count = _build_quality_rows(factor_df, factor_date_col, factor_cols, 'factor')
    metadata.update(
        {
            'fund_quality_rows': fund_quality_rows,
            'factor_quality_rows': factor_quality_rows,
            'fund_duplicate_date_count': fund_duplicate_date_count,
            'factor_duplicate_date_count': factor_duplicate_date_count,
            'fund_extreme_return_count': fund_extreme_count,
            'factor_extreme_return_count': factor_extreme_count,
        }
    )

    return NormalizedDataSource(
        mode='wide_fund_plus_factor',
        merged_fund_returns=fund_returns_full.copy(),
        factor_returns=factor_returns_full.copy(),
        analysis_data=overlap.copy(),
        asset_cols=list(fund_cols),
        factor_cols=list(factor_cols),
        asset_weight_input=normalize_weights(list(fund_cols), asset_weight_input),
        data_source_metadata=metadata,
        weight_source_metadata={},
        fund_returns_full=fund_returns_full,
        factor_returns_full=factor_returns_full,
        fund_factor_overlap=overlap,
    )


def prepare_separate_fund_files(
    fund_file_specs: list[dict[str, Any]],
    factor_df: pd.DataFrame,
    factor_date_col: str,
    factor_cols: list[str],
    values_in_percent: bool,
) -> NormalizedDataSource:
    fund_streams: list[pd.DataFrame] = []
    asset_cols: list[str] = []
    fund_quality_rows: list[dict[str, Any]] = []
    fund_duplicate_date_count = 0
    fund_extreme_count = 0

    for spec in fund_file_specs:
        quality_rows, duplicate_date_count, extreme_count = _build_quality_rows(spec['raw_df'], spec['date_col'], [spec['return_col']], 'fund')
        if quality_rows:
            quality_rows[0]['name'] = spec['fund_name']
            quality_rows[0]['raw_rows'] = int(len(spec['raw_df']))
        fund_quality_rows.extend(
            [
                {
                    **row,
                    'name': spec['fund_name'],
                    'raw_rows': int(len(spec['raw_df'])),
                }
                for row in quality_rows
            ]
        )
        fund_duplicate_date_count += duplicate_date_count
        fund_extreme_count += extreme_count
        stream = prepare_return_stream(
            spec['raw_df'],
            date_col=spec['date_col'],
            return_col=spec['return_col'],
            stream_name=spec['fund_name'],
            values_in_percent=values_in_percent,
        )
        fund_streams.append(stream)
        asset_cols.append(spec['fund_name'])

    if fund_streams:
        fund_returns_full = pd.concat(fund_streams, axis=1, join='outer').sort_index()
        fund_returns_full = fund_returns_full.loc[:, ~fund_returns_full.columns.duplicated()]
    else:
        fund_returns_full = pd.DataFrame()

    factor_returns_full = prepare_factor_stream(
        factor_df,
        date_col=factor_date_col,
        factor_cols=factor_cols,
        values_in_percent=values_in_percent,
    )
    overlap, alignment_method = align_fund_and_factor_returns(fund_returns_full, factor_returns_full)
    metadata = build_data_source_metadata('separate_fund_files', fund_returns_full, factor_returns_full, overlap, alignment_method)
    factor_quality_rows, factor_duplicate_date_count, factor_extreme_count = _build_quality_rows(factor_df, factor_date_col, factor_cols, 'factor')
    metadata.update(
        {
            'fund_quality_rows': fund_quality_rows,
            'factor_quality_rows': factor_quality_rows,
            'fund_duplicate_date_count': fund_duplicate_date_count,
            'factor_duplicate_date_count': factor_duplicate_date_count,
            'fund_extreme_return_count': fund_extreme_count,
            'factor_extreme_return_count': factor_extreme_count,
        }
    )

    return NormalizedDataSource(
        mode='separate_fund_files',
        merged_fund_returns=fund_returns_full.copy(),
        factor_returns=factor_returns_full.copy(),
        analysis_data=overlap.copy(),
        asset_cols=asset_cols,
        factor_cols=list(factor_cols),
        asset_weight_input=normalize_weights(asset_cols, None),
        data_source_metadata=metadata,
        weight_source_metadata={},
        fund_returns_full=fund_returns_full,
        factor_returns_full=factor_returns_full,
        fund_factor_overlap=overlap,
    )