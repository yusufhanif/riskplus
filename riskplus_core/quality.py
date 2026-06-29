"""Data quality helpers for RiskPlus Streamlit."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .models import CoreAnalysisResults


@dataclass(slots=True)
class DataQualityReport:
    overview: pd.DataFrame
    fund_summary: pd.DataFrame
    factor_summary: pd.DataFrame
    factor_correlations: pd.DataFrame
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def validate_selected_panel(
    df: pd.DataFrame,
    date_col: str,
    value_cols: list[str],
    min_observations: int,
    max_missing_pct: float,
) -> tuple[list[str], list[str]]:
    """Validate a single selected panel without requiring both fund and factor columns."""
    errors: list[str] = []
    warnings: list[str] = []

    if not value_cols:
        errors.append('Select at least one return column.')
        return errors, warnings

    selected_cols = [date_col, *value_cols]
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

    numeric_frame = df[value_cols].apply(pd.to_numeric, errors='coerce')
    for col in value_cols:
        col_missing_pct = float(numeric_frame[col].isna().mean())
        if col_missing_pct > max_missing_pct:
            errors.append(f'{col}: {col_missing_pct:.1%} missing (limit: {max_missing_pct:.1%}).')

    candidate = pd.concat([date_parsed.rename('Date'), numeric_frame], axis=1).dropna()
    if len(candidate) < min_observations:
        errors.append(f'Only {len(candidate)} complete rows; need {min_observations}.')

    extreme_mask = numeric_frame.abs() > 5
    if extreme_mask.any().any():
        warnings.append('Extreme returns (>500%) detected. Check if data is scaled correctly.')

    if len(value_cols) > 1:
        corr_matrix = numeric_frame.corr()
        high_corr = (corr_matrix.abs() > 0.75) & (corr_matrix != 1.0)
        if high_corr.any().any():
            warnings.append('Some selected columns are highly correlated (>0.75). Consider removing redundant columns.')

    return errors, warnings


def _as_frame(rows: list[dict[str, Any]], columns: list[str]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(rows)
    for column in columns:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame[columns]


def _build_overview_rows(results: CoreAnalysisResults) -> list[dict[str, Any]]:
    metadata = results.data_source_metadata or {}
    fund_history = results.fund_returns_full if results.fund_returns_full is not None else pd.DataFrame()
    factor_history = results.factor_returns_full if results.factor_returns_full is not None else pd.DataFrame()
    overlap = results.fund_factor_overlap if results.fund_factor_overlap is not None else pd.DataFrame()
    factor_corr_pairs = _build_factor_correlation_pairs(results.factors)

    fund_raw_rows = int(sum(row.get('raw_rows', 0) for row in metadata.get('fund_quality_rows', [])))
    factor_raw_rows = int(sum(row.get('raw_rows', 0) for row in metadata.get('factor_quality_rows', [])))
    fund_cleaned_rows = int(sum(row.get('cleaned_rows', 0) for row in metadata.get('fund_quality_rows', [])))
    factor_cleaned_rows = int(sum(row.get('cleaned_rows', 0) for row in metadata.get('factor_quality_rows', [])))

    return [
        {'metric': 'data_mode', 'value': metadata.get('mode', 'unknown')},
        {'metric': 'alignment_method', 'value': metadata.get('alignment_method', 'unknown')},
        {'metric': 'detected_frequency', 'value': results.frequency},
        {'metric': 'fund_raw_rows_total', 'value': fund_raw_rows},
        {'metric': 'fund_cleaned_rows_total', 'value': fund_cleaned_rows},
        {'metric': 'factor_raw_rows_total', 'value': factor_raw_rows},
        {'metric': 'factor_cleaned_rows_total', 'value': factor_cleaned_rows},
        {'metric': 'merged_overlap_rows', 'value': int(len(overlap))},
        {'metric': 'duplicate_date_count_funds', 'value': int(metadata.get('fund_duplicate_date_count', 0))},
        {'metric': 'duplicate_date_count_factors', 'value': int(metadata.get('factor_duplicate_date_count', 0))},
        {'metric': 'extreme_return_count_funds', 'value': int(metadata.get('fund_extreme_return_count', 0))},
        {'metric': 'extreme_return_count_factors', 'value': int(metadata.get('factor_extreme_return_count', 0))},
        {'metric': 'fund_history_start', 'value': metadata.get('fund_history_label', 'unknown').split(' to ')[0] if metadata.get('fund_history_label') else 'unknown'},
        {'metric': 'fund_history_end', 'value': metadata.get('fund_history_label', 'unknown').split(' to ')[-1] if metadata.get('fund_history_label') else 'unknown'},
        {'metric': 'factor_history_start', 'value': metadata.get('factor_history_label', 'unknown').split(' to ')[0] if metadata.get('factor_history_label') else 'unknown'},
        {'metric': 'factor_history_end', 'value': metadata.get('factor_history_label', 'unknown').split(' to ')[-1] if metadata.get('factor_history_label') else 'unknown'},
        {'metric': 'overlap_start', 'value': metadata.get('overlap_label', 'unknown').split(' to ')[0] if metadata.get('overlap_label') else 'unknown'},
        {'metric': 'overlap_end', 'value': metadata.get('overlap_label', 'unknown').split(' to ')[-1] if metadata.get('overlap_label') else 'unknown'},
        {'metric': 'high_correlation_pairs', 'value': int(len(factor_corr_pairs))},
        {'metric': 'fund_panel_rows_in_memory', 'value': int(len(fund_history))},
        {'metric': 'factor_panel_rows_in_memory', 'value': int(len(factor_history))},
    ]


def _build_factor_correlation_pairs(factor_frame: pd.DataFrame | None, threshold: float = 0.75) -> pd.DataFrame:
    if factor_frame is None or factor_frame.empty or len(factor_frame.columns) < 2:
        return pd.DataFrame(columns=['factor_a', 'factor_b', 'correlation'])

    corr = factor_frame.corr(numeric_only=True)
    rows: list[dict[str, Any]] = []
    columns = list(corr.columns)
    for left_index, left_name in enumerate(columns):
        for right_name in columns[left_index + 1:]:
            value = float(corr.loc[left_name, right_name])
            if pd.notna(value) and abs(value) > threshold:
                rows.append({'factor_a': left_name, 'factor_b': right_name, 'correlation': value})
    return pd.DataFrame(rows, columns=['factor_a', 'factor_b', 'correlation'])


def build_data_quality_report(results: CoreAnalysisResults) -> DataQualityReport:
    metadata = results.data_source_metadata or {}

    fund_summary = _as_frame(
        metadata.get('fund_quality_rows', []),
        ['name', 'label_type', 'raw_rows', 'cleaned_rows', 'first_date', 'last_date', 'missing_pct', 'duplicate_date_count', 'extreme_return_count'],
    )
    factor_summary = _as_frame(
        metadata.get('factor_quality_rows', []),
        ['name', 'label_type', 'raw_rows', 'cleaned_rows', 'first_date', 'last_date', 'missing_pct', 'duplicate_date_count', 'extreme_return_count'],
    )

    overview = pd.DataFrame(_build_overview_rows(results))
    factor_correlations = _build_factor_correlation_pairs(results.factors)

    warnings = list(metadata.get('warnings', []))
    if results.fund_factor_overlap is not None and results.fund_returns_full is not None:
        if len(results.fund_factor_overlap) < len(results.fund_returns_full):
            warnings.append('Rows were dropped after merging because factor data did not overlap with the full fund history.')

    errors = list(metadata.get('errors', []))

    return DataQualityReport(
        overview=overview,
        fund_summary=fund_summary,
        factor_summary=factor_summary,
        factor_correlations=factor_correlations,
        warnings=warnings,
        errors=errors,
    )