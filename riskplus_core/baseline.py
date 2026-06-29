"""RiskPlus workbook baseline extraction and comparison helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data_sources import prepare_wide_fund_file_plus_factor_file
from .engine import run_core_analysis


BASELINE_FIXTURE_DIR = Path('tests/fixtures/riskplus_baseline')
REPORT_DIR = Path('reports/baseline_comparison')
DATA_DIR = Path(__file__).resolve().parent / 'data'
RISKPLUS_WORKBOOK = DATA_DIR / 'Current Risk Plus 7.26.xlsx'
MONTHLY_RETURNS_WORKBOOK = DATA_DIR / 'Monthly Portfolio Returns.xlsx'
FACTOR_RETURNS_WORKBOOK = DATA_DIR / 'pivotalpath_index_datax.xlsx'


@dataclass(slots=True)
class BaselineArtifacts:
    settings: pd.DataFrame
    historical_risk: pd.DataFrame
    simulated_risk: pd.DataFrame
    summary_data: pd.DataFrame
    rb_etl: pd.DataFrame
    rb_stdev: pd.DataFrame
    factor_contribution: pd.DataFrame
    factor_analysis: pd.DataFrame
    factor_correlations: pd.DataFrame
    asset_correlations: pd.DataFrame
    returns_37_fund: pd.DataFrame


def _ensure_directory(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)


def _clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.copy()
    cleaned = cleaned.dropna(axis=0, how='all')
    cleaned = cleaned.dropna(axis=1, how='all')
    return cleaned.reset_index(drop=True)


def _first_non_empty_value(row: pd.Series) -> Any:
    for value in row.iloc[1:]:
        if pd.notna(value):
            return value
    return pd.NA


def extract_riskplus_settings(workbook_path: Path) -> pd.DataFrame:
    frame = pd.read_excel(workbook_path, sheet_name='INDEX', header=None)
    settings_map = {
        'PORTFOLIO': 'portfolio_name',
        'PORTFOLIO VALUE': 'portfolio_value',
        'NUMBER OF ASSETS': 'number_of_assets',
        'BACKFILL': 'backfill',
        'FACTOR MODEL': 'factor_model',
        'DISTRIBUTION TYPE': 'distribution_type',
        'CORRELATIONS': 'correlation_method',
        'STRESS TESTS': 'stress_tests',
        'ANALYSIS PERIOD': 'analysis_period',
        'TIME WINDOW': 'time_window',
        'CONFIDENCE LEVEL': 'confidence_level',
        "RFR (ANNUALIZED)": 'annualized_risk_free_rate',
        'CURRENCY': 'currency',
    }

    records: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        label = row.iloc[0]
        if not isinstance(label, str):
            continue
        normalized_label = label.strip().upper()
        if normalized_label not in settings_map:
            continue
        value = _first_non_empty_value(row)
        records.append({'setting': settings_map[normalized_label], 'value': value})

    settings = pd.DataFrame(records)
    if not settings.empty:
        settings['value'] = settings['value'].apply(_coerce_setting_value)
    return settings


def _coerce_setting_value(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip().strip("'")
        if stripped.isdigit():
            return int(stripped)
        try:
            numeric = float(stripped)
            if numeric.is_integer():
                return int(numeric)
            return numeric
        except Exception:
            return stripped
    if isinstance(value, (int, float, np.integer, np.floating)):
        if float(value).is_integer():
            return int(value)
        return float(value)
    return value


def extract_monthly_returns(monthly_returns_workbook: Path) -> pd.DataFrame:
    frame = pd.read_excel(monthly_returns_workbook, sheet_name='Monthly Performance')
    frame['Date'] = pd.to_datetime(frame['Date'], errors='coerce')
    frame = frame.dropna(subset=['Date']).sort_values('Date')
    return frame.reset_index(drop=True)


def extract_factor_returns(factor_workbook: Path) -> pd.DataFrame:
    workbook = pd.ExcelFile(factor_workbook)
    sheet_name = 'index data' if 'index data' in workbook.sheet_names else workbook.sheet_names[0]
    frame = pd.read_excel(factor_workbook, sheet_name=sheet_name)
    if 'date' in frame.columns:
        frame = frame.rename(columns={'date': 'Date'})
    elif 'Date' not in frame.columns:
        first_col = frame.columns[0]
        frame = frame.rename(columns={first_col: 'Date'})
    frame['Date'] = pd.to_datetime(frame['Date'], errors='coerce')
    frame = frame.dropna(subset=['Date']).sort_values('Date')
    return frame.reset_index(drop=True)


def extract_sheet_frame(workbook_path: Path, sheet_name: str, header: int | None = None, usecols: tuple[int, int] | None = None) -> pd.DataFrame:
    frame = pd.read_excel(workbook_path, sheet_name=sheet_name, header=header)
    if usecols is not None:
        start, end = usecols
        frame = frame.iloc[:, start:end]
    frame = frame.dropna(axis=0, how='all')
    frame = frame.dropna(axis=1, how='all')
    return frame.reset_index(drop=True)


def extract_riskplus_baseline(
    riskplus_workbook: Path = RISKPLUS_WORKBOOK,
    monthly_returns_workbook: Path = MONTHLY_RETURNS_WORKBOOK,
    factor_workbook: Path = FACTOR_RETURNS_WORKBOOK,
    output_dir: Path = BASELINE_FIXTURE_DIR,
) -> BaselineArtifacts:
    _ensure_directory(output_dir)

    settings = extract_riskplus_settings(riskplus_workbook)
    settings.to_csv(output_dir / 'riskplus_settings.csv', index=False)

    monthly_returns = extract_monthly_returns(monthly_returns_workbook)
    monthly_returns.to_csv(output_dir / 'riskplus_37_fund_returns.csv', index=False)

    historical_risk = extract_sheet_frame(riskplus_workbook, 'HISTORICAL RISK', header=7)
    historical_risk.to_csv(output_dir / 'riskplus_historical_risk.csv', index=False)

    simulated_risk = extract_sheet_frame(riskplus_workbook, 'SIMULATED RISK', header=7)
    simulated_risk.to_csv(output_dir / 'riskplus_simulated_risk.csv', index=False)

    summary_data = extract_sheet_frame(riskplus_workbook, 'SUMMARY_DATA')
    summary_data.to_csv(output_dir / 'riskplus_summary_data.csv', index=False)

    rb_etl = extract_rb_sheet(riskplus_workbook, 'RB_ETL')
    rb_etl.to_csv(output_dir / 'riskplus_rb_etl.csv', index=False)

    rb_stdev = extract_rb_sheet(riskplus_workbook, 'RB_STDEV')
    rb_stdev.to_csv(output_dir / 'riskplus_rb_stdev.csv', index=False)

    factor_contribution = extract_sheet_frame(riskplus_workbook, 'FCTR_DATA')
    factor_contribution.to_csv(output_dir / 'riskplus_factor_contribution.csv', index=False)

    factor_analysis = extract_sheet_frame(riskplus_workbook, 'FACTOR ANALYSIS', header=7)
    factor_analysis.to_csv(output_dir / 'riskplus_factor_analysis.csv', index=False)

    factor_correlations = extract_sheet_frame(riskplus_workbook, 'FACTOR CORRELATIONS', header=8)
    factor_correlations.to_csv(output_dir / 'riskplus_factor_correlations.csv', index=False)

    asset_correlations = extract_sheet_frame(riskplus_workbook, 'ASSET CORRELATIONS', header=8)
    asset_correlations.to_csv(output_dir / 'riskplus_asset_correlations.csv', index=False)

    return BaselineArtifacts(
        settings=settings,
        historical_risk=historical_risk,
        simulated_risk=simulated_risk,
        summary_data=summary_data,
        rb_etl=rb_etl,
        rb_stdev=rb_stdev,
        factor_contribution=factor_contribution,
        factor_analysis=factor_analysis,
        factor_correlations=factor_correlations,
        asset_correlations=asset_correlations,
        returns_37_fund=monthly_returns,
    )


def extract_rb_sheet(workbook_path: Path, sheet_name: str) -> pd.DataFrame:
    raw = pd.read_excel(workbook_path, sheet_name=sheet_name, header=None)
    left = raw.iloc[1:, :6].copy()
    right = raw.iloc[1:, 7:].copy()

    left.columns = ['sort_order', 'portfolio_group', 'mean', 'mc_to_etl', 'implied_return', 'difference']
    right.columns = ['sort_order', 'asset', 'mean', 'mc_to_etl', 'implied_return', 'difference']

    left['section'] = 'group'
    right['section'] = 'asset'

    combined = pd.concat([left, right], axis=0, ignore_index=True)
    combined = combined.dropna(axis=0, how='all')
    return combined


def load_factor_returns_sheet(factor_workbook: Path = FACTOR_RETURNS_WORKBOOK) -> pd.DataFrame:
    return extract_factor_returns(factor_workbook)


def build_baseline_comparison(
    baseline_artifacts: BaselineArtifacts,
    factor_returns: pd.DataFrame,
    confidence: float,
    rf_rate: float,
    num_sims: int = 10_000,
    random_seed: int = 42,
) -> dict[str, pd.DataFrame]:
    asset_cols = [col for col in baseline_artifacts.returns_37_fund.columns if col != 'Date']
    fund_df = baseline_artifacts.returns_37_fund[['Date', *asset_cols]].copy()
    factor_cols = [col for col in factor_returns.columns if col != 'Date']

    bundle = prepare_wide_fund_file_plus_factor_file(
        fund_df,
        'Date',
        asset_cols,
        factor_returns,
        'Date',
        factor_cols,
        values_in_percent=False,
    )

    results = run_core_analysis(bundle, rf_rate=rf_rate, confidence=confidence, num_sims=num_sims, random_seed=random_seed)

    return {
        'historical_risk_comparison': compare_historical_risk(results.hist_stats, baseline_artifacts.historical_risk),
        'rb_etl_comparison': compare_risk_budgeting(results.rb_etl, baseline_artifacts.rb_etl, 'asset'),
        'rb_stdev_comparison': compare_risk_budgeting(results.rb_stdev, baseline_artifacts.rb_stdev, 'asset'),
        'factor_contribution_comparison': compare_factor_tables(results.factor_contrib, baseline_artifacts.factor_analysis),
        'summary_metrics_comparison': compare_summary_metrics(results, baseline_artifacts.settings),
    }


def compare_historical_risk(app_hist_stats: dict[str, float], baseline_table: pd.DataFrame) -> pd.DataFrame:
    portfolio_row = baseline_table[baseline_table.iloc[:, 0].astype(str).str.contains('Current 7.26', na=False)]
    if portfolio_row.empty:
        portfolio_row = baseline_table.iloc[[0]].copy()
    reference = portfolio_row.iloc[0]

    return pd.DataFrame(
        {
            'metric': ['ann_mean', 'ann_vol', 'sharpe', 'var', 'etl', 'rachev'],
            'app_value': [app_hist_stats['ann_mean'], app_hist_stats['ann_vol'], app_hist_stats['sharpe'], app_hist_stats['var'], app_hist_stats['etl'], app_hist_stats['rachev']],
            'baseline_value': [
                _get_numeric(reference, 'Ann.Mean'),
                _get_numeric(reference, 'Ann. StDev'),
                _get_numeric(reference, 'Sharpe'),
                np.nan,
                np.nan,
                np.nan,
            ],
        }
    )


def compare_risk_budgeting(app_table: pd.DataFrame, baseline_table: pd.DataFrame, section: str) -> pd.DataFrame:
    app = app_table.copy().rename(
        columns={
            'Asset': 'asset',
            'Weight (%)': 'app_weight_pct',
            'Mean Return (%)': 'app_mean_return_pct',
            'MC to Risk': 'app_mc_to_risk',
            'Implied Return (%)': 'app_implied_return_pct',
            'Difference (%)': 'app_difference_pct',
        }
    )

    baseline = baseline_table[baseline_table['section'] == section].copy()
    if 'asset' not in baseline.columns:
        if 'asset' in baseline.columns:
            pass
        elif 'portfolio_group' in baseline.columns:
            baseline = baseline.rename(columns={'portfolio_group': 'asset'})

    baseline = baseline.rename(
        columns={
            'mean': 'baseline_mean',
            'mc_to_etl': 'baseline_mc_to_risk',
            'implied_return': 'baseline_implied_return',
            'difference': 'baseline_difference',
        }
    )

    merged = app.merge(baseline, on='asset', how='left')
    keep_columns = [
        'asset',
        'app_weight_pct',
        'app_mean_return_pct',
        'app_mc_to_risk',
        'app_implied_return_pct',
        'app_difference_pct',
        'baseline_mean',
        'baseline_mc_to_risk',
        'baseline_implied_return',
        'baseline_difference',
    ]
    existing_columns = [column for column in keep_columns if column in merged.columns]
    return merged[existing_columns]


def compare_factor_tables(app_factor_contrib: dict[str, Any], baseline_factor_analysis: pd.DataFrame) -> pd.DataFrame:
    factor_names = app_factor_contrib.get('factor_names', [])
    app_betas = pd.Series(app_factor_contrib.get('betas', []), index=factor_names) if factor_names else pd.Series(dtype=float)
    rows = []
    for factor_name, beta in app_betas.items():
        rows.append({'factor': factor_name, 'app_beta': float(beta), 'baseline_beta': np.nan, 'abs_diff': np.nan})
    if not rows:
        rows.append({'factor': 'no_overlap', 'app_beta': np.nan, 'baseline_beta': np.nan, 'abs_diff': np.nan})
    return pd.DataFrame(rows)


def compare_summary_metrics(results: Any, settings: pd.DataFrame) -> pd.DataFrame:
    setting_lookup = {row['setting']: row['value'] for _, row in settings.iterrows()}
    return pd.DataFrame(
        {
            'metric': ['portfolio_name', 'number_of_assets', 'confidence_level', 'annualized_risk_free_rate', 'frequency'],
            'baseline_value': [
                setting_lookup.get('portfolio_name'),
                setting_lookup.get('number_of_assets'),
                setting_lookup.get('confidence_level'),
                setting_lookup.get('annualized_risk_free_rate'),
                setting_lookup.get('correlation_method'),
            ],
            'app_value': [
                results.portfolio.name if results.portfolio is not None else None,
                len(results.asset_weights) if results.asset_weights is not None else None,
                np.nan,
                np.nan,
                results.frequency,
            ],
        }
    )


def _get_numeric(row: pd.Series, column_name: str) -> float:
    for idx, value in row.items():
        if isinstance(idx, str) and idx.strip().lower() == column_name.strip().lower():
            try:
                return float(value)
            except Exception:
                return np.nan
    return np.nan


def write_comparison_reports(comparisons: dict[str, pd.DataFrame], output_dir: Path = REPORT_DIR) -> None:
    _ensure_directory(output_dir)
    for name, frame in comparisons.items():
        frame.to_csv(output_dir / f'{name}.csv', index=False)

    summary = [
        '# RiskPlus Baseline Comparison Summary',
        '',
        'This report is intentionally approximate. Exact matching is not expected until factor alignment, factor selection, EWMA covariance, asymmetric Student-t behavior, and the proprietary RiskPlus methodology choices are replicated.',
        '',
    ]
    for name, frame in comparisons.items():
        summary.append(f'- {name}: {len(frame)} rows written')
    (output_dir / 'baseline_comparison_summary.md').write_text('\n'.join(summary), encoding='utf-8')


def run_baseline_comparison(
    baseline_artifacts: BaselineArtifacts | None = None,
    factor_returns: pd.DataFrame | None = None,
    confidence: float | None = None,
    rf_rate: float | None = None,
    num_sims: int = 10_000,
    random_seed: int = 42,
) -> dict[str, pd.DataFrame]:
    # This comparison is intentionally approximate until factor alignment, factor selection,
    # EWMA covariance, asymmetric Student-t behavior, and other RiskPlus methodology choices are matched.
    if baseline_artifacts is None:
        baseline_artifacts = extract_riskplus_baseline()
    if factor_returns is None:
        factor_returns = load_factor_returns_sheet()

    settings = {row['setting']: row['value'] for _, row in baseline_artifacts.settings.iterrows()}
    confidence = confidence if confidence is not None else float(settings.get('confidence_level', 0.95)) / 100.0 if float(settings.get('confidence_level', 0.95)) > 1 else float(settings.get('confidence_level', 0.95))
    rf_rate = rf_rate if rf_rate is not None else float(settings.get('annualized_risk_free_rate', 0.04)) / 100.0 if float(settings.get('annualized_risk_free_rate', 0.04)) > 1 else float(settings.get('annualized_risk_free_rate', 0.04))

    comparisons = build_baseline_comparison(
        baseline_artifacts,
        factor_returns,
        confidence=confidence,
        rf_rate=rf_rate,
        num_sims=num_sims,
        random_seed=random_seed,
    )
    write_comparison_reports(comparisons)
    return comparisons
