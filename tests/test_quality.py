from __future__ import annotations

import numpy as np
import pandas as pd

from riskplus_core.data_sources import prepare_combined_wide_file, prepare_separate_fund_files
from riskplus_core.engine import run_core_analysis
from riskplus_core.quality import build_data_quality_report


def test_data_quality_report_captures_wide_mode_counts() -> None:
    dates = pd.date_range('2020-01-31', periods=6, freq='ME')
    raw = pd.DataFrame(
        {
            'Date': dates,
            'FundA': [0.05, 0.02, np.nan, 0.03, 0.04, 0.01],
            'FundB': [0.04, 0.01, 0.02, 0.03, 0.02, 0.01],
            'Factor1': [0.01, 0.015, 0.012, 0.016, 0.013, 0.014],
            'Factor2': [0.0105, 0.0155, 0.0122, 0.0162, 0.0131, 0.0141],
        }
    )

    bundle = prepare_combined_wide_file(
        raw,
        'Date',
        ['FundA', 'FundB'],
        ['Factor1', 'Factor2'],
        values_in_percent=False,
        asset_weight_input=[0.6, 0.4],
    )
    results = run_core_analysis(bundle, rf_rate=0.02, confidence=0.95, num_sims=200, random_seed=12)
    report = build_data_quality_report(results)

    assert not report.overview.empty
    assert int(report.overview.loc[report.overview['metric'].eq('merged_overlap_rows'), 'value'].iloc[0]) == 6
    assert 'raw_rows' in report.fund_summary.columns
    assert 'cleaned_rows' in report.factor_summary.columns
    assert report.factor_correlations.shape[0] == 1


def test_data_quality_report_explains_rows_dropped_after_merge() -> None:
    fund_dates = pd.date_range('2020-01-31', periods=6, freq='ME')
    factor_dates = pd.date_range('2020-04-30', periods=3, freq='ME')

    fund_file_specs = [
        {
            'raw_df': pd.DataFrame({'Date': fund_dates, 'Return': [0.05, 0.02, 0.01, 0.03, 0.04, 0.01]}),
            'date_col': 'Date',
            'return_col': 'Return',
            'fund_name': 'Fund A',
        },
        {
            'raw_df': pd.DataFrame({'Date': fund_dates, 'Return': [0.04, 0.01, 0.02, 0.03, 0.02, 0.01]}),
            'date_col': 'Date',
            'return_col': 'Return',
            'fund_name': 'Fund B',
        },
    ]
    factor_df = pd.DataFrame(
        {
            'Date': factor_dates,
            'Factor1': [0.01, 0.02, 0.03],
        }
    )

    bundle = prepare_separate_fund_files(
        fund_file_specs,
        factor_df,
        'Date',
        ['Factor1'],
        values_in_percent=False,
    )
    results = run_core_analysis(bundle, rf_rate=0.02, confidence=0.95, num_sims=200, random_seed=12)
    report = build_data_quality_report(results)

    assert int(report.overview.loc[report.overview['metric'].eq('merged_overlap_rows'), 'value'].iloc[0]) == 3
    assert len(report.fund_summary) == 2
    assert len(report.factor_summary) == 1
    assert any('dropped after merging' in warning for warning in report.warnings)
