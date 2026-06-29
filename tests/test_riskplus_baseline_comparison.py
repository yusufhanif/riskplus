from __future__ import annotations

from pathlib import Path

import pandas as pd

from riskplus_core.baseline import BASELINE_FIXTURE_DIR, REPORT_DIR, extract_riskplus_baseline, load_factor_returns_sheet, run_baseline_comparison


def test_baseline_comparison_generates_reports() -> None:
    extract_riskplus_baseline()
    comparisons = run_baseline_comparison(num_sims=1000, random_seed=42)

    expected_reports = [
        REPORT_DIR / 'historical_risk_comparison.csv',
        REPORT_DIR / 'rb_etl_comparison.csv',
        REPORT_DIR / 'rb_stdev_comparison.csv',
        REPORT_DIR / 'factor_contribution_comparison.csv',
        REPORT_DIR / 'summary_metrics_comparison.csv',
        REPORT_DIR / 'baseline_comparison_summary.md',
    ]
    for report_path in expected_reports:
        assert report_path.exists()

    assert 'historical_risk_comparison' in comparisons
    assert 'rb_etl_comparison' in comparisons
    assert 'rb_stdev_comparison' in comparisons


def test_baseline_comparison_uses_37_fund_dataset() -> None:
    extract_riskplus_baseline()
    comparisons = run_baseline_comparison(num_sims=500, random_seed=7)

    rb_etl = comparisons['rb_etl_comparison']
    rb_stdev = comparisons['rb_stdev_comparison']

    assert len(rb_etl) == 37
    assert len(rb_stdev) == 37
    assert 'asset' in rb_etl.columns
    assert 'asset' in rb_stdev.columns


def test_factor_workbook_is_available_for_comparison() -> None:
    factor_returns = load_factor_returns_sheet()
    assert 'Date' in factor_returns.columns
    assert len([column for column in factor_returns.columns if column != 'Date']) > 0