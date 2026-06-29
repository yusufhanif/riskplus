from __future__ import annotations

from pathlib import Path

import pandas as pd

from riskplus_core.baseline import BASELINE_FIXTURE_DIR, extract_riskplus_baseline


def test_extract_riskplus_baseline_creates_expected_files() -> None:
    artifacts = extract_riskplus_baseline()

    expected_files = [
        'riskplus_settings.csv',
        'riskplus_historical_risk.csv',
        'riskplus_simulated_risk.csv',
        'riskplus_summary_data.csv',
        'riskplus_rb_etl.csv',
        'riskplus_rb_stdev.csv',
        'riskplus_factor_contribution.csv',
        'riskplus_factor_analysis.csv',
        'riskplus_factor_correlations.csv',
        'riskplus_asset_correlations.csv',
        'riskplus_37_fund_returns.csv',
    ]

    for file_name in expected_files:
        assert (BASELINE_FIXTURE_DIR / file_name).exists()

    assert not artifacts.settings.empty
    assert not artifacts.historical_risk.empty
    assert not artifacts.simulated_risk.empty
    assert not artifacts.rb_etl.empty
    assert not artifacts.rb_stdev.empty


def test_extracted_settings_and_returns_are_readable() -> None:
    settings = pd.read_csv(BASELINE_FIXTURE_DIR / 'riskplus_settings.csv')
    returns_37 = pd.read_csv(BASELINE_FIXTURE_DIR / 'riskplus_37_fund_returns.csv')

    settings_lookup = dict(zip(settings['setting'], settings['value']))
    assert settings_lookup['portfolio_name'] == 'Current 7.26'
    assert int(settings_lookup['number_of_assets']) == 37
    assert int(settings_lookup['confidence_level']) == 95
    assert int(settings_lookup['annualized_risk_free_rate']) == 4

    fund_cols = [column for column in returns_37.columns if column != 'Date']
    assert len(fund_cols) == 37
    assert returns_37['Date'].notna().all()
    assert pd.to_datetime(returns_37['Date']).min() == pd.Timestamp('2019-08-31')
    assert pd.to_datetime(returns_37['Date']).max() == pd.Timestamp('2026-04-30')