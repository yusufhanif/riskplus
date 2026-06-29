from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from riskplus_core.engine import run_core_analysis
from riskplus_core.data import build_portfolio_series
from riskplus_core.models import NormalizedDataSource
from riskplus_core.data_sources import prepare_wide_fund_file_plus_factor_file
from riskplus_core.risk import compute_historical_stats


def test_run_core_analysis_smoke() -> None:
    rng = np.random.default_rng(42)
    dates = pd.date_range('2020-01-31', periods=48, freq='ME')

    data = pd.DataFrame(
        {
            'FundA': rng.normal(0.01, 0.03, size=len(dates)),
            'FundB': rng.normal(0.008, 0.025, size=len(dates)),
            'FundC': rng.normal(0.012, 0.04, size=len(dates)),
            'Factor1': rng.normal(0.005, 0.02, size=len(dates)),
            'Factor2': rng.normal(0.004, 0.018, size=len(dates)),
            'Factor3': rng.normal(0.003, 0.015, size=len(dates)),
            'Factor4': rng.normal(0.002, 0.01, size=len(dates)),
        },
        index=dates,
    )

    results = run_core_analysis(
        data,
        asset_cols=['FundA', 'FundB', 'FundC'],
        factor_cols=['Factor1', 'Factor2', 'Factor3', 'Factor4'],
        asset_weight_input=[0.5, 0.3, 0.2],
        rf_rate=0.02,
        confidence=0.95,
        num_sims=1000,
        random_seed=7,
    )

    assert results.frequency in {'monthly', 'quarterly', 'unknown'}
    assert results.portfolio is not None
    assert results.asset_weights is not None
    assert results.factors is not None
    assert results.ols_results
    assert results.hist_stats
    assert results.simulated_returns is not None
    assert results.simulated_fund_returns is not None
    assert results.simulated_portfolio_returns is not None
    assert results.sim_stats
    assert results.mc_contribs
    assert results.sys_spec
    assert results.rb_etl is not None
    assert results.rb_stdev is not None
    assert results.factor_contrib
    assert results.bucket_exposures is not None

    assert set(results.ols_results.keys()) >= {'model', 'coef_table', 'r2', 'adj_r2'}
    assert set(results.mc_contribs.keys()) >= {'mc_vol', 'mc_etl', 'mc_etr'}
    assert set(results.sys_spec.keys()) >= {'systematic_var', 'specific_var', 'total_var'}
    assert len(results.simulated_portfolio_returns) == 1000
    assert len(results.mc_contribs['mc_vol']) == 3
    assert len(results.mc_contribs['mc_etl']) == 3
    assert len(results.rb_etl) == 3
    assert len(results.rb_stdev) == 3


def test_run_core_analysis_uses_detected_frequency_for_quarterly_data() -> None:
    rng = np.random.default_rng(123)
    dates = pd.date_range('2020-03-31', periods=16, freq='QE')

    data = pd.DataFrame(
        {
            'FundA': rng.normal(0.01, 0.03, size=len(dates)),
            'FundB': rng.normal(0.008, 0.025, size=len(dates)),
            'FundC': rng.normal(0.012, 0.04, size=len(dates)),
            'Factor1': rng.normal(0.005, 0.02, size=len(dates)),
            'Factor2': rng.normal(0.004, 0.018, size=len(dates)),
            'Factor3': rng.normal(0.003, 0.015, size=len(dates)),
            'Factor4': rng.normal(0.002, 0.01, size=len(dates)),
        },
        index=dates,
    )

    results = run_core_analysis(
        data,
        asset_cols=['FundA', 'FundB', 'FundC'],
        factor_cols=['Factor1', 'Factor2', 'Factor3', 'Factor4'],
        asset_weight_input=[0.5, 0.3, 0.2],
        rf_rate=0.02,
        confidence=0.95,
        num_sims=1000,
        random_seed=11,
    )

    assert results.frequency == 'quarterly'
    assert results.sim_stats['ann_mean'] != results.sim_stats['mean'] * 12
    assert np.isclose(results.sim_stats['ann_mean'], results.sim_stats['mean'] * 4, rtol=0.2)


def test_run_core_analysis_uses_full_fund_history_for_summary_metrics() -> None:
    fund_dates = pd.date_range('2020-01-31', periods=12, freq='ME')
    factor_dates = pd.date_range('2020-10-31', periods=3, freq='ME')

    fund_df = pd.DataFrame(
        {
            'Date': fund_dates,
            'FundA': [0.05] * 6 + [0.0] * 6,
            'FundB': [0.04] * 6 + [0.01] * 6,
        }
    )
    factor_df = pd.DataFrame(
        {
            'Date': factor_dates,
            'Factor1': [0.01, 0.02, 0.03],
        }
    )

    bundle = prepare_wide_fund_file_plus_factor_file(
        fund_df,
        'Date',
        ['FundA', 'FundB'],
        factor_df,
        'Date',
        ['Factor1'],
        values_in_percent=False,
        asset_weight_input=[0.6, 0.4],
    )

    results = run_core_analysis(bundle, rf_rate=0.02, confidence=0.95, num_sims=200, random_seed=5)

    full_portfolio, _ = build_portfolio_series(bundle.merged_fund_returns, ['FundA', 'FundB'], bundle.asset_weight_input)
    expected_hist = compute_historical_stats(full_portfolio, 'monthly', 0.02)

    assert results.hist_stats['obs_count'] == 12
    assert np.isclose(results.hist_stats['ann_mean'], expected_hist['ann_mean'])
    assert np.isclose(results.hist_stats['ann_vol'], expected_hist['ann_vol'])
    assert results.portfolio.index.min() == pd.Timestamp('2020-01-31')


def test_run_core_analysis_rejects_empty_overlap() -> None:
    fund_index = pd.date_range('2020-01-31', periods=3, freq='ME')
    factor_index = pd.date_range('2021-01-31', periods=3, freq='ME')

    bundle = NormalizedDataSource(
        mode='wide_fund_plus_factor',
        merged_fund_returns=pd.DataFrame({'FundA': [0.01, 0.02, 0.03]}, index=fund_index),
        factor_returns=pd.DataFrame({'Factor1': [0.01, 0.02, 0.03]}, index=factor_index),
        analysis_data=pd.DataFrame(index=pd.DatetimeIndex([], dtype='datetime64[ns]')),
        asset_cols=['FundA'],
        factor_cols=['Factor1'],
        asset_weight_input=pd.Series({'FundA': 1.0}),
        data_source_metadata={'mode': 'wide_fund_plus_factor', 'warnings': []},
        fund_returns_full=pd.DataFrame({'FundA': [0.01, 0.02, 0.03]}, index=fund_index),
        factor_returns_full=pd.DataFrame({'Factor1': [0.01, 0.02, 0.03]}, index=factor_index),
        fund_factor_overlap=pd.DataFrame(index=pd.DatetimeIndex([], dtype='datetime64[ns]')),
    )

    with pytest.raises(ValueError, match='No overlapping usable rows'):
        run_core_analysis(bundle, rf_rate=0.02, confidence=0.95, num_sims=1000, random_seed=7)


def test_wide_fund_and_factor_align_on_month_periods() -> None:
    fund_df = pd.DataFrame(
        {
            'Date': ['2020-01-01', '2020-02-01', '2020-03-01'],
            'FundA': [1.0, 2.0, 3.0],
        }
    )
    factor_df = pd.DataFrame(
        {
            'Date': ['2020-01-31', '2020-02-29', '2020-03-31'],
            'Factor1': [0.1, 0.2, 0.3],
        }
    )

    bundle = prepare_wide_fund_file_plus_factor_file(
        fund_df,
        'Date',
        ['FundA'],
        factor_df,
        'Date',
        ['Factor1'],
        values_in_percent=False,
    )

    assert bundle.data_source_metadata['alignment_method'] == 'period_monthly'
    assert len(bundle.analysis_data) == 3

    results = run_core_analysis(bundle, rf_rate=0.02, confidence=0.95, num_sims=1000, random_seed=7)

    assert len(results.portfolio) == 3
    assert results.ols_results