from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from riskplus_core.reporting import (
    make_cumulative_growth_chart,
    make_exposure_chart,
    make_factor_bucket_chart,
    make_factor_bucket_table,
    make_factor_exposure_table,
    make_factor_level_contribution_table,
    make_historical_risk_table_detailed,
    make_historical_risk_table,
    make_historical_vs_simulated_table,
    make_portfolio_pie_chart,
    make_risk_budgeting_chart,
    make_settings_table,
    make_simulated_distribution_chart,
    make_simulated_risk_table_detailed,
    make_simulated_risk_table,
    make_tail_risk_table,
    make_traditional_measures_table,
)


def test_reporting_builders_return_pure_objects() -> None:
    hist_stats = {
        'obs_count': 48,
        'mean': 0.01,
        'ann_mean': 0.12,
        'vol': 0.02,
        'ann_vol': 0.10,
        'skew': 0.1,
        'xkurt': -0.2,
        'var': 0.03,
        'etl': 0.04,
        'etr': 0.05,
        'sharpe': 0.7,
        'starr': 0.5,
        'rachev': 1.2,
        'max_dd': -0.1,
        'best_period': 0.08,
        'worst_period': -0.06,
    }
    sim_stats = dict(hist_stats)
    factor_contrib = {
        'factor_names': ['Factor1', 'Factor2'],
        'betas': pd.Series([0.4, -0.2], index=['Factor1', 'Factor2']),
        'factor_mc_stdev': np.array([0.1, 0.05]),
        'bucket_mapping': {'Equity Risk': ['Factor1'], 'FX Risk': ['Factor2']},
    }
    ols_results = {
        'coef_table': pd.DataFrame(
            {'Coefficient': [0.1, 0.2, 0.3], 'tStat': [1.0, 2.0, 3.0], 'pValue': [0.1, 0.05, 0.01]},
            index=['const', 'Factor1', 'Factor2'],
        )
    }
    rb_table = pd.DataFrame(
        {
            'Asset': ['Portfolio'],
            'Weight (%)': [100.0],
            'Mean Return (%)': [12.0],
            'MC to Risk': [0.1],
            'Implied Return (%)': [10.0],
            'Difference (%)': [2.0],
            'Status': ['↑ Increase'],
        }
    )

    assert isinstance(make_settings_table('Report', 1_000_000, ['A'], ['F1'], 'Student-t', 'Classical', ('2020-01-01', '2020-12-31'), 48, 0.95, 0.02, 1000), pd.DataFrame)
    assert isinstance(make_historical_risk_table(hist_stats), pd.DataFrame)
    assert isinstance(make_simulated_risk_table(sim_stats, 1000), pd.DataFrame)
    assert isinstance(make_historical_vs_simulated_table(hist_stats, sim_stats), pd.DataFrame)
    assert isinstance(make_traditional_measures_table(hist_stats), pd.DataFrame)
    assert isinstance(make_tail_risk_table(hist_stats), pd.DataFrame)
    assert isinstance(make_factor_exposure_table(ols_results), pd.DataFrame)
    assert isinstance(make_factor_level_contribution_table(factor_contrib), pd.DataFrame)
    assert isinstance(make_factor_bucket_table(factor_contrib), pd.DataFrame)

    assert isinstance(make_portfolio_pie_chart(['A'], pd.Series([1.0], index=['A'])), go.Figure)
    assert isinstance(make_simulated_distribution_chart(pd.DataFrame({'Portfolio': [0.01, -0.02, 0.03]}), sim_stats), go.Figure)
    assert isinstance(make_cumulative_growth_chart(pd.Series([0.01, 0.02, -0.01], index=pd.date_range('2020-01-31', periods=3, freq='ME'))), go.Figure)
    assert isinstance(make_risk_budgeting_chart(rb_table, 'ETL'), go.Figure)
    assert isinstance(make_factor_bucket_chart(make_factor_bucket_table(factor_contrib)), go.Figure)
    assert isinstance(make_exposure_chart(pd.DataFrame({'Bucket': ['Equity Risk'], 'Exposure': [0.4]})), go.Figure)

    detailed_hist = make_historical_risk_table_detailed(
        hist_stats,
        {
            'FundA': dict(hist_stats),
            'FundB': dict(hist_stats),
        },
        pd.Series([0.6, 0.4], index=['FundA', 'FundB']),
    )
    assert isinstance(detailed_hist, pd.DataFrame)
    assert len(detailed_hist) == 3
    assert detailed_hist.iloc[0]['Name'] == 'Portfolio'

    detailed_sim = make_simulated_risk_table_detailed(
        sim_stats,
        {
            'FundA': dict(sim_stats),
            'FundB': dict(sim_stats),
        },
        pd.Series([0.6, 0.4], index=['FundA', 'FundB']),
        {'mc_vol': [0.1, 0.2], 'mc_etl': [0.3, 0.4], 'mc_etr': [0.5, 0.6]},
        {'pc_vol': [0.4, 0.6], 'pc_etl': [0.3, 0.7], 'pc_etr': [0.2, 0.8]},
    )
    assert isinstance(detailed_sim, pd.DataFrame)
    assert len(detailed_sim) == 3
    assert detailed_sim.iloc[0]['Name'] == 'Portfolio'