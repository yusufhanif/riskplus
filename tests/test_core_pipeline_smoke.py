from __future__ import annotations

import numpy as np
import pandas as pd

from riskplus_core.engine import run_core_analysis


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