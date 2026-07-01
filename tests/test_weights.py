from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from riskplus_core.data_sources import (
    prepare_combined_wide_file,
    prepare_separate_fund_files,
    prepare_wide_fund_file_plus_factor_file,
)
from riskplus_core.engine import run_core_analysis
from riskplus_core.weights import (
    build_asset_weight_series,
    detect_weight_columns,
    match_weight_names_to_assets,
    normalize_portfolio_weights,
    prepare_weights_table,
    validate_portfolio_weights,
)


def test_equal_weights_37_funds_sum_to_one() -> None:
    asset_cols = [f'Fund_{index:02d}' for index in range(37)]
    weights = normalize_portfolio_weights(pd.Series(1.0, index=asset_cols, dtype=float))

    assert len(weights) == 37
    assert np.isclose(weights.sum(), 1.0)
    assert np.isclose(weights.iloc[0], 1.0 / 37.0)


def test_manual_table_percent_weights_normalize_correctly() -> None:
    weight_table = pd.DataFrame({'Fund': ['FundA', 'FundB', 'FundC'], 'Weight (%)': [5.0, 15.0, 80.0]})
    prepared = prepare_weights_table(weight_table, 'Fund', 'Weight (%)')
    entered_decimal = pd.Series(prepared['Weight'].values / 100.0, index=prepared['Fund'])
    normalized = normalize_portfolio_weights(entered_decimal)

    assert np.isclose(normalized.sum(), 1.0)
    assert np.isclose(normalized.loc['FundA'], 0.05)
    assert np.isclose(normalized.loc['FundB'], 0.15)
    assert np.isclose(normalized.loc['FundC'], 0.80)


def test_detect_weight_columns_prefers_synonyms() -> None:
    df = pd.DataFrame({'Asset Name': ['A'], 'Portfolio Weight': [0.25], 'Other': [1]})

    detected = detect_weight_columns(df)

    assert detected['fund_column'] == 'Asset Name'
    assert detected['weight_column'] == 'Portfolio Weight'


def test_uploaded_weights_exact_match_and_series_build() -> None:
    asset_cols = ['Fund A', 'Fund B', 'Fund C']
    table = pd.DataFrame({'Fund': ['Fund A', 'Fund B', 'Fund C'], 'Weight': [0.5, 0.3, 0.2]})

    built = build_asset_weight_series(asset_cols, table, normalize=False)

    assert list(built.index) == asset_cols
    assert np.allclose(built.values, [0.5, 0.3, 0.2])


def test_missing_and_extra_weights_are_reported() -> None:
    asset_cols = ['Fund A', 'Fund B', 'Fund C']
    weights = pd.Series([0.6, 0.4], index=['Fund A', 'Extra Fund'], dtype=float)

    errors, warnings = validate_portfolio_weights(weights, asset_cols)

    assert not errors
    assert any('missing from the weights file' in warning for warning in warnings)
    assert any('do not match any selected fund return column' in warning for warning in warnings)


def test_duplicate_negative_and_zero_total_weights_are_rejected() -> None:
    asset_cols = ['Fund A', 'Fund B']
    duplicate_weights = pd.Series([0.6, 0.4], index=['Fund A', 'Fund A'], dtype=float)
    negative_weights = pd.Series([0.6, -0.4], index=['Fund A', 'Fund B'], dtype=float)
    zero_weights = pd.Series([0.0, 0.0], index=['Fund A', 'Fund B'], dtype=float)

    duplicate_errors, _ = validate_portfolio_weights(duplicate_weights, asset_cols)
    negative_errors, _ = validate_portfolio_weights(negative_weights, asset_cols)
    zero_errors, _ = validate_portfolio_weights(zero_weights, asset_cols)

    assert any('Duplicate fund names' in error for error in duplicate_errors)
    assert any('Negative weights are not allowed' in error for error in negative_errors)
    assert any('Zero total weight detected' in error for error in zero_errors)


def test_workbook_sheet_selection_uses_selected_sheet(tmp_path: Path) -> None:
    workbook_path = tmp_path / 'weights.xlsx'
    with pd.ExcelWriter(workbook_path) as writer:
        pd.DataFrame({'NotWeights': [1]}).to_excel(writer, sheet_name='Returns', index=False)
        pd.DataFrame({'Fund': ['Fund A'], 'Weight': [0.5]}).to_excel(writer, sheet_name='Weights', index=False)

    default_sheet = pd.read_excel(workbook_path)
    selected_sheet = pd.read_excel(workbook_path, sheet_name='Weights')

    assert list(default_sheet.columns) == ['NotWeights']
    assert list(selected_sheet.columns) == ['Fund', 'Weight']


def _build_sample_combined_df() -> pd.DataFrame:
    dates = pd.date_range('2020-01-31', periods=12, freq='ME')
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        {
            'Date': dates,
            'Fund A': rng.normal(0.01, 0.02, size=len(dates)),
            'Fund B': rng.normal(0.02, 0.02, size=len(dates)),
            'Fund C': rng.normal(0.03, 0.02, size=len(dates)),
            'Factor 1': rng.normal(0.005, 0.01, size=len(dates)),
            'Factor 2': rng.normal(0.004, 0.01, size=len(dates)),
        }
    )


def _expected_weight_series(asset_cols: list[str]) -> pd.Series:
    return pd.Series({'Fund C': 0.2, 'Fund A': 0.5, 'Fund B': 0.3}).reindex(asset_cols).astype(float)


def test_all_upload_modes_accept_asset_weight_input() -> None:
    combined = _build_sample_combined_df()
    asset_cols = ['Fund A', 'Fund B', 'Fund C']
    factor_cols = ['Factor 1', 'Factor 2']

    combined_bundle = prepare_combined_wide_file(combined, 'Date', asset_cols, factor_cols, values_in_percent=False)
    combined_bundle.asset_weight_input = _expected_weight_series(asset_cols)
    combined_results = run_core_analysis(combined_bundle, rf_rate=0.02, confidence=0.95, num_sims=250, random_seed=3)

    wide_fund_df = combined[['Date', *asset_cols]].copy()
    factor_df = combined[['Date', *factor_cols]].copy()
    wide_bundle = prepare_wide_fund_file_plus_factor_file(
        wide_fund_df,
        'Date',
        asset_cols,
        factor_df,
        'Date',
        factor_cols,
        values_in_percent=False,
    )
    wide_bundle.asset_weight_input = _expected_weight_series(asset_cols)
    wide_results = run_core_analysis(wide_bundle, rf_rate=0.02, confidence=0.95, num_sims=250, random_seed=3)

    fund_file_specs = [
        {
            'raw_df': pd.DataFrame({'Date': combined['Date'], 'Return': combined['Fund A']}),
            'date_col': 'Date',
            'return_col': 'Return',
            'fund_name': 'Fund A',
        },
        {
            'raw_df': pd.DataFrame({'Date': combined['Date'], 'Return': combined['Fund B']}),
            'date_col': 'Date',
            'return_col': 'Return',
            'fund_name': 'Fund B',
        },
        {
            'raw_df': pd.DataFrame({'Date': combined['Date'], 'Return': combined['Fund C']}),
            'date_col': 'Date',
            'return_col': 'Return',
            'fund_name': 'Fund C',
        },
    ]
    separate_bundle = prepare_separate_fund_files(fund_file_specs, factor_df, 'Date', factor_cols, values_in_percent=False)
    separate_bundle.asset_weight_input = _expected_weight_series(asset_cols)
    separate_results = run_core_analysis(separate_bundle, rf_rate=0.02, confidence=0.95, num_sims=250, random_seed=3)

    for results in [combined_results, wide_results, separate_results]:
        assert results.asset_weights is not None
        assert list(results.asset_weights.index) == asset_cols
        assert np.allclose(results.asset_weights.values, [0.5, 0.3, 0.2])


def test_large_portfolio_37_fund_risk_budgeting_shape() -> None:
    dates = pd.date_range('2020-01-31', periods=24, freq='ME')
    rng = np.random.default_rng(11)
    asset_cols = [f'Fund {index:02d}' for index in range(37)]
    factor_cols = ['Factor 1', 'Factor 2']

    combined = pd.DataFrame({'Date': dates})
    for col in asset_cols:
        combined[col] = rng.normal(0.01, 0.02, size=len(dates))
    for col in factor_cols:
        combined[col] = rng.normal(0.005, 0.01, size=len(dates))

    bundle = prepare_combined_wide_file(combined, 'Date', asset_cols, factor_cols, values_in_percent=False)
    bundle.asset_weight_input = normalize_portfolio_weights(pd.Series(1.0, index=asset_cols, dtype=float))

    results = run_core_analysis(bundle, rf_rate=0.02, confidence=0.95, num_sims=200, random_seed=9)

    assert len(results.rb_etl) == 37
    assert len(results.asset_weights) == 37
    assert len(results.mc_contribs['mc_vol']) == 37
    assert len(results.mc_contribs['mc_etl']) == 37
    assert len(results.mc_contribs['mc_etr']) == 37
    assert results.rb_etl['Asset'].tolist() == asset_cols
    assert results.rb_stdev['Asset'].tolist() == asset_cols