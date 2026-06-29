"""
RiskPlus-Aligned Portfolio Risk Analytics in Streamlit

This application replicates the core workflow and methodologies of eVestment/BISAM RiskPlus
using open-source Python libraries. Key features include:

- Portfolio and fund-level risk analysis
- Student-t fat-tailed simulation for realistic tail risk metrics
- Historical and simulated risk metrics (VaR, ETL, ETR, Sharpe, STARR, Rachev)
- Systematic vs idiosyncratic risk decomposition via factor models
- Risk contribution analysis (percent and marginal)
- RiskPlus-aligned Streamlit interface and Excel export

Historical analysis uses actual historical returns; simulated analysis uses Student-t distribution.
Default parameters: 95% confidence level, 0.94 EWMA decay, 50,000 simulations.
"""

from __future__ import annotations

from io import BytesIO
import warnings
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from riskplus_core.constants import DEFAULT_CONFIDENCE
from riskplus_core.engine import run_core_analysis
from riskplus_core.data import MIN_OBSERVATIONS, infer_fund_name_from_file, read_uploaded_file
from riskplus_core.data_sources import (
    prepare_combined_wide_file,
    prepare_separate_fund_files,
    prepare_wide_fund_file_plus_factor_file,
)
from riskplus_core.models import NormalizedDataSource
from riskplus_core.quality import build_data_quality_report, validate_selected_panel
from riskplus_core.weights import (
    build_asset_weight_series,
    detect_weight_columns,
    match_weight_names_to_assets,
    normalize_portfolio_weights,
    prepare_weights_table,
    validate_portfolio_weights,
)
from riskplus_core.reporting import (
    make_cumulative_growth_chart,
    make_exposure_chart,
    make_factor_bucket_chart,
    make_factor_bucket_table,
    make_factor_exposure_table,
    make_factor_level_contribution_table,
    make_historical_risk_table,
    make_historical_vs_simulated_table,
    make_portfolio_pie_chart,
    make_risk_budgeting_chart,
    make_settings_table,
    make_simulated_distribution_chart,
    make_simulated_risk_table,
    make_tail_risk_table,
    make_traditional_measures_table,
)
from riskplus_ui.report_tabs import render_report_tabs
from riskplus_ui.workflow import AnalysisSettings, render_guided_workflow

warnings.filterwarnings("ignore", category=RuntimeWarning)

DEFAULT_EWMA_DECAY = 0.94


def render_index_tab(
    asset_cols: list[str],
    report_name: str,
    portfolio_value: float,
    factor_cols: list[str],
    dist_type: str,
    corr_method: str,
    selected_dates: tuple[object, object],
    observations: int,
    confidence: float,
    rf_rate: float,
    num_sims: int,
    fig_pie,
    data_source_metadata: dict[str, Any],
) -> None:
    st.header("Analysis Index & Settings")
    st.subheader("Calculation Settings")
    settings_table = make_settings_table(
        report_name,
        portfolio_value,
        asset_cols,
        factor_cols,
        dist_type,
        corr_method,
        selected_dates,
        observations,
        confidence,
        rf_rate,
        num_sims,
    )
    st.dataframe(settings_table, hide_index=True, use_container_width=True)

    st.caption(
        f"Portfolio history: {data_source_metadata.get('fund_history_label', 'unknown')} | "
        f"Factor overlap: {data_source_metadata.get('overlap_label', 'unknown')}"
    )

    st.subheader("Portfolio Composition")
    st.plotly_chart(fig_pie, use_container_width=True, key="index_portfolio_composition")

    st.subheader("Quick Links")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("📈 **SUMMARY**  \nKey metrics and risk overview")
    with col2:
        st.info("📊 **HISTORICAL RISK**  \nDetailed historical statistics")
    with col3:
        st.info("🎲 **SIMULATED RISK**  \nFat-tail risk metrics")


def render_data_quality_tab(data_quality_report, data_source_metadata: dict[str, Any]) -> None:
    st.header("Data Quality")
    st.caption(
        f"Source mode: {data_source_metadata.get('mode', 'unknown')} | "
        f"Fund history: {data_source_metadata.get('fund_history_label', 'unknown')} | "
        f"Factor overlap: {data_source_metadata.get('overlap_label', 'unknown')}"
    )

    overview_display = data_quality_report.overview.copy()
    if not overview_display.empty:
        st.subheader("Analysis Summary")
        st.dataframe(overview_display, hide_index=True, use_container_width=True)

    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Fund Data")
        if not data_quality_report.fund_summary.empty:
            fund_summary = data_quality_report.fund_summary.copy()
            for column in ['first_date', 'last_date']:
                if column in fund_summary.columns:
                    fund_summary[column] = pd.to_datetime(fund_summary[column], errors='coerce').dt.date
            if 'missing_pct' in fund_summary.columns:
                fund_summary['missing_pct'] = fund_summary['missing_pct'].map(lambda value: f'{float(value):.1%}' if pd.notna(value) else 'n/a')
            st.dataframe(fund_summary, hide_index=True, use_container_width=True)
        else:
            st.info('No fund quality summary was available.')

    with col_right:
        st.subheader("Factor Data")
        if not data_quality_report.factor_summary.empty:
            factor_summary = data_quality_report.factor_summary.copy()
            for column in ['first_date', 'last_date']:
                if column in factor_summary.columns:
                    factor_summary[column] = pd.to_datetime(factor_summary[column], errors='coerce').dt.date
            if 'missing_pct' in factor_summary.columns:
                factor_summary['missing_pct'] = factor_summary['missing_pct'].map(lambda value: f'{float(value):.1%}' if pd.notna(value) else 'n/a')
            st.dataframe(factor_summary, hide_index=True, use_container_width=True)
        else:
            st.info('No factor quality summary was available.')

    st.subheader("Rows Removed By Merge")
    def _overview_value(metric_name: str) -> int:
        if data_quality_report.overview.empty or 'metric' not in data_quality_report.overview.columns:
            return 0
        match = data_quality_report.overview.loc[data_quality_report.overview['metric'].eq(metric_name), 'value']
        if match.empty:
            return 0
        return int(match.iloc[0])

    merge_notes = pd.DataFrame(
        [
            {
                'metric': 'fund raw rows',
                'value': _overview_value('fund_raw_rows_total'),
            },
            {
                'metric': 'fund cleaned rows',
                'value': _overview_value('fund_cleaned_rows_total'),
            },
            {
                'metric': 'factor raw rows',
                'value': _overview_value('factor_raw_rows_total'),
            },
            {
                'metric': 'factor cleaned rows',
                'value': _overview_value('factor_cleaned_rows_total'),
            },
            {
                'metric': 'merged overlap rows used for factor analytics',
                'value': _overview_value('merged_overlap_rows'),
            },
        ]
    )
    st.dataframe(merge_notes, hide_index=True, use_container_width=True)

    if not data_quality_report.factor_correlations.empty:
        st.subheader("Factor Correlations Above 0.75")
        st.dataframe(data_quality_report.factor_correlations, hide_index=True, use_container_width=True)

    if data_quality_report.warnings:
        st.subheader("Warnings")
        for warning in data_quality_report.warnings:
            st.warning(warning)

    if data_quality_report.errors:
        st.subheader("Errors")
        for error in data_quality_report.errors:
            st.error(error)


def render_summary_tab(hist_stats: dict[str, float], sim_stats: dict[str, float], ols_results: dict[str, object], sys_spec: dict[str, Any], simulated_portfolio_returns: pd.Series, data_source_metadata: dict[str, Any]) -> None:
    st.header("Risk Summary")
    st.caption(
        f"Portfolio metrics use full fund history: {data_source_metadata.get('fund_history_label', 'unknown')}; "
        f"factor diagnostics use {data_source_metadata.get('overlap_label', 'unknown')}"
    )
    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
    with col_kpi1:
        st.metric("Ann. Return", f"{hist_stats['ann_mean']:.2%}")
    with col_kpi2:
        st.metric("Ann. Volatility", f"{hist_stats['ann_vol']:.2%}")
    with col_kpi3:
        st.metric("Sharpe Ratio", f"{hist_stats['sharpe']:.2f}")
    with col_kpi4:
        st.metric("VaR (95%)", f"{hist_stats['var']:.2%}")

    st.subheader("Simulated Return Distribution (Student-t)")
    st.plotly_chart(make_simulated_distribution_chart(simulated_portfolio_returns, sim_stats), use_container_width=True, key="summary_simulated_distribution")

    col_trad, col_tail = st.columns(2)
    with col_trad:
        st.subheader("Traditional Measures")
        st.table(make_traditional_measures_table(hist_stats))
    with col_tail:
        st.subheader("Tail Risk Measures")
        st.table(make_tail_risk_table(hist_stats))

    st.subheader("Factor Model Diagnostics")
    col_r2, col_sys = st.columns(2)
    with col_r2:
        st.metric("Model R²", f"{ols_results['r2']:.3f}")
    with col_sys:
        st.metric("Systematic Risk %", f"{sys_spec['systematic_pct']:.1%}")

    st.subheader("Factor Exposures & Significance")
    st.dataframe(make_factor_exposure_table(ols_results), use_container_width=True)


def render_historical_risk_tab(hist_stats: dict[str, float], portfolio: pd.Series, data_source_metadata: dict[str, Any]) -> None:
    st.header("Historical Risk Statistics")
    st.caption(f"Actual historical period statistics using {data_source_metadata.get('fund_history_label', 'unknown')}.")
    st.dataframe(make_historical_risk_table(hist_stats), hide_index=True, use_container_width=True)
    st.subheader("Cumulative Growth")
    st.plotly_chart(make_cumulative_growth_chart(portfolio), use_container_width=True, key="historical_cumulative_growth")


def render_simulated_risk_tab(hist_stats: dict[str, float], sim_stats: dict[str, float], num_sims: int, simulated_portfolio_returns: pd.Series, data_source_metadata: dict[str, Any]) -> None:
    st.header("Simulated Risk Statistics")
    st.caption(f"Based on {num_sims:,} Student-t Monte Carlo simulations using {data_source_metadata.get('fund_history_label', 'unknown')}.")
    st.dataframe(make_simulated_risk_table(sim_stats, num_sims), hide_index=True, use_container_width=True)
    st.subheader("Historical vs. Simulated Comparison")
    st.dataframe(make_historical_vs_simulated_table(hist_stats, sim_stats), hide_index=True, use_container_width=True)
    st.plotly_chart(make_simulated_distribution_chart(simulated_portfolio_returns, sim_stats), use_container_width=True, key="simulated_distribution_tab")


def render_risk_budgeting_tab(rb_table: pd.DataFrame, risk_label: str, data_source_metadata: dict[str, Any]) -> None:
    st.dataframe(rb_table, hide_index=True, use_container_width=True)
    st.caption(f"Risk budgeting uses the overlap period {data_source_metadata.get('overlap_label', 'unknown')}.")
    st.subheader("Risk-Return Analysis")
    st.plotly_chart(make_risk_budgeting_chart(rb_table, risk_label), use_container_width=True, key=f"risk_budgeting_{risk_label.lower()}")


def render_factor_contribution_tab(factor_contrib: dict[str, Any], show_factor_buckets: bool, data_source_metadata: dict[str, Any]) -> None:
    st.header("Factor Contribution to Portfolio Risk")
    st.caption(
        "Percentage Contribution to Risk by factor bucket shows how a given factor bucket contributes to the overall portfolio risk. "
        f"Factor analytics use {data_source_metadata.get('overlap_label', 'unknown')}."
    )
    col_sys, col_spec = st.columns(2)
    with col_sys:
        st.metric("Systematic Risk (StDev %)", f"{factor_contrib['systematic_stdev_pct']:.1%}")
    with col_spec:
        st.metric("Specific/Idiosyncratic Risk (%)", f"{factor_contrib['specific_stdev_pct']:.1%}")

    if show_factor_buckets and factor_contrib["bucket_mapping"]:
        st.subheader("Factor Risk by Bucket")
        bucket_df = make_factor_bucket_table(factor_contrib)
        st.dataframe(bucket_df, hide_index=True, use_container_width=True)
        st.plotly_chart(make_factor_bucket_chart(bucket_df), use_container_width=True, key="factor_bucket_chart")

    st.subheader("Factor-Level Contributions")
    st.dataframe(make_factor_level_contribution_table(factor_contrib), hide_index=True, use_container_width=True)


def render_exposure_tab(bucket_exposures: pd.DataFrame, data_source_metadata: dict[str, Any]) -> None:
    st.header("Exposure by Factor Bucket")
    st.caption(f"The sensitivity of the portfolio toward each market segment (factor basket) using {data_source_metadata.get('overlap_label', 'unknown')}.")
    if not bucket_exposures.empty:
        st.subheader("Portfolio Exposure by Bucket")
        st.dataframe(bucket_exposures, hide_index=True, use_container_width=True)
        st.plotly_chart(make_exposure_chart(bucket_exposures), use_container_width=True, key="factor_bucket_exposure")
    else:
        st.info("Not enough factor data for exposure decomposition. Ensure you have multiple factors selected.")


def _make_weight_preview_table(asset_cols: list[str], raw_weights: pd.Series, normalized_weights: pd.Series, status_message: str) -> pd.DataFrame:
    raw_values = raw_weights.reindex(asset_cols).fillna(0.0).to_numpy(dtype=float)
    normalized_values = normalized_weights.reindex(asset_cols).fillna(0.0).to_numpy(dtype=float)
    preview = pd.DataFrame({'Fund': asset_cols})
    preview['Weight (%)'] = raw_values * 100.0
    preview['Normalized Weight (%)'] = normalized_values * 100.0
    preview['Status'] = status_message
    return preview


def _read_weights_upload(uploaded_file, sheet_name: str | None = None) -> pd.DataFrame:
    if uploaded_file.name.lower().endswith('.csv'):
        return pd.read_csv(BytesIO(uploaded_file.getvalue()))
    if sheet_name is None:
        return pd.read_excel(BytesIO(uploaded_file.getvalue()))
    return pd.read_excel(BytesIO(uploaded_file.getvalue()), sheet_name=sheet_name)


def collect_portfolio_weights(asset_cols: list[str], base_weights: pd.Series | None = None) -> tuple[pd.Series, dict[str, Any]] | None:
    st.subheader("Portfolio Weights")
    if not asset_cols:
        st.info("Select fund columns before configuring weights.")
        return None

    weight_method = st.selectbox(
        "Weight input method",
        options=["Equal weight", "Manual editable table", "Upload weights file", "Read weights from workbook sheet"],
        index=0,
        key="weight_input_method",
    )

    initial_weights = base_weights if base_weights is not None and not base_weights.empty else pd.Series(1.0 / len(asset_cols), index=asset_cols, dtype=float)
    raw_weights = initial_weights.reindex(asset_cols).fillna(0.0).astype(float)
    metadata: dict[str, Any] = {
        'weight_input_method': weight_method,
        'weights_sum_before_normalization': float(raw_weights.sum()),
        'weights_sum_after_normalization': None,
        'missing_weight_funds': [],
        'extra_weight_rows': [],
        'normalization_applied': False,
        'negative_weight_detected': False,
        'zero_weight_funds': [],
    }

    if weight_method == "Equal weight":
        normalized_weights = normalize_portfolio_weights(pd.Series(1.0, index=asset_cols, dtype=float))
        metadata['weights_sum_after_normalization'] = float(normalized_weights.sum())
        metadata['normalization_applied'] = True
        preview = _make_weight_preview_table(asset_cols, normalized_weights, normalized_weights, 'Equal weight')
        st.dataframe(preview, hide_index=True, use_container_width=True)
        st.success("Weights successfully normalized to 100%.")
        if not st.button("Run Analysis", type="primary", key="run_analysis_weights_equal"):
            return None
        return normalized_weights, metadata

    if weight_method == "Manual editable table":
        editor_table = pd.DataFrame(
            {
                'Fund': asset_cols,
                'Weight (%)': (raw_weights * 100.0).round(6),
            }
        )
        edited_table = st.data_editor(
            editor_table,
            hide_index=True,
            num_rows='fixed',
            use_container_width=True,
            disabled=['Fund'],
            key='manual_weight_editor',
        )
        normalize_to_100 = st.checkbox('Normalize weights to 100%', value=True, key='manual_weight_normalize')
        entered_decimal = pd.Series(pd.to_numeric(edited_table['Weight (%)'], errors='coerce').fillna(0.0).values / 100.0, index=edited_table['Fund'])
        metadata['weights_sum_before_normalization'] = float(entered_decimal.sum())
        if (entered_decimal < 0).any():
            st.error('Negative weights are not allowed.')
            st.stop()
        if normalize_to_100:
            normalized_weights = normalize_portfolio_weights(entered_decimal)
            metadata['normalization_applied'] = True
            st.success('Weights successfully normalized to 100%.')
        else:
            total = float(entered_decimal.sum())
            if not np.isclose(total, 1.0, atol=0.005):
                st.error(f'Weights sum to {total * 100:.1f}%. Normalize to 100% or revise the weights.')
                st.stop()
            normalized_weights = entered_decimal.reindex(asset_cols).fillna(0.0).astype(float)
        metadata['weights_sum_after_normalization'] = float(normalized_weights.sum())
        preview = _make_weight_preview_table(asset_cols, entered_decimal, normalized_weights, 'Editable table')
        st.dataframe(preview, hide_index=True, use_container_width=True)
        if not st.button('Run Analysis', type='primary', key='run_analysis_weights_manual'):
            return None
        return normalized_weights.reindex(asset_cols).fillna(0.0).astype(float), metadata

    weights_upload = st.file_uploader('Upload weights file', type=['csv', 'xlsx', 'xls'], key='weights_upload_file')
    if weights_upload is None:
        return None

    sheet_name: str | None = None
    if weights_upload.name.lower().endswith(('.xlsx', '.xls')):
        workbook = pd.ExcelFile(BytesIO(weights_upload.getvalue()))
        sheet_name = st.selectbox('Weights sheet', options=workbook.sheet_names, index=0, key='weights_sheet_name')
    weights_df = _read_weights_upload(weights_upload, sheet_name=sheet_name)
    if weights_df.empty:
        st.error('Weights file is empty.')
        st.stop()

    detected = detect_weight_columns(weights_df)
    column_options = weights_df.columns.tolist()
    default_fund_col = detected['fund_column'] if detected['fund_column'] in column_options else column_options[0]
    default_weight_col = detected['weight_column'] if detected['weight_column'] in column_options else column_options[min(1, len(column_options) - 1)]
    fund_col = st.selectbox('Fund column', options=column_options, index=column_options.index(default_fund_col), key='weights_fund_column')
    weight_col = st.selectbox('Weight column', options=column_options, index=column_options.index(default_weight_col), key='weights_value_column')

    prepared_table = prepare_weights_table(weights_df, fund_col, weight_col)
    prepared_table = prepared_table.dropna(subset=['Fund'])
    if prepared_table.empty:
        st.error('No usable fund names were found in the weights file.')
        st.stop()

    if prepared_table['Fund'].duplicated().any():
        dupes = sorted(set(prepared_table.loc[prepared_table['Fund'].duplicated(), 'Fund'].tolist()))
        st.error(f"Duplicate fund names in weights file: {', '.join(dupes)}")
        st.stop()

    inferred_as_percent = bool(prepared_table['Weight'].abs().max() > 1.5)
    treat_as_percent = st.checkbox('Treat file weights as percentages', value=inferred_as_percent, key='weights_treat_percent')
    normalize_to_100 = st.checkbox('Normalize weights to 100%', value=True, key='weights_normalize_upload')

    mapping_table = match_weight_names_to_assets(prepared_table['Fund'].tolist(), asset_cols)
    edited_mapping = st.data_editor(
        mapping_table,
        hide_index=True,
        use_container_width=True,
        disabled=['Uploaded Fund Name', 'Matched Return Column', 'Match Confidence'],
        key='weights_match_editor',
    )

    fuzzy_rows = edited_mapping[(edited_mapping['Match Confidence'] < 1.0) & (~edited_mapping['User Confirmed'].astype(bool))]
    if not fuzzy_rows.empty:
        st.error('Confirm fuzzy matches before applying uploaded weights.')
        st.stop()

    confirmed_mapping = edited_mapping[['Uploaded Fund Name', 'Matched Return Column', 'User Confirmed']].copy()
    confirmed_mapping.loc[confirmed_mapping['Matched Return Column'].eq(''), 'Matched Return Column'] = confirmed_mapping['Uploaded Fund Name']
    mapping_lookup = dict(zip(confirmed_mapping['Uploaded Fund Name'], confirmed_mapping['Matched Return Column']))

    mapped_table = prepared_table.copy()
    mapped_table['Fund'] = mapped_table['Fund'].map(mapping_lookup).fillna(mapped_table['Fund'])
    mapped_table['Weight'] = pd.to_numeric(mapped_table['Weight'], errors='coerce').fillna(0.0)
    if treat_as_percent:
        mapped_table['Weight'] = mapped_table['Weight'] / 100.0

    raw_weight_series = build_asset_weight_series(asset_cols, mapped_table.rename(columns={'Fund': 'Fund', 'Weight': 'Weight'}), normalize=False)

    metadata['weights_sum_before_normalization'] = float(raw_weight_series.sum())

    validation_errors, validation_warnings = validate_portfolio_weights(raw_weight_series, asset_cols)
    for warning in validation_warnings:
        st.warning(warning)
    if validation_errors:
        for error in validation_errors:
            st.error(error)
        st.stop()

    if normalize_to_100:
        normalized_weights = normalize_portfolio_weights(raw_weight_series)
        metadata['normalization_applied'] = True
        st.success('Weights successfully normalized to 100%.')
    else:
        total = float(raw_weight_series.fillna(0.0).sum())
        if not np.isclose(total, 1.0, atol=0.005):
            st.error(f'Weights sum to {total * 100:.1f}%. Normalize to 100% or revise the weights.')
            st.stop()
        normalized_weights = raw_weight_series.reindex(asset_cols).fillna(0.0).astype(float)

    normalized_weights = normalized_weights.reindex(asset_cols).fillna(0.0).astype(float)
    metadata['weights_sum_after_normalization'] = float(normalized_weights.sum())
    metadata['negative_weight_detected'] = bool((raw_weight_series < 0).any())
    metadata['missing_weight_funds'] = [asset for asset in asset_cols if asset not in raw_weight_series.index]
    metadata['extra_weight_rows'] = [name for name in raw_weight_series.index.unique() if name not in asset_cols]
    metadata['zero_weight_funds'] = [asset for asset in asset_cols if float(normalized_weights.get(asset, 0.0)) == 0.0]

    if metadata['missing_weight_funds']:
        st.warning(f"{len(metadata['missing_weight_funds'])} selected funds were missing from the weights file and were assigned 0% weight.")
    if metadata['extra_weight_rows']:
        st.warning(f"{len(metadata['extra_weight_rows'])} rows in the weights file do not match any selected fund return column.")

    preview = _make_weight_preview_table(asset_cols, raw_weight_series, normalized_weights, 'Uploaded weights')
    st.dataframe(preview, hide_index=True, use_container_width=True)

    if not st.button('Run Analysis', type='primary', key='run_analysis_weights_upload'):
        return None

    return normalized_weights, metadata


def collect_data_source_bundle(values_in_percent: bool, max_missing_pct: float) -> NormalizedDataSource | None:
    st.subheader("1. Data Source Mode")
    data_source_mode = st.selectbox(
        "Data source mode",
        options=[
            "Combined wide file",
            "Wide fund file + separate factor file",
            "Separate fund files",
        ],
        index=2,
    )

    if data_source_mode == "Combined wide file":
        st.subheader("2. Combined Wide File")
        combined_upload = st.file_uploader("Upload combined wide file", type=["csv", "xlsx", "xls"])
        if combined_upload is None:
            return None

        combined_raw = read_uploaded_file(combined_upload.name, combined_upload.getvalue())
        combined_date_col = st.selectbox("Date column", options=combined_raw.columns.tolist(), index=0)
        combined_value_cols = [col for col in combined_raw.columns if col != combined_date_col]
        combined_fund_cols = st.multiselect("Fund return columns", options=combined_value_cols, default=[])
        combined_factor_cols = st.multiselect(
            "Factor return columns",
            options=[col for col in combined_value_cols if col not in combined_fund_cols],
            default=[],
        )

        errors, warnings_list = validate_selected_panel(
            combined_raw,
            combined_date_col,
            [*combined_fund_cols, *combined_factor_cols],
            MIN_OBSERVATIONS,
            max_missing_pct,
        )
        for warning in warnings_list:
            st.warning(warning)
        if errors:
            for error in errors:
                st.error(error)
            st.stop()

        return prepare_combined_wide_file(
            combined_raw,
            combined_date_col,
            combined_fund_cols,
            combined_factor_cols,
            values_in_percent,
        )

    if data_source_mode == "Wide fund file + separate factor file":
        st.subheader("2. Wide Fund File + Factor File")
        fund_upload = st.file_uploader("Upload wide fund file", type=["csv", "xlsx", "xls"], key="wide_fund_upload")
        factor_upload = st.file_uploader("Upload factor file", type=["csv", "xlsx", "xls"], key="wide_factor_upload")
        if fund_upload is None or factor_upload is None:
            return None

        fund_raw = read_uploaded_file(fund_upload.name, fund_upload.getvalue())
        factor_raw = read_uploaded_file(factor_upload.name, factor_upload.getvalue())

        fund_date_col = st.selectbox("Fund file date column", options=fund_raw.columns.tolist(), index=0)
        fund_cols = st.multiselect(
            "Fund return columns",
            options=[col for col in fund_raw.columns if col != fund_date_col],
            default=[],
        )
        factor_date_col = st.selectbox("Factor file date column", options=factor_raw.columns.tolist(), index=0)
        factor_cols = st.multiselect(
            "Factor return columns",
            options=[col for col in factor_raw.columns if col != factor_date_col],
            default=[],
        )

        fund_errors, fund_warnings = validate_selected_panel(
            fund_raw,
            fund_date_col,
            fund_cols,
            MIN_OBSERVATIONS,
            max_missing_pct,
        )
        factor_errors, factor_warnings = validate_selected_panel(
            factor_raw,
            factor_date_col,
            factor_cols,
            MIN_OBSERVATIONS,
            max_missing_pct,
        )
        for warning in [*fund_warnings, *factor_warnings]:
            st.warning(warning)
        if fund_errors or factor_errors:
            for error in [*fund_errors, *factor_errors]:
                st.error(error)
            st.stop()

        return prepare_wide_fund_file_plus_factor_file(
            fund_raw,
            fund_date_col,
            fund_cols,
            factor_raw,
            factor_date_col,
            factor_cols,
            values_in_percent,
        )

    st.subheader("2. Separate Fund Files")
    fund_uploads = st.file_uploader(
        "Upload one file per fund",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=True,
        key="separate_fund_uploads",
    )
    factor_upload = st.file_uploader("Upload factor returns file", type=["csv", "xlsx", "xls"], key="separate_factor_upload")
    if not fund_uploads or factor_upload is None:
        return None

    fund_file_specs: list[dict[str, Any]] = []
    for upload in fund_uploads:
        st.caption(upload.name)
        fund_raw = read_uploaded_file(upload.name, upload.getvalue())
        date_col = st.selectbox(
            f"Date column for {upload.name}",
            options=fund_raw.columns.tolist(),
            index=0,
            key=f"separate_date_{upload.name}",
        )
        numeric_guess = [
            col for col in fund_raw.columns
            if col != date_col and pd.to_numeric(fund_raw[col], errors='coerce').notna().mean() > 0.5
        ]
        if not numeric_guess:
            st.error(f"{upload.name} has no numeric return columns.")
            st.stop()

        return_col = st.selectbox(
            f"Return column for {upload.name}",
            options=numeric_guess,
            index=0,
            key=f"separate_return_{upload.name}",
        )
        fund_name = st.text_input(
            f"Fund name for {upload.name}",
            value=infer_fund_name_from_file(upload.name),
            key=f"separate_name_{upload.name}",
        )

        fund_errors, fund_warnings = validate_selected_panel(
            fund_raw,
            date_col,
            [return_col],
            MIN_OBSERVATIONS,
            max_missing_pct,
        )
        for warning in fund_warnings:
            st.warning(warning)
        if fund_errors:
            for error in fund_errors:
                st.error(error)
            st.stop()

        fund_file_specs.append(
            {
                'raw_df': fund_raw,
                'date_col': date_col,
                'return_col': return_col,
                'fund_name': fund_name,
            }
        )

    factor_raw = read_uploaded_file(factor_upload.name, factor_upload.getvalue())
    factor_date_col = st.selectbox("Factor file date column", options=factor_raw.columns.tolist(), index=0)
    factor_cols = st.multiselect(
        "Factor return columns",
        options=[col for col in factor_raw.columns if col != factor_date_col],
        default=[],
    )

    factor_errors, factor_warnings = validate_selected_panel(
        factor_raw,
        factor_date_col,
        factor_cols,
        MIN_OBSERVATIONS,
        max_missing_pct,
    )
    for warning in factor_warnings:
        st.warning(warning)
    if factor_errors:
        for error in factor_errors:
            st.error(error)
        st.stop()

    return prepare_separate_fund_files(
        fund_file_specs,
        factor_raw,
        factor_date_col,
        factor_cols,
        values_in_percent,
    )

def main() -> None:
    st.set_page_config(page_title="RiskPlus Streamlit", layout="wide")

    with st.sidebar:
        st.header("📊 Data & Analysis Setup")

        st.subheader("Analysis Settings")
        values_in_percent = st.checkbox("Values are in percent (5 = 5%)", value=False)
        max_missing_pct = st.slider("Max missing data %", 0.0, 50.0, 5.0, 1.0) / 100.0

        report_name = st.text_input("Report name", value="Portfolio Risk Analysis")
        portfolio_value = st.number_input("Portfolio value", value=1000000.0, step=100000.0)
        rf_rate = st.slider("Risk-free rate (annual %)", 0.0, 10.0, 2.0, 0.1) / 100.0

        confidence = st.slider("Confidence level", 0.85, 0.99, DEFAULT_CONFIDENCE, 0.01)
        num_sims = st.selectbox("Number of simulations", options=[10000, 50000, 100000], index=1)
        corr_method = st.selectbox("Covariance method", options=["Classical", "EWMA"], index=0)
        ewma_decay = st.slider("EWMA decay factor", 0.90, 0.99, DEFAULT_EWMA_DECAY, 0.01) if corr_method == "EWMA" else DEFAULT_EWMA_DECAY
        dist_type = st.selectbox("Distribution type", options=["Student-t", "Gaussian"], index=0)
        show_factor_buckets = st.checkbox("Show factor bucket analysis", value=True)
        display_mode = st.radio("Exposure display mode", options=["Weighted portfolio exposure", "Standalone beta"], index=0)

    settings = AnalysisSettings(
        report_name=report_name,
        portfolio_value=portfolio_value,
        rf_rate=rf_rate,
        confidence=confidence,
        num_sims=int(num_sims),
        corr_method=corr_method,
        ewma_decay=ewma_decay,
        dist_type=dist_type,
        show_factor_buckets=show_factor_buckets,
        display_mode=display_mode,
        values_in_percent=values_in_percent,
        max_missing_pct=max_missing_pct,
    )

    workflow_state = render_guided_workflow(settings)

    if not workflow_state.run_requested:
        st.info("Complete the workflow above, then run the analysis.")
        st.stop()

    if not workflow_state.can_run or workflow_state.mapping_result.bundle is None or workflow_state.weight_result.weights is None:
        st.error("Resolve the workflow errors before running the analysis.")
        st.stop()

    data_source_bundle = workflow_state.mapping_result.bundle
    data_source_bundle.asset_weight_input = workflow_state.weight_result.weights
    data_source_bundle.weight_source_metadata = workflow_state.weight_result.metadata

    try:
        core_results = run_core_analysis(
            data_source_bundle,
            rf_rate=rf_rate,
            confidence=confidence,
            num_sims=int(num_sims),
            random_seed=42,
        )
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    portfolio = core_results.portfolio
    selected_dates = (portfolio.index.min().date(), portfolio.index.max().date()) if portfolio is not None else (None, None)

    render_report_tabs(
        core_results,
        report_name,
        portfolio_value,
        dist_type,
        corr_method,
        selected_dates,
        confidence,
        rf_rate,
        num_sims,
        show_factor_buckets,
    )

    st.sidebar.divider()
    st.sidebar.caption("**RiskPlus Streamlit MVP**  \nStudent-t simulation, fat-tail risk metrics, systematic/idiosyncratic decomposition. Methodology caveats: Does not implement proprietary RiskPlus copula or stepwise regression.")


if __name__ == "__main__":
    main()