"""Report tab renderers for the RiskPlus Streamlit app."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from riskplus_core.models import CoreAnalysisResults
from riskplus_core.explanations import (
    explain_data_quality,
    explain_factor_bucket_exposure,
    explain_factor_model_section,
    explain_historical_risk_metric,
    explain_risk_budgeting_etl,
    explain_risk_budgeting_stdev,
    explain_simulated_risk_metric,
    explain_systematic_specific_risk,
)
from riskplus_core.quality import build_data_quality_report
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
        st.info("Portfolio summary and headline metrics")
    with col2:
        st.info("Historical risk and tail statistics")
    with col3:
        st.info("Simulated risk and scenario outputs")


def render_data_quality_tab(data_quality_report, data_source_metadata: dict[str, Any]) -> None:
    st.header("Data Quality")
    with st.expander("Explain this result", expanded=False):
        st.caption(explain_data_quality())
    st.caption(
        f"Source mode: {data_source_metadata.get('mode', 'unknown')} | "
        f"Fund history: {data_source_metadata.get('fund_history_label', 'unknown')} | "
        f"Factor overlap: {data_source_metadata.get('overlap_label', 'unknown')}"
    )

    overview_display = data_quality_report.overview.copy()
    if not overview_display.empty:
        if 'value' in overview_display.columns:
            overview_display['value'] = overview_display['value'].map(lambda value: '' if pd.isna(value) else str(value))
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
                fund_summary['missing_pct'] = fund_summary['missing_pct'].map(
                    lambda value: f'{float(value):.1%}' if pd.notna(value) else 'n/a'
                )
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
                factor_summary['missing_pct'] = factor_summary['missing_pct'].map(
                    lambda value: f'{float(value):.1%}' if pd.notna(value) else 'n/a'
                )
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
            {'metric': 'fund raw rows', 'value': _overview_value('fund_raw_rows_total')},
            {'metric': 'fund cleaned rows', 'value': _overview_value('fund_cleaned_rows_total')},
            {'metric': 'factor raw rows', 'value': _overview_value('factor_raw_rows_total')},
            {'metric': 'factor cleaned rows', 'value': _overview_value('factor_cleaned_rows_total')},
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


def render_summary_tab(
    hist_stats: dict[str, float],
    sim_stats: dict[str, float],
    ols_results: dict[str, object],
    sys_spec: dict[str, Any],
    simulated_portfolio_returns: pd.Series,
    data_source_metadata: dict[str, Any],
) -> None:
    st.header("Risk Summary")
    with st.expander("Explain this result", expanded=False):
        st.caption(explain_factor_model_section())
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
    st.plotly_chart(
        make_simulated_distribution_chart(simulated_portfolio_returns, sim_stats),
        use_container_width=True,
        key="summary_simulated_distribution",
    )

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
    with st.expander("Explain this result", expanded=False):
        st.caption(explain_historical_risk_metric('ann_mean'))
    st.caption(f"Actual historical period statistics using {data_source_metadata.get('fund_history_label', 'unknown')}.")
    st.dataframe(make_historical_risk_table(hist_stats), hide_index=True, use_container_width=True)
    st.subheader("Cumulative Growth")
    st.plotly_chart(make_cumulative_growth_chart(portfolio), use_container_width=True, key="historical_cumulative_growth")


def render_simulated_risk_tab(
    hist_stats: dict[str, float],
    sim_stats: dict[str, float],
    num_sims: int,
    simulated_portfolio_returns: pd.Series,
    data_source_metadata: dict[str, Any],
) -> None:
    st.header("Simulated Risk Statistics")
    with st.expander("Explain this result", expanded=False):
        st.caption(explain_simulated_risk_metric('var'))
    st.caption(f"Based on {num_sims:,} Student-t Monte Carlo simulations using {data_source_metadata.get('fund_history_label', 'unknown')}.")
    st.dataframe(make_simulated_risk_table(sim_stats, num_sims), hide_index=True, use_container_width=True)
    st.subheader("Historical vs. Simulated Comparison")
    st.dataframe(make_historical_vs_simulated_table(hist_stats, sim_stats), hide_index=True, use_container_width=True)
    st.plotly_chart(
        make_simulated_distribution_chart(simulated_portfolio_returns, sim_stats),
        use_container_width=True,
        key="simulated_distribution_tab",
    )


def render_risk_budgeting_tab(rb_table: pd.DataFrame, risk_label: str, data_source_metadata: dict[str, Any]) -> None:
    st.dataframe(rb_table, hide_index=True, use_container_width=True)
    st.caption(f"Risk budgeting uses the overlap period {data_source_metadata.get('overlap_label', 'unknown')}.")
    with st.expander("Explain this result", expanded=False):
        if risk_label == 'ETL':
            st.caption(explain_risk_budgeting_etl())
        else:
            st.caption(explain_risk_budgeting_stdev())
    st.subheader("Risk-Return Analysis")
    st.plotly_chart(make_risk_budgeting_chart(rb_table, risk_label), use_container_width=True, key=f"risk_budgeting_{risk_label.lower()}")


def render_factor_contribution_tab(factor_contrib: dict[str, Any], show_factor_buckets: bool, data_source_metadata: dict[str, Any]) -> None:
    st.header("Factor Contribution to Portfolio Risk")
    with st.expander("Explain this result", expanded=False):
        st.caption(explain_systematic_specific_risk())
    st.caption(
        "Percentage Contribution to Risk by factor bucket shows how a given factor bucket contributes to the overall portfolio risk. "
        f"Factor analytics use {data_source_metadata.get('overlap_label', 'unknown')}.")
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
    with st.expander("Explain this result", expanded=False):
        st.caption(explain_factor_bucket_exposure())
    st.caption(f"The sensitivity of the portfolio toward each market segment (factor basket) using {data_source_metadata.get('overlap_label', 'unknown')}.")
    if not bucket_exposures.empty:
        st.subheader("Portfolio Exposure by Bucket")
        st.dataframe(bucket_exposures, hide_index=True, use_container_width=True)
        st.plotly_chart(make_exposure_chart(bucket_exposures), use_container_width=True, key="factor_bucket_exposure")
    else:
        st.info("Not enough factor data for exposure decomposition. Ensure you have multiple factors selected.")


def render_report_tabs(
    core_results: CoreAnalysisResults,
    report_name: str,
    portfolio_value: float,
    dist_type: str,
    corr_method: str,
    selected_dates: tuple[object, object],
    confidence: float,
    rf_rate: float,
    num_sims: int,
    show_factor_buckets: bool,
) -> None:
    data_source_metadata = core_results.data_source_metadata
    asset_cols = core_results.asset_weights.index.tolist() if core_results.asset_weights is not None else []
    factor_cols = list(core_results.factors.columns) if core_results.factors is not None else []
    portfolio = core_results.portfolio if core_results.portfolio is not None else pd.Series(dtype=float)
    asset_weights = core_results.asset_weights if core_results.asset_weights is not None else pd.Series(dtype=float)
    hist_stats = core_results.hist_stats
    sim_stats = core_results.sim_stats
    ols_results = core_results.ols_results
    sys_spec = core_results.sys_spec
    simulated_portfolio_returns = (
        core_results.simulated_portfolio_returns if core_results.simulated_portfolio_returns is not None else pd.Series(dtype=float)
    )
    data_quality_report = build_data_quality_report(core_results)

    tabs = st.tabs([
        "INDEX",
        "DATA QUALITY",
        "SUMMARY",
        "HISTORICAL RISK",
        "SIMULATED RISK",
        "RISK BUDGETING (ETL)",
        "RISK BUDGETING (StDev)",
        "FACTOR CONTRIBUTION",
        "EXPOSURE BY FACTOR BUCKET",
    ])

    with tabs[0]:
        render_index_tab(
            asset_cols,
            report_name,
            portfolio_value,
            factor_cols,
            dist_type,
            corr_method,
            selected_dates,
            len(portfolio),
            confidence,
            rf_rate,
            num_sims,
            make_portfolio_pie_chart(asset_cols, asset_weights),
            data_source_metadata,
        )

    with tabs[1]:
        render_data_quality_tab(data_quality_report, data_source_metadata)

    with tabs[2]:
        render_summary_tab(hist_stats, sim_stats, ols_results, sys_spec, simulated_portfolio_returns, data_source_metadata)

    with tabs[3]:
        render_historical_risk_tab(hist_stats, portfolio, data_source_metadata)

    with tabs[4]:
        render_simulated_risk_tab(hist_stats, sim_stats, num_sims, simulated_portfolio_returns, data_source_metadata)

    with tabs[5]:
        st.header("Risk Budgeting by ETL")
        st.caption(
            "Implied Return is the return an asset must deliver to justify its contribution to portfolio ETL. "
            "Assets with actual returns above the implied line may justify increased weight; below may suggest reduction."
        )
        render_risk_budgeting_tab(core_results.rb_etl, "ETL", data_source_metadata)

    with tabs[6]:
        st.header("Risk Budgeting by Standard Deviation")
        st.caption("Implied Return is the return an asset must deliver to justify its contribution to portfolio standard deviation.")
        render_risk_budgeting_tab(core_results.rb_stdev, "StDev", data_source_metadata)

    with tabs[7]:
        render_factor_contribution_tab(core_results.factor_contrib, show_factor_buckets, data_source_metadata)

    with tabs[8]:
        render_exposure_tab(core_results.bucket_exposures, data_source_metadata)
