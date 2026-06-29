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

import warnings
from typing import Any

import pandas as pd
import streamlit as st
from riskplus_core.constants import DEFAULT_CONFIDENCE
from riskplus_core.engine import run_core_analysis
from riskplus_core.data import (
    MIN_OBSERVATIONS,
    infer_fund_name_from_file,
    merge_analysis_frames,
    prepare_factor_stream,
    prepare_return_stream,
    prepare_analysis_data,
    read_uploaded_file,
    validate_raw_data,
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

    st.subheader("Portfolio Composition")
    st.plotly_chart(fig_pie, use_container_width=True)

    st.subheader("Quick Links")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("📈 **SUMMARY**  \nKey metrics and risk overview")
    with col2:
        st.info("📊 **HISTORICAL RISK**  \nDetailed historical statistics")
    with col3:
        st.info("🎲 **SIMULATED RISK**  \nFat-tail risk metrics")


def render_summary_tab(hist_stats: dict[str, float], sim_stats: dict[str, float], ols_results: dict[str, object], sys_spec: dict[str, Any], simulated_returns: pd.DataFrame) -> None:
    st.header("Risk Summary")
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
    st.plotly_chart(make_simulated_distribution_chart(simulated_returns, sim_stats), use_container_width=True)

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


def render_historical_risk_tab(hist_stats: dict[str, float], portfolio: pd.Series) -> None:
    st.header("Historical Risk Statistics")
    st.caption("Actual historical period statistics")
    st.dataframe(make_historical_risk_table(hist_stats), hide_index=True, use_container_width=True)
    st.subheader("Cumulative Growth")
    st.plotly_chart(make_cumulative_growth_chart(portfolio), use_container_width=True)


def render_simulated_risk_tab(hist_stats: dict[str, float], sim_stats: dict[str, float], num_sims: int) -> None:
    st.header("Simulated Risk Statistics")
    st.caption(f"Based on {num_sims:,} Student-t Monte Carlo simulations")
    st.dataframe(make_simulated_risk_table(sim_stats, num_sims), hide_index=True, use_container_width=True)
    st.subheader("Historical vs. Simulated Comparison")
    st.dataframe(make_historical_vs_simulated_table(hist_stats, sim_stats), hide_index=True, use_container_width=True)


def render_risk_budgeting_tab(rb_table: pd.DataFrame, risk_label: str) -> None:
    st.dataframe(rb_table, hide_index=True, use_container_width=True)
    st.subheader("Risk-Return Analysis")
    st.plotly_chart(make_risk_budgeting_chart(rb_table, risk_label), use_container_width=True)


def render_factor_contribution_tab(factor_contrib: dict[str, Any], show_factor_buckets: bool) -> None:
    st.header("Factor Contribution to Portfolio Risk")
    st.caption("Percentage Contribution to Risk by factor bucket shows how a given factor bucket contributes to the overall portfolio risk.")
    col_sys, col_spec = st.columns(2)
    with col_sys:
        st.metric("Systematic Risk (StDev %)", f"{factor_contrib['systematic_stdev_pct']:.1%}")
    with col_spec:
        st.metric("Specific/Idiosyncratic Risk (%)", f"{factor_contrib['specific_stdev_pct']:.1%}")

    if show_factor_buckets and factor_contrib["bucket_mapping"]:
        st.subheader("Factor Risk by Bucket")
        bucket_df = make_factor_bucket_table(factor_contrib)
        st.dataframe(bucket_df, hide_index=True, use_container_width=True)
        st.plotly_chart(make_factor_bucket_chart(bucket_df), use_container_width=True)

    st.subheader("Factor-Level Contributions")
    st.dataframe(make_factor_level_contribution_table(factor_contrib), hide_index=True, use_container_width=True)


def render_exposure_tab(bucket_exposures: pd.DataFrame) -> None:
    st.header("Exposure by Factor Bucket")
    st.caption("The sensitivity of the portfolio toward each market segment (factor basket).")
    if not bucket_exposures.empty:
        st.subheader("Portfolio Exposure by Bucket")
        st.dataframe(bucket_exposures, hide_index=True, use_container_width=True)
        st.plotly_chart(make_exposure_chart(bucket_exposures), use_container_width=True)
    else:
        st.info("Not enough factor data for exposure decomposition. Ensure you have multiple factors selected.")

def main() -> None:
    st.set_page_config(page_title="RiskPlus Streamlit", layout="wide")

    if "analysis_ready" not in st.session_state:
        st.session_state.analysis_ready = False
    if "merged_data" not in st.session_state:
        st.session_state.merged_data = None
    if "fund_frames" not in st.session_state:
        st.session_state.fund_frames = {}
    if "fund_weights" not in st.session_state:
        st.session_state.fund_weights = {}
    if "fund_return_cols" not in st.session_state:
        st.session_state.fund_return_cols = {}
    if "factor_frame" not in st.session_state:
        st.session_state.factor_frame = None
    if "factor_cols" not in st.session_state:
        st.session_state.factor_cols = []
    
    # ========== SIDEBAR: UPLOAD & SETTINGS ==========
    with st.sidebar:
        st.header("📊 Data & Analysis Setup")

        st.subheader("1. Fund Return Files")
        fund_uploads = st.file_uploader(
            "Upload one file per fund",
            type=["csv", "xlsx", "xls"],
            accept_multiple_files=True,
        )

        st.subheader("2. Factor Return File")
        factor_upload = st.file_uploader("Upload factor returns file", type=["csv", "xlsx", "xls"])

        st.subheader("3. Analysis Settings")
        date_col = st.text_input("Date column name", value="Date")
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

        uploaded_fund_raws: dict[str, pd.DataFrame] = {}
        uploaded_fund_names: list[str] = []
        factor_raw: pd.DataFrame | None = None

        if fund_uploads:
            st.subheader("4. Fund Mapping")
            for upload in fund_uploads:
                try:
                    fund_raw = read_uploaded_file(upload.name, upload.getvalue())
                except Exception as exc:
                    st.error(f"{upload.name}: {exc}")
                    continue

                if fund_raw.empty:
                    st.error(f"{upload.name} is empty.")
                    continue

                uploaded_fund_raws[upload.name] = fund_raw
                uploaded_fund_names.append(upload.name)

                numeric_guess = [
                    col for col in fund_raw.columns
                    if col != date_col and pd.to_numeric(fund_raw[col], errors="coerce").notna().mean() > 0.5
                ]
                if not numeric_guess:
                    st.error(f"{upload.name} has no numeric return columns.")
                    continue

                st.caption(upload.name)
                st.text_input(
                    f"Fund name for {upload.name}",
                    value=infer_fund_name_from_file(upload.name),
                    key=f"fund_name_{upload.name}",
                )
                st.selectbox(
                    f"Return column in {upload.name}",
                    options=numeric_guess,
                    index=0,
                    key=f"fund_return_{upload.name}",
                )
                st.number_input(
                    f"Weight for {upload.name}",
                    value=float(1.0 / max(len(fund_uploads), 1)),
                    min_value=0.0,
                    step=0.05,
                    key=f"fund_weight_{upload.name}",
                )

        if factor_upload is not None:
            try:
                factor_raw = read_uploaded_file(factor_upload.name, factor_upload.getvalue())
            except Exception as exc:
                st.error(str(exc))
                factor_raw = None

        run_clicked = st.button("Run Analysis", type="primary")

        if run_clicked:
            if not fund_uploads:
                st.error("Upload at least one fund file.")
                st.stop()
            if factor_upload is None:
                st.error("Upload a factor file.")
                st.stop()

            if factor_raw is None:
                st.stop()

            if factor_raw.empty:
                st.error("Factor file is empty.")
                st.stop()

            factor_cols = [col for col in factor_raw.columns if col != date_col]
            if not factor_cols:
                st.error("Factor file must contain at least one factor return column.")
                st.stop()

            factor_frame = prepare_factor_stream(
                factor_raw,
                date_col=date_col,
                factor_cols=factor_cols,
                values_in_percent=values_in_percent,
            )

            fund_frames: dict[str, pd.DataFrame] = {}
            fund_weights: dict[str, float] = {}
            fund_return_cols: dict[str, str] = {}

            for upload in fund_uploads:
                fund_raw = uploaded_fund_raws.get(upload.name)
                if fund_raw is None:
                    continue

                numeric_guess = [
                    col for col in fund_raw.columns
                    if col != date_col and pd.to_numeric(fund_raw[col], errors="coerce").notna().mean() > 0.5
                ]
                if not numeric_guess:
                    continue

                stream_name = st.session_state.get(f"fund_name_{upload.name}", infer_fund_name_from_file(upload.name))
                return_col = st.session_state.get(f"fund_return_{upload.name}", numeric_guess[0])
                fund_weight = float(st.session_state.get(f"fund_weight_{upload.name}", 1.0 / max(len(fund_uploads), 1)))

                prepared_stream = prepare_return_stream(
                    fund_raw,
                    date_col=date_col,
                    return_col=return_col,
                    stream_name=stream_name,
                    values_in_percent=values_in_percent,
                )
                fund_frames[stream_name] = prepared_stream
                fund_weights[stream_name] = fund_weight
                fund_return_cols[stream_name] = return_col

            merged = merge_analysis_frames(fund_frames, factor_frame, join_type="inner")

            if merged.empty:
                st.error("No overlapping dates remained after merging fund and factor files.")
                st.stop()

            if len(merged) < MIN_OBSERVATIONS:
                st.error(f"Merged dataset has {len(merged)} rows; need at least {MIN_OBSERVATIONS}.")
                st.stop()

            st.session_state.analysis_ready = True
            st.session_state.merged_data = merged
            st.session_state.fund_frames = fund_frames
            st.session_state.fund_weights = fund_weights
            st.session_state.fund_return_cols = fund_return_cols
            st.session_state.factor_frame = factor_frame
            st.session_state.factor_cols = factor_cols

    if not st.session_state.analysis_ready or st.session_state.merged_data is None:
        st.info("Upload all fund files and the factor file, then click Run Analysis.")
        st.stop()

    merged_data = st.session_state.merged_data
    fund_frames = st.session_state.fund_frames
    fund_weights = st.session_state.fund_weights
    factor_frame = st.session_state.factor_frame
    factor_cols = st.session_state.factor_cols
    asset_cols = list(fund_frames.keys())

    if asset_cols:
        weight_series = pd.Series(fund_weights, dtype=float).reindex(asset_cols).fillna(0.0)
    else:
        weight_series = pd.Series(dtype=float)
    asset_weight_input = weight_series if not weight_series.empty else None
    raw_df = merged_data.reset_index()
    factor_df = factor_frame.reset_index()

    # Column mapping
    st.subheader("Column Mapping")
    st.caption("Fund files are uploaded separately. Select dates and confirm the factor columns from the factor file.")
    all_cols = raw_df.columns.tolist()
    date_col = st.selectbox("Date column", options=all_cols, index=0)

    if not factor_cols:
        factor_cols = [col for col in factor_df.columns if col != date_col]
    if not factor_cols:
        st.error("No factor columns available in the factor file.")
        st.stop()
    # ========== VALIDATION ==========
    errors, warnings_list = validate_raw_data(
        raw_df,
        date_col=date_col,
        asset_cols=asset_cols,
        factor_cols=factor_cols,
        min_observations=MIN_OBSERVATIONS,
        max_missing_pct=max_missing_pct,
    )
    
    if warnings_list:
        for w in warnings_list:
            st.warning(w)
    
    if errors:
        for err in errors:
            st.error(err)
        st.stop()
    
    # ========== DATA PREPARATION ==========
    data = prepare_analysis_data(
        raw_df,
        date_col=date_col,
        asset_cols=asset_cols,
        factor_cols=factor_cols,
        values_in_percent=values_in_percent,
    )
    
    if data.empty or len(data) < MIN_OBSERVATIONS:
        st.error(f"Insufficient data after cleaning (min {MIN_OBSERVATIONS} required).")
        st.stop()
    
    # Date range selection
    min_date = data.index.min().date()
    max_date = data.index.max().date()
    selected_dates = st.sidebar.slider(
        "Analysis period",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
    )
    
    filtered = data.loc[str(selected_dates[0]) : str(selected_dates[1])]
    if len(filtered) < MIN_OBSERVATIONS:
        st.error(f"Date range has {len(filtered)} rows; expand to at least {MIN_OBSERVATIONS}.")
        st.stop()
    
    # ========== CORE ANALYSIS ==========
    core_results = run_core_analysis(
        filtered,
        asset_cols,
        factor_cols,
        asset_weight_input,
        rf_rate,
        confidence,
        int(num_sims),
        random_seed=42,
    )

    portfolio = core_results.portfolio
    asset_weights = core_results.asset_weights
    ols_results = core_results.ols_results
    hist_stats = core_results.hist_stats
    sim_stats = core_results.sim_stats
    simulated_returns = core_results.simulated_returns
    sys_spec = core_results.sys_spec
    
    # ========== TABS: INDEX, SUMMARY, HISTORICAL RISK, SIMULATED RISK ==========
    tabs = st.tabs(["INDEX", "SUMMARY", "HISTORICAL RISK", "SIMULATED RISK", 
                    "RISK BUDGETING (ETL)", "RISK BUDGETING (StDev)", 
                    "FACTOR CONTRIBUTION", "EXPOSURE BY FACTOR BUCKET"])
    
    # ===== TAB 1: INDEX =====
    with tabs[0]:
        render_index_tab(
            asset_cols,
            report_name,
            portfolio_value,
            factor_cols,
            dist_type,
            corr_method,
            selected_dates,
            len(filtered),
            confidence,
            rf_rate,
            num_sims,
            make_portfolio_pie_chart(asset_cols, asset_weights),
        )
    
    # ===== TAB 2: SUMMARY =====
    with tabs[1]:
        render_summary_tab(hist_stats, sim_stats, ols_results, sys_spec, simulated_returns)
    
    # ===== TAB 3: HISTORICAL RISK =====
    with tabs[2]:
        render_historical_risk_tab(hist_stats, portfolio)
    
    # ===== TAB 4: SIMULATED RISK =====
    with tabs[3]:
        render_simulated_risk_tab(hist_stats, sim_stats, num_sims)
    
    # ===== TAB 5: RISK BUDGETING (ETL) =====
    with tabs[4]:
        st.header("Risk Budgeting by ETL")
        st.caption("Implied Return is the return an asset must deliver to justify its contribution to portfolio ETL. Assets with actual returns above the implied line may justify increased weight; below may suggest reduction.")
        render_risk_budgeting_tab(core_results.rb_etl, "ETL")
    
    # ===== TAB 6: RISK BUDGETING (StDev) =====
    with tabs[5]:
        st.header("Risk Budgeting by Standard Deviation")
        st.caption("Implied Return is the return an asset must deliver to justify its contribution to portfolio standard deviation.")
        render_risk_budgeting_tab(core_results.rb_stdev, "StDev")
    
    # ===== TAB 7: FACTOR CONTRIBUTION =====
    with tabs[6]:
        render_factor_contribution_tab(core_results.factor_contrib, show_factor_buckets)
    
    # ===== TAB 8: EXPOSURE BY FACTOR BUCKET =====
    with tabs[7]:
        render_exposure_tab(core_results.bucket_exposures)
    
    # ========== FOOTER: METADATA & EXPORT =====
    st.sidebar.divider()
    st.sidebar.caption("**RiskPlus Streamlit MVP**  \nStudent-t simulation, fat-tail risk metrics, systematic/idiosyncratic decomposition. Methodology caveats: Does not implement proprietary RiskPlus copula or stepwise regression.")


if __name__ == "__main__":
    main()