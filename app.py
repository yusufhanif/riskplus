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
from typing import Any
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats
import statsmodels.api as sm
import streamlit as st
from statsmodels.stats.outliers_influence import variance_inflation_factor
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from riskplus_core.analytics import (
    DEFAULT_CONFIDENCE,
    DEFAULT_NUM_SIMULATIONS,
    DEFAULT_RF_RATE,
    DEFAULT_STUDENT_T_DF,
    compute_factor_bucket_exposures,
    compute_factor_contribution,
    compute_historical_stats,
    compute_implied_returns,
    compute_marginal_risk_contributions,
    compute_percent_risk_contributions,
    compute_risk_budgeting_table,
    compute_risk_metrics_from_dist,
    compute_systematic_specific_risk,
    fit_student_t_distribution,
    get_factor_bucket_mapping,
    simulate_fat_tailed_returns,
)
from riskplus_core.data import (
    MIN_OBSERVATIONS,
    annualization_factor,
    build_portfolio_series,
    compute_vif,
    detect_frequency,
    prepare_analysis_data,
    read_uploaded_file,
    run_ols,
    validate_raw_data,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)

DEFAULT_EWMA_DECAY = 0.94

def main() -> None:
    st.set_page_config(page_title="RiskPlus Streamlit", layout="wide")
    
    # ========== SIDEBAR: UPLOAD & SETTINGS ==========
    with st.sidebar:
        st.header("📊 Data & Analysis Setup")
        
        # File upload
        upload = st.file_uploader("Upload portfolio & factor data", type=["csv", "xlsx", "xls"])
        
        if upload is None:
            st.info("Upload a file containing date, portfolio return, and factor return columns.")
            st.stop()
        
        try:
            raw_df = read_uploaded_file(upload.name, upload.getvalue())
        except Exception as exc:
            st.error(str(exc))
            st.stop()
        
        if raw_df.empty:
            st.error("Uploaded file is empty.")
            st.stop()
        
        # Column mapping
        st.subheader("Column Mapping")
        all_cols = raw_df.columns.tolist()
        date_col = st.selectbox("Date column", options=all_cols, index=0)
        
        numeric_guess = [
            col for col in all_cols
            if col != date_col and pd.to_numeric(raw_df[col], errors="coerce").notna().mean() > 0.5
        ]
        if not numeric_guess:
            st.error("No numeric columns detected.")
            st.stop()
        
        asset_mode = st.radio(
            "Return input mode",
            options=["Single portfolio column", "Multiple fund streams"],
            index=0,
        )

        asset_weight_input = None
        if asset_mode == "Single portfolio column":
            asset_cols = [st.selectbox("Portfolio return column", options=numeric_guess, index=0)]
        else:
            default_assets = numeric_guess[: min(3, len(numeric_guess))]
            asset_cols = st.multiselect("Fund return columns", options=numeric_guess, default=default_assets)
            if asset_cols:
                st.caption("Optional weights for the selected funds. Leave them as-is to use equal weights.")
                asset_weights_dict: dict[str, float] = {}
                default_weight = 1.0 / len(asset_cols)
                for asset_col in asset_cols:
                    asset_weights_dict[asset_col] = st.number_input(
                        f"Weight for {asset_col}",
                        value=float(default_weight),
                        min_value=0.0,
                        step=0.05,
                        key=f"weight_{asset_col}",
                    )
                asset_weight_input = pd.Series(asset_weights_dict, dtype=float)

        if not asset_cols:
            st.error("Select at least one return column.")
            st.stop()

        factor_options = [col for col in numeric_guess if col not in asset_cols]
        default_factors = factor_options[: min(4, len(factor_options))]
        factor_cols = st.multiselect("Factor return columns", options=factor_options, default=default_factors)
        
        # Data formatting
        values_in_percent = st.checkbox("Values are in percent (5 = 5%)", value=False)
        max_missing_pct = st.slider("Max missing data %", 0.0, 50.0, 5.0, 1.0) / 100.0
        
        # Portfolio metadata
        st.subheader("Portfolio Metadata")
        report_name = st.text_input("Report name", value="Portfolio Risk Analysis")
        portfolio_value = st.number_input("Portfolio value", value=1000000.0, step=100000.0)
        rf_rate = st.slider("Risk-free rate (annual %)", 0.0, 10.0, 2.0, 0.1) / 100.0
        
        # Risk parameters
        st.subheader("Risk Analysis Parameters")
        confidence = st.slider("Confidence level", 0.85, 0.99, DEFAULT_CONFIDENCE, 0.01)
        num_sims = st.selectbox("Number of simulations", options=[10000, 50000, 100000], index=1)
        
        # Correlation method
        corr_method = st.selectbox("Covariance method", options=["Classical", "EWMA"], index=0)
        ewma_decay = st.slider("EWMA decay factor", 0.90, 0.99, DEFAULT_EWMA_DECAY, 0.01) if corr_method == "EWMA" else DEFAULT_EWMA_DECAY
        
        # Distribution
        dist_type = st.selectbox("Distribution type", options=["Student-t", "Gaussian"], index=0)
        
        # Risk Budgeting & Factor settings
        st.subheader("Risk Decomposition Settings")
        show_factor_buckets = st.checkbox("Show factor bucket analysis", value=True)
        display_mode = st.radio("Exposure display mode", options=["Weighted portfolio exposure", "Standalone beta"], index=0)
    
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
    freq = detect_frequency(filtered.index)
    portfolio, asset_weights = build_portfolio_series(filtered, asset_cols, asset_weight_input)
    factors = filtered[factor_cols]
    
    # Run OLS factor model
    ols_results = run_ols(portfolio, factors)
    hist_stats = compute_historical_stats(portfolio, freq, rf_rate)
    
    # Simulate fat-tailed returns
    simulated_returns = simulate_fat_tailed_returns(
        portfolio, n_sims=int(num_sims), random_seed=42
    )
    sim_stats = compute_historical_stats(simulated_returns["Portfolio"], "monthly", rf_rate)
    
    # Risk contributions (using simulated returns)
    weights = np.array([1.0])  # Portfolio only has one "asset"
    mc_contribs = compute_marginal_risk_contributions(weights, simulated_returns, confidence)
    
    # Systematic vs specific
    sys_spec = compute_systematic_specific_risk(portfolio, factors, ols_results["model"])

    portfolio_label = "Portfolio" if len(asset_cols) == 1 else "Aggregated Portfolio"
    
    # ========== TABS: INDEX, SUMMARY, HISTORICAL RISK, SIMULATED RISK ==========
    tabs = st.tabs(["INDEX", "SUMMARY", "HISTORICAL RISK", "SIMULATED RISK", 
                    "RISK BUDGETING (ETL)", "RISK BUDGETING (StDev)", 
                    "FACTOR CONTRIBUTION", "EXPOSURE BY FACTOR BUCKET"])
    
    # ===== TAB 1: INDEX =====
    with tabs[0]:
        st.header("Analysis Index & Settings")
        
        # Calculation settings table
        settings_data = {
            "Parameter": [
                "Report Name",
                "Portfolio Value",
                "Number of Funds",
                "Factor Model",
                "Distribution Type",
                "Correlation Method",
                "Analysis Period",
                "Observations",
                "Confidence Level",
                "Risk-Free Rate (annual)",
                "Number of Simulations",
            ],
            "Value": [
                report_name,
                f"${portfolio_value:,.0f}",
                len(asset_cols),
                f"OLS ({len(factor_cols)} factors)",
                dist_type,
                corr_method,
                f"{selected_dates[0]} to {selected_dates[1]}",
                len(filtered),
                f"{confidence:.1%}",
                f"{rf_rate:.2%}",
                f"{num_sims:,}",
            ],
        }
        st.subheader("Calculation Settings")
        st.dataframe(pd.DataFrame(settings_data), hide_index=True, use_container_width=True)
        
        # Portfolio composition
        st.subheader("Portfolio Composition")
        comp_data = {
            "Asset": asset_cols + factor_cols,
            "Weight": list(asset_weights.values) + [0.0] * len(factor_cols),
        }
        fig_pie = px.pie(
            values=list(asset_weights.values),
            names=asset_cols,
            title="Portfolio Structure",
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Navigation
        st.subheader("Quick Links")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info("📈 **SUMMARY**  \nKey metrics and risk overview")
        with col2:
            st.info("📊 **HISTORICAL RISK**  \nDetailed historical statistics")
        with col3:
            st.info("🎲 **SIMULATED RISK**  \nFat-tail risk metrics")
    
    # ===== TAB 2: SUMMARY =====
    with tabs[1]:
        st.header("Risk Summary")
        
        # Portfolio snapshot KPIs
        col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
        with col_kpi1:
            st.metric("Ann. Return", f"{hist_stats['ann_mean']:.2%}")
        with col_kpi2:
            st.metric("Ann. Volatility", f"{hist_stats['ann_vol']:.2%}")
        with col_kpi3:
            st.metric("Sharpe Ratio", f"{hist_stats['sharpe']:.2f}")
        with col_kpi4:
            st.metric("VaR (95%)", f"{hist_stats['var']:.2%}")
        
        # Simulated distribution
        st.subheader("Simulated Return Distribution (Student-t)")
        sim_rets = simulated_returns["Portfolio"].values
        mean_ret = np.mean(sim_rets)
        etl = sim_stats["etl"]
        etr = sim_stats["etr"]
        
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(x=sim_rets, nbinsx=50, name="Simulated Returns"))
        fig_dist.add_vline(x=-etl, line_dash="dash", line_color="red", annotation_text="VaR (95%)")
        fig_dist.add_vline(x=mean_ret, line_dash="solid", line_color="green", annotation_text="Mean")
        fig_dist.add_vline(x=etr, line_dash="dash", line_color="blue", annotation_text="ETR (95%)")
        fig_dist.update_layout(title="Simulated Return Distribution", xaxis_title="Return", yaxis_title="Frequency")
        st.plotly_chart(fig_dist, use_container_width=True)
        
        # Risk metrics comparison
        col_trad, col_tail = st.columns(2)
        with col_trad:
            st.subheader("Traditional Measures")
            trad_data = {
                "Metric": ["Mean Return", "Volatility", "Sharpe Ratio"],
                "Value": [
                    f"{hist_stats['ann_mean']:.2%}",
                    f"{hist_stats['ann_vol']:.2%}",
                    f"{hist_stats['sharpe']:.2f}",
                ],
            }
            st.table(pd.DataFrame(trad_data))
        
        with col_tail:
            st.subheader("Tail Risk Measures")
            tail_data = {
                "Metric": ["VaR (95%)", "ETL (95%)", "STARR", "Rachev Ratio"],
                "Value": [
                    f"{hist_stats['var']:.2%}",
                    f"{hist_stats['etl']:.2%}",
                    f"{hist_stats['starr']:.2f}",
                    f"{hist_stats['rachev']:.2f}",
                ],
            }
            st.table(pd.DataFrame(tail_data))
        
        # Factor contributions
        st.subheader("Factor Model Diagnostics")
        col_r2, col_sys = st.columns(2)
        with col_r2:
            st.metric("Model R²", f"{ols_results['r2']:.3f}")
        with col_sys:
            st.metric("Systematic Risk %", f"{sys_spec['systematic_pct']:.1%}")
        
        # Top factors by contribution
        st.subheader("Factor Exposures & Significance")
        coef_display = ols_results["coef_table"].copy()
        coef_display = coef_display[coef_display.index != "const"]
        st.dataframe(coef_display[["Coefficient", "tStat", "pValue"]], use_container_width=True)
    
    # ===== TAB 3: HISTORICAL RISK =====
    with tabs[2]:
        st.header("Historical Risk Statistics")
        st.caption("Actual historical period statistics")
        
        hist_risk_data = {
            "Metric": [
                "Observations",
                "Mean Return",
                "Annualized Mean",
                "Volatility",
                "Annualized Volatility",
                "Skewness",
                "Excess Kurtosis",
                "VaR (95%)",
                "ETL (95%)",
                "ETR (95%)",
                "Sharpe Ratio",
                "STARR",
                "Rachev Ratio",
                "Max Drawdown",
                "Best Period",
                "Worst Period",
            ],
            "Portfolio": [
                hist_stats["obs_count"],
                f"{hist_stats['mean']:.4f}",
                f"{hist_stats['ann_mean']:.2%}",
                f"{hist_stats['vol']:.4f}",
                f"{hist_stats['ann_vol']:.2%}",
                f"{hist_stats['skew']:.3f}",
                f"{hist_stats['xkurt']:.3f}",
                f"{hist_stats['var']:.2%}",
                f"{hist_stats['etl']:.2%}",
                f"{hist_stats['etr']:.2%}",
                f"{hist_stats['sharpe']:.3f}",
                f"{hist_stats['starr']:.3f}",
                f"{hist_stats['rachev']:.3f}",
                f"{hist_stats['max_dd']:.2%}",
                f"{hist_stats['best_period']:.2%}",
                f"{hist_stats['worst_period']:.2%}",
            ],
        }
        
        hist_df = pd.DataFrame(hist_risk_data)
        st.dataframe(hist_df, hide_index=True, use_container_width=True)
        
        # Cumulative return plot
        st.subheader("Cumulative Growth")
        wealth = (1 + portfolio).cumprod()
        fig_wealth = px.line(
            x=wealth.index,
            y=wealth.values,
            labels={"x": "Date", "y": "Growth of $1"},
            title="Cumulative Portfolio Growth",
        )
        st.plotly_chart(fig_wealth, use_container_width=True)
    
    # ===== TAB 4: SIMULATED RISK =====
    with tabs[3]:
        st.header("Simulated Risk Statistics")
        st.caption(f"Based on {num_sims:,} Student-t Monte Carlo simulations")
        
        sim_risk_data = {
            "Metric": [
                "Observations (simulated)",
                "Mean Return",
                "Annualized Mean",
                "Volatility",
                "Annualized Volatility",
                "Skewness",
                "Excess Kurtosis",
                "VaR (95%)",
                "ETL (95%)",
                "ETR (95%)",
                "Sharpe Ratio",
                "STARR",
                "Rachev Ratio",
            ],
            "Portfolio": [
                int(num_sims),
                f"{sim_stats['mean']:.4f}",
                f"{sim_stats['ann_mean']:.2%}",
                f"{sim_stats['vol']:.4f}",
                f"{sim_stats['ann_vol']:.2%}",
                f"{sim_stats['skew']:.3f}",
                f"{sim_stats['xkurt']:.3f}",
                f"{sim_stats['var']:.2%}",
                f"{sim_stats['etl']:.2%}",
                f"{sim_stats['etr']:.2%}",
                f"{sim_stats['sharpe']:.3f}",
                f"{sim_stats['starr']:.3f}",
                f"{sim_stats['rachev']:.3f}",
            ],
        }
        
        sim_df = pd.DataFrame(sim_risk_data)
        st.dataframe(sim_df, hide_index=True, use_container_width=True)
        
        # Comparison: Historical vs Simulated
        st.subheader("Historical vs. Simulated Comparison")
        comparison_data = {
            "Metric": ["Ann. Mean", "Ann. Vol", "Sharpe", "VaR (95%)", "ETL (95%)", "Rachev"],
            "Historical": [
                f"{hist_stats['ann_mean']:.2%}",
                f"{hist_stats['ann_vol']:.2%}",
                f"{hist_stats['sharpe']:.2f}",
                f"{hist_stats['var']:.2%}",
                f"{hist_stats['etl']:.2%}",
                f"{hist_stats['rachev']:.2f}",
            ],
            "Simulated": [
                f"{sim_stats['ann_mean']:.2%}",
                f"{sim_stats['ann_vol']:.2%}",
                f"{sim_stats['sharpe']:.2f}",
                f"{sim_stats['var']:.2%}",
                f"{sim_stats['etl']:.2%}",
                f"{sim_stats['rachev']:.2f}",
            ],
        }
        st.dataframe(pd.DataFrame(comparison_data), hide_index=True, use_container_width=True)
    
    # ===== TAB 5: RISK BUDGETING (ETL) =====
    with tabs[4]:
        st.header("Risk Budgeting by ETL")
        st.caption("Implied Return is the return an asset must deliver to justify its contribution to portfolio ETL. Assets with actual returns above the implied line may justify increased weight; below may suggest reduction.")
        
        # Compute risk budgeting for ETL
        rb_etl = compute_risk_budgeting_table(
            np.array([1.0]),
            simulated_returns,
            mc_contribs["mc_etl"],
            "ETL",
            rf_rate,
            annualization_factor(freq),
        )
        
        # Display table
        st.subheader("Risk Budgeting Table (ETL)")
        st.dataframe(rb_etl, hide_index=True, use_container_width=True)
        
        # RiskPlus-style chart: MC to ETL vs Return
        st.subheader("Risk-Return Analysis")
        fig_rb_etl = px.scatter(
            rb_etl,
            x="MC to Risk",
            y="Mean Return (%)",
            hover_data=["Asset", "Status"],
            labels={"MC to Risk": "Marginal Contribution to ETL", "Mean Return (%)": "Expected Return (%)"},
            title="Mean Return vs. Marginal ETL Contribution",
        )
        # Add implied return line as scatter
        fig_rb_etl.add_scatter(
            x=rb_etl["MC to Risk"],
            y=rb_etl["Implied Return (%)"],
            mode="lines+markers",
            name="Implied Return (Risk-Adjusted)",
            line=dict(dash="dash", color="red"),
        )
        st.plotly_chart(fig_rb_etl, use_container_width=True)
    
    # ===== TAB 6: RISK BUDGETING (StDev) =====
    with tabs[5]:
        st.header("Risk Budgeting by Standard Deviation")
        st.caption("Implied Return is the return an asset must deliver to justify its contribution to portfolio standard deviation.")
        
        # Risk budgeting for StDev
        rb_stdev = compute_risk_budgeting_table(
            np.array([1.0]),
            simulated_returns,
            mc_contribs["mc_vol"],
            "StDev",
            rf_rate,
            annualization_factor(freq),
        )
        
        st.subheader("Risk Budgeting Table (StDev)")
        st.dataframe(rb_stdev, hide_index=True, use_container_width=True)
        
        st.subheader("Risk-Return Analysis")
        fig_rb_stdev = px.scatter(
            rb_stdev,
            x="MC to Risk",
            y="Mean Return (%)",
            hover_data=["Asset", "Status"],
            labels={"MC to Risk": "Marginal Contribution to StDev", "Mean Return (%)": "Expected Return (%)"},
            title="Mean Return vs. Marginal StDev Contribution",
        )
        fig_rb_stdev.add_scatter(
            x=rb_stdev["MC to Risk"],
            y=rb_stdev["Implied Return (%)"],
            mode="lines+markers",
            name="Implied Return (Risk-Adjusted)",
            line=dict(dash="dash", color="red"),
        )
        st.plotly_chart(fig_rb_stdev, use_container_width=True)
    
    # ===== TAB 7: FACTOR CONTRIBUTION =====
    with tabs[6]:
        st.header("Factor Contribution to Portfolio Risk")
        st.caption("Percentage Contribution to Risk by factor bucket shows how a given factor bucket contributes to the overall portfolio risk.")
        
        factor_contrib = compute_factor_contribution(
            ols_results["model"],
            factors,
            portfolio,
            simulated_returns["Portfolio"],
            confidence,
        )
        
        # Section 1: Systematic vs Specific
        col_sys, col_spec = st.columns(2)
        with col_sys:
            st.metric("Systematic Risk (StDev %)", f"{factor_contrib['systematic_stdev_pct']:.1%}")
        with col_spec:
            st.metric("Specific/Idiosyncratic Risk (%)", f"{factor_contrib['specific_stdev_pct']:.1%}")
        
        # Section 2: Factor bucketing
        if show_factor_buckets and factor_contrib["bucket_mapping"]:
            st.subheader("Factor Risk by Bucket")
            
            bucket_data = []
            for bucket, factors_in_bucket in factor_contrib["bucket_mapping"].items():
                bucket_contrib_stdev = sum([
                    abs(factor_contrib["factor_mc_stdev"][i])
                    for i, f in enumerate(factor_contrib["factor_names"])
                    if f in factors_in_bucket
                ]) / (np.sum(np.abs(factor_contrib["factor_mc_stdev"])) + 1e-10)
                
                bucket_data.append({
                    "Factor Bucket": bucket,
                    "PC to StDev (%)": bucket_contrib_stdev * 100,
                    "Factor Count": len(factors_in_bucket),
                })
            
            bucket_df = pd.DataFrame(bucket_data)
            st.dataframe(bucket_df, hide_index=True, use_container_width=True)
            
            # Bucketing chart
            fig_bucket = px.bar(
                bucket_df,
                x="Factor Bucket",
                y="PC to StDev (%)",
                title="Risk Contribution by Factor Bucket",
                labels={"PC to StDev (%)": "Contribution to StDev (%)"},
            )
            st.plotly_chart(fig_bucket, use_container_width=True)
        
        # Section 3: Individual factors
        st.subheader("Factor-Level Contributions")
        factor_table = pd.DataFrame({
            "Factor": factor_contrib["factor_names"],
            "Beta": factor_contrib["betas"].values,
            "MC to StDev": factor_contrib["factor_mc_stdev"],
        })
        st.dataframe(factor_table, hide_index=True, use_container_width=True)
    
    # ===== TAB 8: EXPOSURE BY FACTOR BUCKET =====
    with tabs[7]:
        st.header("Exposure by Factor Bucket")
        st.caption("The sensitivity of the portfolio toward each market segment (factor basket).")
        
        # Compute exposures from OLS betas
        betas = ols_results["coef_table"].index
        betas_values = ols_results["coef_table"]["Coefficient"]
        bucket_mapping = get_factor_bucket_mapping(factor_cols)
        
        bucket_exposures = compute_factor_bucket_exposures(
            betas_values,
            bucket_mapping,
        )
        
        if not bucket_exposures.empty:
            st.subheader("Portfolio Exposure by Bucket")
            st.dataframe(bucket_exposures, hide_index=True, use_container_width=True)
            
            fig_exposure = px.bar(
                bucket_exposures,
                x="Bucket",
                y="Exposure",
                title="Portfolio Sensitivity by Factor Bucket",
                labels={"Exposure": "Net Exposure"},
            )
            st.plotly_chart(fig_exposure, use_container_width=True)
        else:
            st.info("Not enough factor data for exposure decomposition. Ensure you have multiple factors selected.")
    
    # ========== FOOTER: METADATA & EXPORT =====
    st.sidebar.divider()
    st.sidebar.caption("**RiskPlus Streamlit MVP**  \nStudent-t simulation, fat-tail risk metrics, systematic/idiosyncratic decomposition. Methodology caveats: Does not implement proprietary RiskPlus copula or stepwise regression.")


if __name__ == "__main__":
    main()