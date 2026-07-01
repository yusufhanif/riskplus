"""Pure DataFrame and Plotly builders for the Streamlit UI."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def _fmt_pct(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value) * 100:.2f}"


def _fmt_num(value: float | int | None, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}"


def make_settings_table(
    report_name: str,
    portfolio_value: float,
    asset_cols: list[str],
    factor_cols: list[str],
    dist_type: str,
    corr_method: str,
    selected_dates: tuple[object, object],
    observations: int,
    confidence: float,
    rf_rate: float,
    num_sims: int,
) -> pd.DataFrame:
    """Build the settings summary table for the Index tab."""
    return pd.DataFrame(
        {
            'Parameter': [
                'Report Name',
                'Portfolio Value',
                'Number of Funds',
                'Factor Model',
                'Distribution Type',
                'Correlation Method',
                'Analysis Period',
                'Observations',
                'Confidence Level',
                'Risk-Free Rate (annual)',
                'Number of Simulations',
            ],
            'Value': [
                report_name,
                f'${portfolio_value:,.0f}',
                len(asset_cols),
                f'OLS ({len(factor_cols)} factors)',
                dist_type,
                corr_method,
                f'{selected_dates[0]} to {selected_dates[1]}',
                observations,
                f'{confidence:.1%}',
                f'{rf_rate:.2%}',
                f'{num_sims:,}',
            ],
        }
    )


def make_historical_risk_table(hist_stats: dict[str, float]) -> pd.DataFrame:
    """Build the historical risk table shown in the Historical Risk tab."""
    return pd.DataFrame(
        {
            'Metric': [
                'Observations',
                'Mean Return',
                'Annualized Mean',
                'Volatility',
                'Annualized Volatility',
                'Skewness',
                'Excess Kurtosis',
                'VaR (95%)',
                'ETL (95%)',
                'ETR (95%)',
                'Sharpe Ratio',
                'STARR',
                'Rachev Ratio',
                'Max Drawdown',
                'Best Period',
                'Worst Period',
            ],
            'Portfolio': [
                hist_stats['obs_count'],
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
    )


def make_historical_risk_table_detailed(
    portfolio_stats: dict[str, float],
    hist_stats_by_fund: dict[str, dict[str, float]],
    asset_weights: pd.Series,
) -> pd.DataFrame:
    def _row(name: str, stats: dict[str, float], weight: float | None) -> dict[str, object]:
        return {
            'Name': name,
            'Weight (%)': '' if weight is None else f"{weight * 100:.2f}",
            'Mean': _fmt_num(stats.get('mean'), 4),
            'StDev': _fmt_num(stats.get('vol'), 4),
            'Sharpe': _fmt_num(stats.get('sharpe'), 3),
            'STARR': _fmt_num(stats.get('starr'), 3),
            'Rachev': _fmt_num(stats.get('rachev'), 3),
            'Skew': _fmt_num(stats.get('skew'), 3),
            'Excess Kurtosis': _fmt_num(stats.get('xkurt'), 3),
            'Max Drawdown (%)': _fmt_pct(stats.get('max_dd')),
            'Ann. Mean': _fmt_pct(stats.get('ann_mean')),
            'Ann. StDev': _fmt_pct(stats.get('ann_vol')),
            'VaR (95%)': _fmt_pct(stats.get('var')),
            'ETL (95%)': _fmt_pct(stats.get('etl')),
            'ETR (95%)': _fmt_pct(stats.get('etr')),
            'Observations': int(stats.get('obs_count', 0) or 0),
        }

    rows: list[dict[str, object]] = []
    rows.append(_row('Portfolio', portfolio_stats, None))
    for fund in asset_weights.index.tolist():
        fund_stats = hist_stats_by_fund.get(fund)
        if fund_stats is None:
            continue
        rows.append(_row(fund, fund_stats, float(asset_weights.loc[fund])))

    return pd.DataFrame(rows)


def make_simulated_risk_table(sim_stats: dict[str, float], num_sims: int) -> pd.DataFrame:
    """Build the simulated risk table shown in the Simulated Risk tab."""
    return pd.DataFrame(
        {
            'Metric': [
                'Observations (simulated)',
                'Mean Return',
                'Annualized Mean',
                'Volatility',
                'Annualized Volatility',
                'Skewness',
                'Excess Kurtosis',
                'VaR (95%)',
                'ETL (95%)',
                'ETR (95%)',
                'Sharpe Ratio',
                'STARR',
                'Rachev Ratio',
            ],
            'Portfolio': [
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
    )


def make_simulated_risk_table_detailed(
    portfolio_stats: dict[str, float],
    sim_stats_by_fund: dict[str, dict[str, float]],
    asset_weights: pd.Series,
    mc_contribs: dict[str, object],
    pc_contribs: dict[str, object],
) -> pd.DataFrame:
    mc_vol = list(mc_contribs.get('mc_vol', []))
    mc_etl = list(mc_contribs.get('mc_etl', []))
    mc_etr = list(mc_contribs.get('mc_etr', []))

    pc_vol = list(pc_contribs.get('pc_vol', []))
    pc_etl = list(pc_contribs.get('pc_etl', []))
    pc_etr = list(pc_contribs.get('pc_etr', []))

    rows: list[dict[str, object]] = []
    rows.append(
        {
            'Name': 'Portfolio',
            'Weight (%)': '100.00',
            'ETR (%)': _fmt_pct(portfolio_stats.get('etr')),
            'ETL (%)': _fmt_pct(portfolio_stats.get('etl')),
            'VaR (%)': _fmt_pct(portfolio_stats.get('var')),
            'PC to StDev (%)': '100.00',
            'PC to ETL (%)': '100.00',
            'PC to ETR (%)': '100.00',
            'MC to StDev (bps)': '',
            'MC to ETL (bps)': '',
            'MC to ETR (bps)': '',
            'Ann. Mean': _fmt_pct(portfolio_stats.get('ann_mean')),
            'Ann. StDev': _fmt_pct(portfolio_stats.get('ann_vol')),
            'Observations (simulated)': int(portfolio_stats.get('obs_count', 0) or 0),
        }
    )

    for idx, fund in enumerate(asset_weights.index.tolist()):
        stats = sim_stats_by_fund.get(fund)
        if stats is None:
            continue

        rows.append(
            {
                'Name': fund,
                'Weight (%)': f"{float(asset_weights.iloc[idx]) * 100:.2f}",
                'ETR (%)': _fmt_pct(stats.get('etr')),
                'ETL (%)': _fmt_pct(stats.get('etl')),
                'VaR (%)': _fmt_pct(stats.get('var')),
                'PC to StDev (%)': _fmt_num((pc_vol[idx] * 100) if idx < len(pc_vol) else None, 2),
                'PC to ETL (%)': _fmt_num((pc_etl[idx] * 100) if idx < len(pc_etl) else None, 2),
                'PC to ETR (%)': _fmt_num((pc_etr[idx] * 100) if idx < len(pc_etr) else None, 2),
                'MC to StDev (bps)': _fmt_num((mc_vol[idx] * 10000) if idx < len(mc_vol) else None, 2),
                'MC to ETL (bps)': _fmt_num((mc_etl[idx] * 10000) if idx < len(mc_etl) else None, 2),
                'MC to ETR (bps)': _fmt_num((mc_etr[idx] * 10000) if idx < len(mc_etr) else None, 2),
                'Ann. Mean': _fmt_pct(stats.get('ann_mean')),
                'Ann. StDev': _fmt_pct(stats.get('ann_vol')),
                'Observations (simulated)': int(stats.get('obs_count', 0) or 0),
            }
        )

    return pd.DataFrame(rows)


def make_historical_vs_simulated_table(hist_stats: dict[str, float], sim_stats: dict[str, float]) -> pd.DataFrame:
    """Build the historical versus simulated comparison table."""
    return pd.DataFrame(
        {
            'Metric': ['Ann. Mean', 'Ann. Vol', 'Sharpe', 'VaR (95%)', 'ETL (95%)', 'Rachev'],
            'Historical': [
                f"{hist_stats['ann_mean']:.2%}",
                f"{hist_stats['ann_vol']:.2%}",
                f"{hist_stats['sharpe']:.2f}",
                f"{hist_stats['var']:.2%}",
                f"{hist_stats['etl']:.2%}",
                f"{hist_stats['rachev']:.2f}",
            ],
            'Simulated': [
                f"{sim_stats['ann_mean']:.2%}",
                f"{sim_stats['ann_vol']:.2%}",
                f"{sim_stats['sharpe']:.2f}",
                f"{sim_stats['var']:.2%}",
                f"{sim_stats['etl']:.2%}",
                f"{sim_stats['rachev']:.2f}",
            ],
        }
    )


def make_traditional_measures_table(hist_stats: dict[str, float]) -> pd.DataFrame:
    """Build the compact traditional measures table."""
    return pd.DataFrame(
        {
            'Metric': ['Mean Return', 'Volatility', 'Sharpe Ratio'],
            'Value': [
                f"{hist_stats['ann_mean']:.2%}",
                f"{hist_stats['ann_vol']:.2%}",
                f"{hist_stats['sharpe']:.2f}",
            ],
        }
    )


def make_tail_risk_table(hist_stats: dict[str, float]) -> pd.DataFrame:
    """Build the compact tail-risk measures table."""
    return pd.DataFrame(
        {
            'Metric': ['VaR (95%)', 'ETL (95%)', 'STARR', 'Rachev Ratio'],
            'Value': [
                f"{hist_stats['var']:.2%}",
                f"{hist_stats['etl']:.2%}",
                f"{hist_stats['starr']:.2f}",
                f"{hist_stats['rachev']:.2f}",
            ],
        }
    )


def make_factor_exposure_table(ols_results: dict[str, object]) -> pd.DataFrame:
    """Return the factor exposure table derived from OLS coefficients."""
    coef_display = ols_results['coef_table'].copy()
    return coef_display[coef_display.index != 'const'][['Coefficient', 'tStat', 'pValue']]


def make_factor_level_contribution_table(factor_contrib: dict[str, object]) -> pd.DataFrame:
    """Build the factor-level contribution table used in the factor tab."""
    return pd.DataFrame(
        {
            'Factor': factor_contrib['factor_names'],
            'Beta': factor_contrib['betas'].values,
            'MC to StDev': factor_contrib['factor_mc_stdev'],
        }
    )


def make_factor_bucket_table(factor_contrib: dict[str, object]) -> pd.DataFrame:
    """Build the bucket-level contribution table used for factor grouping."""
    bucket_data = []
    for bucket, factors_in_bucket in factor_contrib['bucket_mapping'].items():
        bucket_contrib_stdev = sum(
            [
                abs(factor_contrib['factor_mc_stdev'][i])
                for i, factor_name in enumerate(factor_contrib['factor_names'])
                if factor_name in factors_in_bucket
            ]
        ) / (sum(abs(factor_contrib['factor_mc_stdev'])) + 1e-10)

        bucket_data.append(
            {
                'Factor Bucket': bucket,
                'PC to StDev (%)': bucket_contrib_stdev * 100,
                'Factor Count': len(factors_in_bucket),
            }
        )

    return pd.DataFrame(bucket_data)


def make_portfolio_pie_chart(asset_cols: list[str], asset_weights: pd.Series) -> go.Figure:
    """Build the portfolio composition pie chart."""
    return px.pie(
        values=list(asset_weights.values),
        names=asset_cols,
        title='Portfolio Structure',
    )


def make_simulated_distribution_chart(simulated_returns: pd.Series | pd.DataFrame, sim_stats: dict[str, float]) -> go.Figure:
    """Build the simulated return distribution histogram with summary lines."""
    if isinstance(simulated_returns, pd.DataFrame):
        sim_rets = simulated_returns['Portfolio'].values
    else:
        sim_rets = simulated_returns.values

    mean_ret = float(sim_rets.mean())
    etl = sim_stats['etl']
    etr = sim_stats['etr']

    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(x=sim_rets, nbinsx=50, name='Simulated Returns'))
    fig_dist.add_vline(x=-etl, line_dash='dash', line_color='red', annotation_text='VaR (95%)')
    fig_dist.add_vline(x=mean_ret, line_dash='solid', line_color='green', annotation_text='Mean')
    fig_dist.add_vline(x=etr, line_dash='dash', line_color='blue', annotation_text='ETR (95%)')
    fig_dist.update_layout(title='Simulated Return Distribution', xaxis_title='Return', yaxis_title='Frequency')
    return fig_dist


def make_cumulative_growth_chart(portfolio: pd.Series) -> go.Figure:
    """Build the cumulative growth line chart."""
    wealth = (1 + portfolio).cumprod()
    return px.line(
        x=wealth.index,
        y=wealth.values,
        labels={'x': 'Date', 'y': 'Growth of $1'},
        title='Cumulative Portfolio Growth',
    )


def make_risk_budgeting_chart(rb_table: pd.DataFrame, risk_label: str) -> go.Figure:
    """Build the risk budgeting scatter chart with implied return overlay."""
    fig = px.scatter(
        rb_table,
        x='MC to Risk',
        y='Mean Return (%)',
        hover_data=['Asset', 'Status'],
        labels={'MC to Risk': f'Marginal Contribution to {risk_label}', 'Mean Return (%)': 'Expected Return (%)'},
        title=f'Mean Return vs. Marginal {risk_label} Contribution',
    )
    fig.add_scatter(
        x=rb_table['MC to Risk'],
        y=rb_table['Implied Return (%)'],
        mode='lines+markers',
        name='Implied Return (Risk-Adjusted)',
        line=dict(dash='dash', color='red'),
    )
    return fig


def make_factor_bucket_chart(bucket_df: pd.DataFrame) -> go.Figure:
    """Build the factor bucket contribution bar chart."""
    return px.bar(
        bucket_df,
        x='Factor Bucket',
        y='PC to StDev (%)',
        title='Risk Contribution by Factor Bucket',
        labels={'PC to StDev (%)': 'Contribution to StDev (%)'},
    )


def make_exposure_chart(bucket_exposures: pd.DataFrame) -> go.Figure:
    """Build the factor bucket exposure bar chart."""
    return px.bar(
        bucket_exposures,
        x='Bucket',
        y='Exposure',
        title='Portfolio Sensitivity by Factor Bucket',
        labels={'Exposure': 'Net Exposure'},
    )