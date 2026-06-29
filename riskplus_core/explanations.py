"""Reusable plain-English explanations for RiskPlus Streamlit outputs."""

from __future__ import annotations

_HISTORICAL_METRIC_EXPLANATIONS = {
    'ann_mean': (
        'Annual return shows the average growth rate implied by the historical portfolio series. '
        'Higher is better when it is earned without an outsized increase in risk. '
        'A portfolio manager might compare it with volatility and drawdown before deciding whether the current mix is worth keeping. '
        'This is an MVP historical estimate based on the uploaded data, not a proprietary RiskPlus output.'
    ),
    'ann_vol': (
        'Annual volatility measures how much the portfolio has fluctuated over the sample. '
        'Higher values mean a wider range of outcomes and usually more uncertainty. '
        'A PM may consider diversification, position sizing, or rebalancing if volatility is higher than expected. '
        'This is a standard open-source calculation, so it may not match a proprietary RiskPlus implementation exactly.'
    ),
    'sharpe': (
        'The Sharpe ratio compares return to total volatility after subtracting the risk-free rate. '
        'Higher values generally indicate more return for each unit of risk. '
        'A low or negative Sharpe ratio can suggest the portfolio is not being compensated enough for the risk it is taking. '
        'This is a simplified MVP measure and should be read alongside the rest of the risk report.'
    ),
    'var': (
        'Value at Risk estimates a loss threshold at the chosen confidence level. '
        'A larger magnitude means worse downside risk. '
        'If VaR is higher than expected, a PM may reduce concentration, hedge, or review the tail behavior of the portfolio. '
        'VaR is only a threshold estimate and does not describe losses beyond that point.'
    ),
    'etl': (
        'Expected Tail Loss estimates the average loss in the tail beyond the VaR cutoff. '
        'Higher values mean the left tail is heavier and the portfolio can lose more once the threshold is breached. '
        'A PM may use this to decide whether the portfolio needs better downside protection or lower risk exposure. '
        'This is an open-source tail-risk estimate, not a proprietary RiskPlus tail model.'
    ),
    'etr': (
        'Expected Tail Return measures the average outcome in the favorable tail. '
        'Higher values mean the upside tail is stronger. '
        'A PM may compare this with tail loss to judge whether upside compensation is adequate relative to downside risk. '
        'It is a descriptive metric for the sample, not a guarantee about future upside.'
    ),
    'max_dd': (
        'Maximum drawdown shows the deepest peak-to-trough decline in the historical path. '
        'More negative values mean a worse experience for an investor who stayed invested through the sample. '
        'A PM may use it to assess whether the portfolio can tolerate the size of loss implied by the current strategy. '
        'This metric is path-dependent and reflects only the observed sample period.'
    ),
}

_SIMULATED_METRIC_EXPLANATIONS = {
    'var': (
        'Simulated VaR estimates a downside threshold from the Student-t simulation. '
        'A higher magnitude means the simulated distribution is producing worse tail losses. '
        'A PM might lower risk, raise cash, or test hedges if the simulated tail looks too severe. '
        'Because this is an MVP simulation, it approximates tail behavior rather than reproducing proprietary RiskPlus methodology.'
    ),
    'etl': (
        'Simulated ETL averages the losses that fall beyond the VaR cutoff in the simulation. '
        'Higher values mean the simulated tail is heavier and more expensive in bad states. '
        'A PM may use that signal to reduce concentration or diversify away from correlated exposures. '
        'This remains an approximation built from the selected input data and simulation assumptions.'
    ),
    'starr': (
        'The STARR ratio compares return to expected tail loss instead of total volatility. '
        'Higher values suggest the portfolio is earning more return for each unit of tail risk. '
        'A low value can indicate that the portfolio is not well compensated for its downside exposure. '
        'This is an MVP interpretation and should not be treated as a vendor-equivalent metric.'
    ),
    'rachev': (
        'The Rachev ratio compares favorable tail outcomes with unfavorable tail outcomes. '
        'Higher values are generally better because the upside tail is more attractive relative to the downside tail. '
        'A PM may look at it when judging whether the payoff profile is asymmetric in a useful way. '
        'The metric is useful for comparison, but it still depends on the chosen sample and simulation setup.'
    ),
}


def _lookup_text(mapping: dict[str, str], metric_name: str, fallback: str) -> str:
    key = metric_name.strip().lower()
    return mapping.get(key, fallback)


def explain_historical_risk_metric(metric_name: str) -> str:
    """Return plain-English help text for a historical risk metric."""
    fallback = (
        f'{metric_name} is one of the historical risk outputs in the report. '
        'Higher or lower values should be interpreted in the context of the portfolio objective, '
        'the sample period, and the rest of the risk table. '
        'This is an MVP analysis based on uploaded data, so it should be treated as a directional guide rather than a proprietary replication.'
    )
    return _lookup_text(_HISTORICAL_METRIC_EXPLANATIONS, metric_name, fallback)


def explain_simulated_risk_metric(metric_name: str) -> str:
    """Return plain-English help text for a simulation-based risk metric."""
    fallback = (
        f'{metric_name} is one of the simulated risk outputs. '
        'Higher or lower values indicate the behavior of the Student-t scenario model under the selected assumptions. '
        'A PM may use it to compare scenario risk with the historical sample and decide whether to rebalance or hedge. '
        'The result is an MVP approximation and does not claim proprietary RiskPlus parity.'
    )
    return _lookup_text(_SIMULATED_METRIC_EXPLANATIONS, metric_name, fallback)


def explain_factor_model_section() -> str:
    return (
        'This section summarizes the factor regression that links the portfolio to the selected factors. '
        'Stronger factor significance and higher R² mean the model explains more of the portfolio path, while weak values can mean the chosen factors are incomplete or noisy. '
        'A PM might review factor selection, add missing exposures, or reduce redundant factors if the diagnostics look weak. '
        'This is an open-source linear model and does not recreate every proprietary RiskPlus modeling choice.'
    )


def explain_systematic_specific_risk() -> str:
    return (
        'Systematic risk is the portion tied to the factor model; specific risk is the part left over after the model explains what it can. '
        'A higher systematic share means the portfolio is more driven by market or style exposures, while a higher specific share means stock or asset idiosyncrasies matter more. '
        'A PM may use this to decide whether to diversify, change factor exposure, or accept more benchmark-driven behavior. '
        'This split is a practical approximation built from the available data and regression model.'
    )


def explain_risk_budgeting_etl() -> str:
    return (
        'ETL risk budgeting shows how much each asset contributes to expected tail loss. '
        'Assets with high ETL contributions are making the downside tail worse, even if their average return looks acceptable. '
        'A PM may trim those positions, hedge them, or pair them with diversifying exposures. '
        'The numbers are useful for decision support, but they are still an MVP risk-budgeting estimate rather than a vendor-identical output.'
    )


def explain_risk_budgeting_stdev() -> str:
    return (
        'Standard deviation risk budgeting shows which assets contribute most to day-to-day volatility. '
        'High contributors are driving more total variability in the portfolio, even if they are not the main source of tail risk. '
        'A PM may use this to rebalance, size positions more evenly, or separate volatility reduction from tail-risk reduction. '
        'This is a conventional volatility-based decomposition, not a proprietary RiskPlus calculation.'
    )


def explain_factor_bucket_exposure() -> str:
    return (
        'Factor bucket exposure groups the factor betas into broader market segments so you can see where the portfolio is leaning. '
        'Large positive or negative values mean the portfolio is more sensitive to that bucket, while values near zero suggest little exposure. '
        'A PM may use this to compare the portfolio against intended style bets and to check for unintended concentration. '
        'The bucket mapping is a practical simplification, so it should be treated as a directional interpretation.'
    )


def explain_data_quality() -> str:
    return (
        'The data quality view explains what was loaded, what was cleaned, and what ended up in the analysis. '
        'High missingness, duplicate dates, or large merge losses can reduce confidence in the reported risk outputs. '
        'A PM may revisit the upload format, confirm date alignment, or remove problematic rows before relying on the results. '
        'This view helps explain the MVP pipeline, but it does not change the underlying calculations.'
    )


def explain_output_label(label: str) -> str:
    """Fallback helper for section-specific captions."""
    return (
        f'{label} helps interpret the reported numbers. '
        'Higher or lower values should be read as directional signals rather than precise forecasts. '
        'If the metric looks surprising, a PM should inspect the inputs, the sample period, and any warnings in the data quality section.'
    )
