from __future__ import annotations

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


def test_explanation_helpers_return_non_empty_text() -> None:
    explanations = [
        explain_data_quality(),
        explain_factor_model_section(),
        explain_factor_bucket_exposure(),
        explain_systematic_specific_risk(),
        explain_risk_budgeting_etl(),
        explain_risk_budgeting_stdev(),
        explain_historical_risk_metric('ann_mean'),
        explain_historical_risk_metric('var'),
        explain_simulated_risk_metric('var'),
        explain_simulated_risk_metric('starr'),
    ]

    for explanation in explanations:
        assert isinstance(explanation, str)
        assert explanation.strip()
        assert len(explanation.strip()) > 20
