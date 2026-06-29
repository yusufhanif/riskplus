from __future__ import annotations

import numpy as np
import pandas as pd

from riskplus_core.portfolio import build_portfolio_series, normalize_weights


def test_equal_weight_portfolio_construction() -> None:
    data = pd.DataFrame(
        {
            'A': [0.01, 0.02, 0.03],
            'B': [0.02, 0.01, 0.00],
            'C': [0.00, 0.01, 0.02],
        },
        index=pd.date_range('2020-01-31', periods=3, freq='ME'),
    )

    portfolio, weights = build_portfolio_series(data, ['A', 'B', 'C'])

    expected = data.mean(axis=1)
    assert np.allclose(portfolio.values, expected.values)
    assert np.allclose(weights.values, [1 / 3, 1 / 3, 1 / 3])


def test_supplied_weights_normalize_correctly() -> None:
    weights = normalize_weights(['A', 'B', 'C'], [2.0, 1.0, 1.0])

    assert np.allclose(weights.values, [0.5, 0.25, 0.25])


def test_zero_or_negative_weights_fall_back_to_equal_weights() -> None:
    zero_weights = normalize_weights(['A', 'B', 'C'], [0.0, 0.0, 0.0])
    negative_sum_weights = normalize_weights(['A', 'B', 'C'], [1.0, -1.0, 0.0])

    assert np.allclose(zero_weights.values, [1 / 3, 1 / 3, 1 / 3])
    assert np.allclose(negative_sum_weights.values, [1 / 3, 1 / 3, 1 / 3])