from __future__ import annotations

import numpy as np
import pandas as pd

from riskplus_core.data import merge_analysis_frames, prepare_analysis_data, prepare_factor_stream, prepare_return_stream


def test_return_stream_percent_scales_once() -> None:
    raw = pd.DataFrame({'Date': ['2020-01-31'], 'Return': [5.0]})

    stream = prepare_return_stream(raw, 'Date', 'Return', 'FundA', values_in_percent=True)

    assert stream.iloc[0, 0] == 0.05


def test_return_stream_decimal_stays_decimal() -> None:
    raw = pd.DataFrame({'Date': ['2020-01-31'], 'Return': [0.05]})

    stream = prepare_return_stream(raw, 'Date', 'Return', 'FundA', values_in_percent=False)

    assert stream.iloc[0, 0] == 0.05


def test_merged_prepared_data_scales_once() -> None:
    fund_raw = pd.DataFrame({'Date': ['2020-01-31'], 'Return': [5.0]})
    factor_raw = pd.DataFrame({'Date': ['2020-01-31'], 'Factor1': [2.0]})

    fund_stream = prepare_return_stream(fund_raw, 'Date', 'Return', 'FundA', values_in_percent=True)
    factor_stream = prepare_factor_stream(factor_raw, 'Date', ['Factor1'], values_in_percent=True)

    merged = merge_analysis_frames({'FundA': fund_stream}, factor_stream, join_type='inner')
    prepared = prepare_analysis_data(merged.reset_index(), 'Date', ['FundA'], ['Factor1'], values_in_percent=True)

    assert np.isclose(prepared.loc[pd.Timestamp('2020-01-31'), 'FundA'], 0.05)
    assert np.isclose(prepared.loc[pd.Timestamp('2020-01-31'), 'Factor1'], 0.02)