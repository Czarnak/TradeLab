import numpy as np
import pandas as pd
import pytest

from trade_lab.signals.temporal import CyclicalTemporalSignal


def test_temporal_signal_adds_expected_columns_and_values_for_month_component():
    index = pd.to_datetime(["2026-01-01", "2026-04-01", "2026-07-01"])
    df = pd.DataFrame({"close": [1.0, 2.0, 3.0]}, index=index)
    signal = CyclicalTemporalSignal(component="month", period=12)

    result = signal.compute(df.copy())

    sin_col, cos_col = signal.output_columns
    expected_angle = 2 * np.pi * result.index.month / 12
    np.testing.assert_allclose(result[sin_col].to_numpy(), np.sin(expected_angle))
    np.testing.assert_allclose(result[cos_col].to_numpy(), np.cos(expected_angle))


def test_temporal_signal_formats_output_column_names_for_int_and_float_periods():
    int_period = CyclicalTemporalSignal(component="day_of_year", period=90)
    float_period = CyclicalTemporalSignal(component="day_of_year", period=90.5)

    assert int_period.output_columns == [
        "signal__day_of_year_p90_sin",
        "signal__day_of_year_p90_cos",
    ]
    assert float_period.output_columns == [
        "signal__day_of_year_p90.5_sin",
        "signal__day_of_year_p90.5_cos",
    ]


def test_temporal_signal_rejects_unknown_component():
    with pytest.raises(ValueError, match="Unknown component"):
        CyclicalTemporalSignal(component="quarter", period=4)


def test_temporal_signal_rejects_non_positive_period():
    with pytest.raises(ValueError, match="period must be positive"):
        CyclicalTemporalSignal(component="month", period=0)


def test_temporal_signal_plot_runs_without_error():
    index = pd.date_range("2026-01-01", periods=4, freq="D")
    df = pd.DataFrame({"close": [1.0, 2.0, 3.0, 4.0]}, index=index)
    signal = CyclicalTemporalSignal(component="day_of_week", period=7)
    out = signal.compute(df.copy())

    signal.plot(out)
