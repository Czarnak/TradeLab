from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from trade_lab.signals.external import ExternalSignal


def _sample_external_df() -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=5, freq="D")
    return pd.DataFrame({"Close": [100, 101, 102, 103, 104], "VIX": [20, 22, 19, 25, 21]}, index=idx)


def test_external_signal_default_name_is_sanitized_from_column():
    signal = ExternalSignal(column="FED.Rate-Target")

    assert signal.name == "fed_rate_target"
    assert signal.output_columns == ["indicator__fed_rate_target__lag_0"]


def test_external_signal_compute_passthrough_without_normalization():
    df = _sample_external_df()
    signal = ExternalSignal(column="VIX", name="vix_feature")

    out = signal.compute(df.copy())

    col = signal.output_columns[0]
    assert col in out.columns
    pd.testing.assert_series_equal(out[col], df["VIX"], check_names=False)


def test_external_signal_compute_applies_normalization():
    df = _sample_external_df()
    signal = ExternalSignal(column="VIX", normalization=lambda s: (s - 20.0) / 10.0)

    out = signal.compute(df.copy())

    expected = (df["VIX"] - 20.0) / 10.0
    pd.testing.assert_series_equal(out[signal.output_columns[0]], expected, check_names=False)


def test_external_signal_compute_applies_lag_and_renames_output_column():
    df = _sample_external_df()
    signal = ExternalSignal(column="VIX", lag=2)

    out = signal.compute(df.copy())

    raw_col = signal._raw_output_columns[0]
    final_col = signal.output_columns[0]
    assert raw_col not in out.columns
    assert final_col in out.columns
    pd.testing.assert_series_equal(out[final_col], df["VIX"].shift(2), check_names=False)


def test_external_signal_compute_raises_for_missing_column():
    df = pd.DataFrame({"Close": [1, 2, 3]})
    signal = ExternalSignal(column="VIX")

    with pytest.raises(KeyError, match="column 'VIX' not found in DataFrame"):
        signal.compute(df)


def test_external_signal_to_signal_strength_is_not_supported():
    signal = ExternalSignal(column="VIX")

    with pytest.raises(NotImplementedError, match="does not support StandardStrategy"):
        signal.to_signal_strength(pd.DataFrame({"VIX": [1, 2, 3]}))


def test_external_signal_plot_runs_without_error(monkeypatch: pytest.MonkeyPatch):
    df = _sample_external_df()
    signal = ExternalSignal(column="VIX")
    out = signal.compute(df.copy())
    calls = {"show": 0}

    def _show(_self):
        calls["show"] += 1

    monkeypatch.setattr(go.Figure, "show", _show, raising=False)

    signal.plot(out)

    assert calls["show"] == 1
    assert np.isfinite(out[signal.output_columns[0]].to_numpy()).all()
