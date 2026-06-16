"""Excess vol-normalized return target with a pluggable trailing-drift estimator.

The legacy target demeans the forward return by a simple trailing-``drift_window``
mean of daily log-returns. In a persistent bull market that mean lags the true
drift, leaving the label net-positive (``frac>0`` ~0.54) so the model rarely emits
a short-side signal. The ``drift_method`` knob adds EWMA / expanding estimators;
EWMA tracks an accelerating regime more closely, pushing the label distribution
toward zero so short signals become reachable. ``"rolling"`` is the default and
reproduces the legacy behaviour exactly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trade_lab.ml.targets import BaseTarget, ExcessVolNormalizedReturn


def _close_df(prices) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=len(prices), freq="D")
    return pd.DataFrame({"Close": np.asarray(prices, dtype=float)}, index=idx)


def _ramp_uptrend(n: int, noise: float = 0.0008, seed: int = 0) -> pd.DataFrame:
    # Accelerating bull: daily drift ramps up, so a trailing mean lags the recent
    # (higher) drift and under-subtracts it from the forward return.
    rng = np.random.default_rng(seed)
    rets = np.linspace(0.0, 0.004, n) + rng.normal(0.0, noise, n)
    close = 100.0 * np.exp(np.cumsum(rets))
    return _close_df(close)


def test_excess_default_is_rolling_and_named_target():
    target = ExcessVolNormalizedReturn()
    assert target.drift_method == "rolling"
    out = target.generate(_ramp_uptrend(300))
    assert out.name == "target"


def test_rolling_matches_legacy_manual_formula():
    df = _close_df(np.linspace(100.0, 130.0, 40))
    periods, vol_window, drift_window, scale = 1, 3, 5, 1.0
    out = ExcessVolNormalizedReturn(
        periods=periods, vol_window=vol_window, drift_window=drift_window,
        scale=scale, drift_method="rolling",
    ).generate(df)

    close = df["Close"]
    daily = np.log(close / close.shift(1))
    mu_n = daily.rolling(drift_window).mean() * periods
    vol_n = daily.rolling(vol_window).std() * np.sqrt(periods)
    fwd = np.log(close.shift(-periods) / close)
    expected = np.tanh((fwd - mu_n) / (vol_n * scale)).to_numpy()

    np.testing.assert_allclose(out.to_numpy(), expected, equal_nan=True)


def test_ewma_matches_manual_formula():
    df = _close_df(np.linspace(100.0, 130.0, 60))
    periods, vol_window, drift_window, span, scale = 1, 3, 5, 4, 1.0
    out = ExcessVolNormalizedReturn(
        periods=periods, vol_window=vol_window, drift_window=drift_window,
        drift_span=span, scale=scale, drift_method="ewma",
    ).generate(df)

    close = df["Close"]
    daily = np.log(close / close.shift(1))
    mu_n = daily.ewm(span=span, min_periods=drift_window).mean() * periods
    vol_n = daily.rolling(vol_window).std() * np.sqrt(periods)
    fwd = np.log(close.shift(-periods) / close)
    expected = np.tanh((fwd - mu_n) / (vol_n * scale)).to_numpy()

    np.testing.assert_allclose(out.to_numpy(), expected, equal_nan=True)


def test_expanding_matches_manual_formula():
    df = _close_df(np.linspace(100.0, 130.0, 40))
    periods, vol_window, drift_window, scale = 1, 3, 5, 1.0
    out = ExcessVolNormalizedReturn(
        periods=periods, vol_window=vol_window, drift_window=drift_window,
        scale=scale, drift_method="expanding",
    ).generate(df)

    close = df["Close"]
    daily = np.log(close / close.shift(1))
    mu_n = daily.expanding(min_periods=drift_window).mean() * periods
    vol_n = daily.rolling(vol_window).std() * np.sqrt(periods)
    fwd = np.log(close.shift(-periods) / close)
    expected = np.tanh((fwd - mu_n) / (vol_n * scale)).to_numpy()

    np.testing.assert_allclose(out.to_numpy(), expected, equal_nan=True)


def test_ewma_span_defaults_to_drift_window():
    target = ExcessVolNormalizedReturn(drift_window=120, drift_method="ewma")
    assert target.drift_span == 120


def test_normalization_has_no_lookahead():
    # mu_n / vol_n at t use only data up to t, so truncating the series just after
    # t's forward horizon must leave the label at t unchanged.
    df = _ramp_uptrend(80)
    kwargs = dict(
        periods=5, vol_window=4, drift_window=10, drift_method="ewma", drift_span=6
    )
    full = ExcessVolNormalizedReturn(**kwargs).generate(df)

    t = 40  # a fully-warmed row
    truncated = ExcessVolNormalizedReturn(**kwargs).generate(df.iloc[: t + 5 + 1])

    assert truncated.iloc[t] == pytest.approx(full.iloc[t])


def test_ewma_pushes_labels_negative_in_a_bull_market():
    # Core design claim: in an accelerating uptrend, EWMA demeaning subtracts more
    # drift than the lagging trailing mean, so the label distribution shifts toward
    # zero and more labels turn negative -> short signals become reachable.
    df = _ramp_uptrend(500)
    common = dict(periods=5, vol_window=21, drift_window=252)
    roll = ExcessVolNormalizedReturn(**common, drift_method="rolling").generate(df)
    ewma = ExcessVolNormalizedReturn(
        **common, drift_method="ewma", drift_span=63
    ).generate(df)

    frac_neg = lambda s: float(np.mean(s.dropna().to_numpy() < 0))
    assert ewma.mean() < roll.mean()      # strictly more demeaned
    assert frac_neg(ewma) > frac_neg(roll)  # strictly more short-able labels


def test_invalid_drift_method_raises():
    with pytest.raises(ValueError, match="drift_method"):
        ExcessVolNormalizedReturn(drift_method="bogus")


def test_is_base_target_subclass():
    assert issubclass(ExcessVolNormalizedReturn, BaseTarget)
