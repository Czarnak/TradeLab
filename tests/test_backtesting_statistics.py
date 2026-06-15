"""Statistical-significance helpers for honest strategy evaluation.

Covers probabilistic / deflated Sharpe (Bailey & Lopez de Prado), block-bootstrap
Sharpe confidence intervals, per-regime breakdown, baseline promotion gates and a
minimum-trade gate. No SciPy dependency — the normal CDF/inverse come from the
stdlib ``statistics.NormalDist``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trade_lab.backtesting.statistics import (
    block_bootstrap_sharpe_ci,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    min_trade_gate,
    probabilistic_sharpe_ratio,
    promotion_gates,
    regime_breakdown,
)


# ---------------------------------------------------------------------------
# Probabilistic Sharpe ratio
# ---------------------------------------------------------------------------


def test_psr_is_one_half_when_benchmark_equals_observed():
    # benchmark == observed => z == 0 => PSR == 0.5 exactly.
    psr = probabilistic_sharpe_ratio(0.1, n_obs=101, benchmark_sr=0.1)
    assert psr == pytest.approx(0.5, abs=1e-12)


def test_psr_rises_as_observed_exceeds_benchmark_and_with_sample_size():
    low = probabilistic_sharpe_ratio(0.1, n_obs=101, benchmark_sr=0.0)
    high_n = probabilistic_sharpe_ratio(0.1, n_obs=401, benchmark_sr=0.0)
    assert 0.5 < low < 1.0
    assert high_n > low  # more observations => more confident


def test_psr_matches_closed_form_for_gaussian_returns():
    # SR=0.1, N=101, skew=0, kurt=3: z = 0.1*sqrt(100)/sqrt(1+0.5*0.01).
    psr = probabilistic_sharpe_ratio(0.1, n_obs=101, benchmark_sr=0.0)
    assert psr == pytest.approx(0.8407, abs=2e-3)


# ---------------------------------------------------------------------------
# Expected max Sharpe (multiple-testing hurdle) + Deflated Sharpe
# ---------------------------------------------------------------------------


def test_expected_max_sharpe_grows_with_trials_and_variance():
    assert expected_max_sharpe(sr_variance=0.04, n_trials=2) < expected_max_sharpe(
        sr_variance=0.04, n_trials=50
    )
    assert expected_max_sharpe(sr_variance=0.01, n_trials=50) < expected_max_sharpe(
        sr_variance=0.04, n_trials=50
    )


def test_expected_max_sharpe_is_zero_without_multiple_testing():
    assert expected_max_sharpe(sr_variance=0.04, n_trials=1) == 0.0


def test_deflated_sharpe_equals_psr_when_single_trial():
    dsr = deflated_sharpe_ratio(0.1, n_obs=101, sr_variance=0.04, n_trials=1)
    psr0 = probabilistic_sharpe_ratio(0.1, n_obs=101, benchmark_sr=0.0)
    assert dsr == pytest.approx(psr0)


def test_deflated_sharpe_is_penalised_by_many_trials():
    dsr = deflated_sharpe_ratio(0.1, n_obs=101, sr_variance=0.04, n_trials=55)
    psr0 = probabilistic_sharpe_ratio(0.1, n_obs=101, benchmark_sr=0.0)
    assert dsr < psr0  # the multiple-testing hurdle deflates confidence


# ---------------------------------------------------------------------------
# Block-bootstrap Sharpe CI
# ---------------------------------------------------------------------------


def _deterministic_returns() -> np.ndarray:
    # Deterministic, has variance and a slight positive drift.
    return np.sin(np.arange(252) / 5.0) * 0.01 + 0.0005


def test_block_bootstrap_point_estimate_matches_annualised_sharpe():
    r = _deterministic_returns()
    expected = r.mean() / r.std(ddof=1) * np.sqrt(252)

    result = block_bootstrap_sharpe_ci(r, block_size=10, n_boot=200, seed=42)

    assert result.sharpe == pytest.approx(expected)
    assert result.ci_low <= result.ci_high


def test_block_bootstrap_is_reproducible_with_seed():
    r = _deterministic_returns()
    a = block_bootstrap_sharpe_ci(r, block_size=10, n_boot=200, seed=7)
    b = block_bootstrap_sharpe_ci(r, block_size=10, n_boot=200, seed=7)
    assert (a.ci_low, a.ci_high) == (b.ci_low, b.ci_high)


# ---------------------------------------------------------------------------
# Regime breakdown
# ---------------------------------------------------------------------------


def test_regime_breakdown_computes_metrics_per_subset():
    idx = pd.date_range("2026-01-01", periods=5, freq="D")
    returns = pd.Series([0.01, 0.02, 0.01, -0.01, -0.02], index=idx)
    regimes = pd.Series(["calm", "calm", "calm", "stress", "stress"], index=idx)

    table = regime_breakdown(returns, regimes)

    assert set(table.index) == {"calm", "stress"}
    assert table.loc["calm", "n"] == 3
    assert table.loc["stress", "n"] == 2
    assert table.loc["calm", "sharpe"] > 0
    assert table.loc["stress", "sharpe"] < 0


# ---------------------------------------------------------------------------
# Promotion gates
# ---------------------------------------------------------------------------


def test_promotion_gates_pass_only_when_strategy_beats_baseline():
    table = promotion_gates(
        strategy_value=0.5,
        baselines={"buy_hold": 0.3, "always_long": 0.6},
    )

    table = table.set_index("baseline")
    assert bool(table.loc["buy_hold", "passed"]) is True
    assert bool(table.loc["always_long", "passed"]) is False


# ---------------------------------------------------------------------------
# Minimum-trade gate
# ---------------------------------------------------------------------------


def test_min_trade_gate_flags_insufficient_samples():
    assert min_trade_gate(n_trades=11, min_trades=30).passed is False
    assert min_trade_gate(n_trades=30, min_trades=30).passed is True
