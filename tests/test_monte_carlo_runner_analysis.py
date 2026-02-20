from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import trade_lab.monte_carlo as mc
import trade_lab.monte_carlo.runner as mc_runner
from trade_lab.monte_carlo.analysis import MonteCarloAnalysis
from trade_lab.monte_carlo.generators import BaseGenerator
from trade_lab.monte_carlo.runner import MonteCarloResult, MonteCarloRunner


def _sample_ohlcv() -> pd.DataFrame:
    index = pd.date_range("2026-02-01", periods=5, freq="D")
    return pd.DataFrame(
        {
            "Open": [10, 11, 12, 13, 14],
            "High": [11, 12, 13, 14, 15],
            "Low": [9, 10, 11, 12, 13],
            "Close": [10, 11, 12, 13, 14],
            "Volume": [100, 110, 120, 130, 140],
        },
        index=index,
    )


class _TrackingGenerator(BaseGenerator):
    def __init__(self, seed=None):
        super().__init__(seed=seed)
        self.seen_seeds = []

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        self.seen_seeds.append(self.seed)
        return df.copy()


class _DummyEngine:
    def __init__(self):
        self.calls = 0
        self.seen_inputs = []

    def run_on(self, synthetic_df: pd.DataFrame):
        self.calls += 1
        self.seen_inputs.append(synthetic_df.copy())
        return SimpleNamespace(
            metrics={
                "score": self.calls,
                "maybe_none": None if self.calls == 1 else self.calls * 10,
            }
        )


def test_module_exports_expected_public_api():
    assert "MonteCarloRunner" in mc.__all__
    assert "MonteCarloAnalysis" in mc.__all__
    assert hasattr(mc, "BlockBootstrap")
    assert hasattr(mc, "GBMSimulator")


def test_runner_run_collects_metrics_converts_none_to_nan_and_advances_seed():
    df = _sample_ohlcv()
    engine = _DummyEngine()
    generator = _TrackingGenerator(seed=100)
    runner = MonteCarloRunner(engine=engine, generator=generator, n_simulations=3, verbose=False)

    result = runner.run(df)

    assert result.n_simulations == 3
    assert result.generator_name == "_TrackingGenerator"
    assert result.metric_series["score"] == [1.0, 2.0, 3.0]
    assert np.isnan(result.metric_series["maybe_none"][0])
    assert result.metric_series["maybe_none"][1:] == [20.0, 30.0]
    assert generator.seen_seeds == [100, 101, 102]
    assert engine.calls == 3


def test_runner_run_filters_metrics_when_subset_is_configured():
    df = _sample_ohlcv()
    engine = _DummyEngine()
    generator = _TrackingGenerator(seed=1)
    runner = MonteCarloRunner(
        engine=engine,
        generator=generator,
        n_simulations=2,
        metrics=["score"],
        verbose=False,
    )

    result = runner.run(df)

    assert list(result.metric_series.keys()) == ["score"]
    assert result.metric_series["score"] == [1.0, 2.0]


def test_runner_make_iterator_returns_range_when_verbose_disabled():
    runner = MonteCarloRunner(_DummyEngine(), _TrackingGenerator(seed=1), n_simulations=2, verbose=False)

    iterator = runner._make_iterator(range(2))

    assert list(iterator) == [0, 1]


def test_runner_make_iterator_uses_tqdm_when_available(monkeypatch):
    called = {}

    def fake_tqdm(indices, total, desc, unit, dynamic_ncols):
        called["total"] = total
        called["desc"] = desc
        called["unit"] = unit
        called["dynamic_ncols"] = dynamic_ncols
        return list(indices)

    monkeypatch.setattr(mc_runner, "_TQDM_AVAILABLE", True)
    monkeypatch.setattr(mc_runner, "_tqdm", fake_tqdm)

    runner = MonteCarloRunner(_DummyEngine(), _TrackingGenerator(seed=1), n_simulations=2, verbose=True)
    iterator = runner._make_iterator(range(2))

    assert iterator == [0, 1]
    assert called == {
        "total": 2,
        "desc": "Monte Carlo (_TrackingGenerator)",
        "unit": "sim",
        "dynamic_ncols": True,
    }


def test_runner_make_iterator_uses_fallback_when_tqdm_unavailable(monkeypatch, capsys):
    monkeypatch.setattr(mc_runner, "_TQDM_AVAILABLE", False)

    runner = MonteCarloRunner(_DummyEngine(), _TrackingGenerator(seed=1), n_simulations=3, verbose=True)
    iterator = runner._make_iterator(range(3))

    assert isinstance(iterator, mc_runner._FallbackProgress)
    assert list(iterator) == [0, 1, 2]

    captured = capsys.readouterr().out
    assert "Monte Carlo (_TrackingGenerator): 3/3 simulations complete (100%)" in captured


def test_run_single_simulation_handles_none_seed_and_returns_metrics_dict():
    df = _sample_ohlcv()
    engine = _DummyEngine()
    generator = _TrackingGenerator(seed=None)
    runner = MonteCarloRunner(engine=engine, generator=generator, n_simulations=1, verbose=False)

    metrics = runner._run_single_simulation(df, simulation_index=9)

    assert metrics["score"] == 1
    assert generator.seen_seeds == [None]


def _analysis_result() -> MonteCarloResult:
    return MonteCarloResult(
        n_simulations=4,
        metric_series={
            "sharpe": [1.0, 2.0, np.nan, 3.0],
            "all_nan": [np.nan, np.nan, np.nan, np.nan],
        },
        generator_name="DummyGen",
    )


def test_analysis_summary_default_and_custom_percentiles_and_distributions():
    analysis = MonteCarloAnalysis(_analysis_result())

    summary_default = analysis.summary()
    summary_custom = analysis.summary(percentiles=[10, 90])
    distributions = analysis.distributions()

    assert "p5" in summary_default.columns
    assert "p95" in summary_default.columns
    assert "p10" in summary_custom.columns
    assert "p90" in summary_custom.columns
    assert summary_default.loc["sharpe", "n_valid"] == 3
    assert summary_default.loc["sharpe", "n_nan"] == 1
    assert np.isnan(summary_default.loc["all_nan", "mean"])
    assert summary_default.loc["all_nan", "n_valid"] == 0
    assert summary_default.loc["all_nan", "n_nan"] == 4
    np.testing.assert_allclose(distributions["sharpe"][:2], np.array([1.0, 2.0]))
    assert np.isnan(distributions["sharpe"][2])


def test_analysis_confidence_interval_percentile_of_available_metrics_and_keyerror():
    analysis = MonteCarloAnalysis(_analysis_result())

    ci = analysis.confidence_interval("sharpe", lower=0, upper=100)
    ci_all_nan = analysis.confidence_interval("all_nan")
    percentile = analysis.percentile_of("sharpe", 2.0)
    percentile_all_nan = analysis.percentile_of("all_nan", 42.0)

    assert ci == (1.0, 3.0)
    assert np.isnan(ci_all_nan[0]) and np.isnan(ci_all_nan[1])
    assert pytest.approx(percentile, rel=1e-12) == (2 / 3) * 100
    assert np.isnan(percentile_all_nan)
    assert analysis.available_metrics() == ["sharpe", "all_nan"]

    with pytest.raises(KeyError, match="Metric 'missing' not found"):
        analysis.confidence_interval("missing")

def test_runner_import_path_handles_missing_tqdm(monkeypatch):
    import builtins
    import importlib.util
    import sys
    from pathlib import Path

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "tqdm":
            raise ImportError("forced missing tqdm")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    module_name = "runner_no_tqdm_for_test"
    module_path = Path(mc_runner.__file__)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    assert module._TQDM_AVAILABLE is False

