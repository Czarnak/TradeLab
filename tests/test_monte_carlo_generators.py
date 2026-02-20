import math

import numpy as np
import pandas as pd

from trade_lab.monte_carlo.generators import (
    _close_from_log_returns,
    _default_block_size,
    _log_returns,
    _reconstruct_ohlcv,
    BaseGenerator,
    BlockBootstrap,
    CircularBlockBootstrap,
    GBMSimulator,
    ReturnShuffler,
)


def _sample_ohlcv() -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=8, freq="D")
    return pd.DataFrame(
        {
            "Open": [100, 101, 102, 103, 104, 105, 106, 107],
            "High": [101, 102, 103, 104, 105, 106, 107, 108],
            "Low": [99, 100, 101, 102, 103, 104, 105, 106],
            "Close": [100, 101, 103, 104, 106, 108, 110, 111],
            "Volume": [1000, 1200, 1100, 1300, 1250, 1400, 1350, 1500],
        },
        index=index,
    )


def test_default_block_size_applies_minimum_and_cube_root_rule():
    assert _default_block_size(1) == 2
    assert _default_block_size(27) == 3


def test_log_returns_and_reconstruction_round_trip():
    close = np.array([100.0, 110.0, 121.0])
    log_ret = _log_returns(close)

    reconstructed = _close_from_log_returns(log_ret, initial_close=close[0])

    np.testing.assert_allclose(reconstructed, close)


def test_reconstruct_ohlcv_enforces_constraints_and_skips_invalid_ratios():
    original = _sample_ohlcv().copy()
    original.loc[original.index[2], "Close"] = 0.0
    synthetic_close = np.linspace(90.0, 110.0, num=len(original))

    out = _reconstruct_ohlcv(synthetic_close, original, np.random.default_rng(123))

    assert list(out.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert out.index.equals(original.index)
    np.testing.assert_allclose(out["Close"].to_numpy(), synthetic_close)
    assert np.isfinite(out.to_numpy(dtype=float)).all()
    assert (out["High"] >= np.maximum(out["Open"], out["Close"])).all()
    assert (out["Low"] <= np.minimum(out["Open"], out["Close"])).all()


class _ToyGenerator(BaseGenerator):
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.copy()


def test_base_generator_with_seed_returns_shallow_copy_without_mutating_original():
    gen = _ToyGenerator(seed=7)

    new_gen = gen.with_seed(42)

    assert isinstance(new_gen, _ToyGenerator)
    assert new_gen is not gen
    assert gen.seed == 7
    assert new_gen.seed == 42


def _assert_standard_output_shape(df: pd.DataFrame, out: pd.DataFrame):
    assert out.index.equals(df.index)
    assert list(out.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert len(out) == len(df)
    assert (out["Close"] > 0).all()


def test_return_shuffler_is_reproducible_for_fixed_seed():
    df = _sample_ohlcv()

    out_a = ReturnShuffler(seed=11).generate(df)
    out_b = ReturnShuffler(seed=11).generate(df)

    _assert_standard_output_shape(df, out_a)
    pd.testing.assert_frame_equal(out_a, out_b)


def test_block_bootstrap_supports_default_and_explicit_block_size():
    df = _sample_ohlcv()

    out_default = BlockBootstrap(block_size=None, seed=3).generate(df)
    out_explicit = BlockBootstrap(block_size=2, seed=3).generate(df)

    _assert_standard_output_shape(df, out_default)
    _assert_standard_output_shape(df, out_explicit)
    pd.testing.assert_frame_equal(out_explicit, BlockBootstrap(block_size=2, seed=3).generate(df))


def test_circular_block_bootstrap_is_reproducible_for_fixed_seed():
    df = _sample_ohlcv()

    out_a = CircularBlockBootstrap(block_size=3, seed=9).generate(df)
    out_b = CircularBlockBootstrap(block_size=3, seed=9).generate(df)

    _assert_standard_output_shape(df, out_a)
    pd.testing.assert_frame_equal(out_a, out_b)


def test_gbm_simulator_is_reproducible_and_positive():
    df = _sample_ohlcv()

    out_a = GBMSimulator(seed=5).generate(df)
    out_b = GBMSimulator(seed=5).generate(df)

    _assert_standard_output_shape(df, out_a)
    pd.testing.assert_frame_equal(out_a, out_b)
    assert math.isfinite(float(out_a["Close"].mean()))
