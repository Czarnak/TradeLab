# TradeLab — Bug Fix Prompt

## Context

You are working on **TradeLab**, a modular Python backtesting framework.
The project is structured under `src/trade_lab/` with the following relevant
modules:

```
src/trade_lab/
  backtesting/
    engine.py
    metrics.py
  optimization/
    optimizer.py
    param_space.py
    objective.py
    result.py
  monte_carlo/
    generators.py
    runner.py
    analysis.py
```

This prompt describes **four bugs** to fix. Each section gives the exact file,
the problem, the root cause, and the correct fix to apply. No other changes
should be made. After applying all fixes, update CHANGELOG.md with a patch
entry (v0.3.1).

---

## Fix 1 — Critical: wrong optimisation direction for signed metrics

**File:** `src/trade_lab/optimization/optimizer.py`

**Problem:**
The `_MINIMIZE_METRICS` set includes `max_drawdown`, `avg_loss`,
`long_avg_loss`, and `short_avg_loss`. However, all four of these metrics are
stored as **negative numbers** by `compute_metrics`:

- `max_drawdown` is computed as `drawdown.min()` where
  `drawdown = (equity - peak) / peak` — always ≤ 0.
  Example: a 25% drawdown is stored as `-0.25`.
- `avg_loss`, `long_avg_loss`, `short_avg_loss` are mean PnL values of losing
  trades — always ≤ 0.

When Optuna **minimises** a negative metric, it searches for the *most negative*
value — i.e. the **worst** drawdown or the **biggest** loss. This is exactly
backwards.

`annual_volatility` and `total_commission` are genuinely positive numbers, so
minimising those is correct.

**Fix:**
Remove `max_drawdown`, `avg_loss`, `long_avg_loss`, `short_avg_loss` from
`_MINIMIZE_METRICS`. The corrected set should be:

```python
_MINIMIZE_METRICS = {
    'annual_volatility',
    'total_commission',
}
```

Optuna will then **maximise** the signed-negative metrics, which means
searching for the value closest to zero — the correct behaviour (smallest
drawdown, smallest average loss).

---

## Fix 2 — Per-simulation exception handling in MonteCarloRunner

**File:** `src/trade_lab/monte_carlo/runner.py`

**Problem:**
In `MonteCarloRunner._run_single_simulation()`, if a synthetic OHLCV series
causes any exception (e.g. a GBM path collapsing near zero triggers a
division error in an indicator, or a degenerate series produces all-NaN
signals), the exception propagates uncaught and kills the entire simulation
run. All completed simulations are lost.

**Fix:**
Wrap the body of `_run_single_simulation` in a `try/except Exception` block.
On failure, log a warning to stderr and return a dict of NaN values for all
known metrics. Use the existing `metric_series` keys if any simulations have
already completed, or return an empty dict if this is the very first
simulation. This matches the existing convention where `None`/NaN metric
values are already handled gracefully by the caller.

Replace the current `_run_single_simulation` method:

```python
def _run_single_simulation(
    self,
    df: pd.DataFrame,
    simulation_index: int,
) -> dict:
    base_seed = self.generator.seed
    sim_seed = None if base_seed is None else base_seed + simulation_index
    sim_generator = self.generator.with_seed(sim_seed)

    try:
        synthetic_df = sim_generator.generate(df)
        backtest_result = self.engine.run_on(synthetic_df)
        return backtest_result.metrics
    except Exception as exc:
        import sys
        print(
            f"[MonteCarloRunner] simulation {simulation_index} failed "
            f"({type(exc).__name__}: {exc}) — recording NaN for all metrics.",
            file=sys.stderr,
        )
        # Return NaN for every metric key seen so far, so the caller can
        # still append a consistent entry to metric_series.
        return {key: float('nan') for key in self._known_metric_keys}

@property
def _known_metric_keys(self) -> list[str]:
    """Metric keys seen in at least one completed simulation.
    Returns an empty list before any simulation has completed."""
    return []  # See implementation note below
```

**Implementation note:** The cleanest implementation threads a mutable
`_seen_keys: set[str]` set through `run()` rather than storing it on the
instance. Refactor `run()` and `_run_single_simulation()` as follows:

```python
def run(self, df: pd.DataFrame) -> MonteCarloResult:
    metric_series: dict[str, list[float]] = {}
    seen_keys: set[str] = set()

    indices = range(self.n_simulations)
    iterator = self._make_iterator(indices)

    for i in iterator:
        sim_result = self._run_single_simulation(df, i, seen_keys)

        selected = (
            {k: v for k, v in sim_result.items() if k in self.metrics}
            if self.metrics is not None
            else sim_result
        )

        for key, value in selected.items():
            seen_keys.add(key)
            metric_series.setdefault(key, []).append(
                float(value) if value is not None else float('nan')
            )

    return MonteCarloResult(
        n_simulations=self.n_simulations,
        metric_series=metric_series,
        generator_name=type(self.generator).__name__,
    )

def _run_single_simulation(
    self,
    df: pd.DataFrame,
    simulation_index: int,
    seen_keys: set[str],
) -> dict:
    base_seed = self.generator.seed
    sim_seed = None if base_seed is None else base_seed + simulation_index
    sim_generator = self.generator.with_seed(sim_seed)

    try:
        synthetic_df = sim_generator.generate(df)
        backtest_result = self.engine.run_on(synthetic_df)
        return backtest_result.metrics
    except Exception as exc:
        import sys
        print(
            f"[MonteCarloRunner] simulation {simulation_index} failed "
            f"({type(exc).__name__}: {exc}) — recording NaN for all metrics.",
            file=sys.stderr,
        )
        return {key: float('nan') for key in seen_keys}
```

---

## Fix 3 — CircularBlockBootstrap does not guard against oversized block_size

**File:** `src/trade_lab/monte_carlo/generators.py`

**Problem:**
`CircularBlockBootstrap.generate()` tiles the return series as
`circular = np.tile(log_ret, 2)` and then samples blocks of length
`block_size` starting at positions in `[0, n)`.

If `block_size >= n`, the slice `circular[start:start + block_size]` for
`start` values near `n-1` will return fewer than `block_size` elements
(the doubled array is only `2n` long). The resampled series silently gets
truncated, producing incorrect results with no error.

**Fix:**
Add a validation guard at the start of `generate()`, before the block size
is resolved from `_default_block_size`. If `block_size >= n`, raise a
`ValueError` with a clear message.

```python
def generate(self, df: pd.DataFrame) -> pd.DataFrame:
    rng = self._make_rng()
    close = df['Close'].to_numpy(dtype=float)
    log_ret = _log_returns(close)
    n = len(log_ret)

    block_size = (
        self.block_size
        if self.block_size is not None
        else _default_block_size(n)
    )

    if block_size >= n:
        raise ValueError(
            f"CircularBlockBootstrap: block_size ({block_size}) must be "
            f"less than the number of return observations ({n}). "
            f"Reduce block_size or use a longer data series."
        )

    # ... rest of the method unchanged
```

---

## Fix 4 — README uses `strategy=None` misleadingly

**File:** `README.md`  
**File:** `src/trade_lab/backtesting/engine.py`

**Problem:**
The optimization example in README.md shows:

```python
engine = BacktestEngine(strategy=None, ticker='SPY',
                        start='2015-01-01', end='2024-01-01')
full_df = engine.fetch_data()
```

This works at runtime because `fetch_data()` never calls `strategy`, but
`strategy=None` is semantically wrong and would cause an `AttributeError`
if the user then called `run()` or `run_on()` on the same engine. It also
looks like an omission/mistake in the docs.

**Fix — part A (`engine.py`):**
Make `strategy` optional (default `None`) in `BacktestEngine.__init__()`.
Add a guard in `run_on()` and `run()` that raises `ValueError` if `strategy`
is `None` when those methods are called. Update the docstring accordingly.

```python
def __init__(
    self,
    strategy: BaseStrategy | None = None,
    ticker: str | None = None,
    start: str | None = None,
    end: str | None = None,
    initial_capital: float = 100_000.0,
    commission: float = 0.001,
    slippage: float = 0.0005,
):
    ...
```

In `run()` and `run_on()`, add at the top:

```python
if self.strategy is None:
    raise ValueError(
        "BacktestEngine.run()/run_on() requires a strategy. "
        "Set strategy= at construction or assign engine.strategy before calling run."
    )
```

**Fix — part B (`README.md`):**
Update the optimization section example to use a dedicated engine instance
for data fetching only, making the intent explicit:

```python
# Fetch data once using a minimal engine (strategy not needed for fetch)
data_engine = BacktestEngine(ticker='SPY', start='2015-01-01', end='2024-01-01')
full_df = data_engine.fetch_data()
```

---

## CHANGELOG entry

Add the following entry at the top of CHANGELOG.md (above the `[0.3.0]` entry):

```markdown
## [0.3.1] - 2026-02-20

### Fixed

- `optimization/optimizer.py`: `_MINIMIZE_METRICS` incorrectly included
  `max_drawdown`, `avg_loss`, `long_avg_loss`, `short_avg_loss` — all of which
  are stored as negative numbers by `compute_metrics`. Optuna was minimising
  them, searching for the worst possible values. Removed from the set; these
  metrics are now correctly maximised (value closest to zero = best).
  `annual_volatility` and `total_commission` remain in `_MINIMIZE_METRICS`
  as they are genuinely positive.
- `monte_carlo/runner.py`: a single failing simulation (e.g. degenerate
  synthetic series) crashed the entire run and discarded all completed results.
  Added per-simulation `try/except` in `_run_single_simulation`; failures now
  record NaN for all metrics and print a warning to stderr.
- `monte_carlo/generators.py`: `CircularBlockBootstrap.generate()` silently
  produced incorrect results when `block_size >= n` (number of return
  observations). Added an explicit `ValueError` guard.
- `backtesting/engine.py`: `strategy` parameter is now optional (`None` by
  default). A clear `ValueError` is raised if `run()` or `run_on()` is called
  without a strategy set. Allows constructing an engine solely for
  `fetch_data()` without passing a placeholder `strategy=None`.
- `README.md`: updated optimization example to use a dedicated engine instance
  for `fetch_data()`, making the pattern explicit and semantically correct.
```

---
