# Graph Report - .  (2026-05-19)

## Corpus Check

- 91 files · ~53,541 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary

- 1101 nodes · 3479 edges · 57 communities detected
- Extraction: 40% EXTRACTED · 60% INFERRED · 0% AMBIGUOUS · INFERRED: 2098 edges (avg confidence: 0.58)
- Token cost: 10,563 input · 1,856 output

## Community Hubs (Navigation)

- [[_COMMUNITY_Base Signal & Position Sizing|Base Signal & Position Sizing]]
- [[_COMMUNITY_Strategy & Backtesting Engine|Strategy & Backtesting Engine]]
- [[_COMMUNITY_Indicators & Feature Matrix|Indicators & Feature Matrix]]
- [[_COMMUNITY_Monte Carlo Analysis|Monte Carlo Analysis]]
- [[_COMMUNITY_Technical Indicators (Bollinger, CCI)|Technical Indicators (Bollinger, CCI)]]
- [[_COMMUNITY_ML Models & Targets|ML Models & Targets]]
- [[_COMMUNITY_Risk Management Base Interface|Risk Management Base Interface]]
- [[_COMMUNITY_External Signals & Indicators|External Signals & Indicators]]
- [[_COMMUNITY_MQL5 Export Infrastructure|MQL5 Export Infrastructure]]
- [[_COMMUNITY_Visualization & Performance Metrics|Visualization & Performance Metrics]]
- [[_COMMUNITY_MQL5 Code Generation & ONNX|MQL5 Code Generation & ONNX]]
- [[_COMMUNITY_ML Optimization (OptunaPruning)|ML Optimization (Optuna/Pruning)]]
- [[_COMMUNITY_Strategy Implementation Examples|Strategy Implementation Examples]]
- [[_COMMUNITY_Lag-Reduced Moving Averages (DEMATEMA)|Lag-Reduced Moving Averages (DEMA/TEMA)]]
- [[_COMMUNITY_Backtesting Pipeline|Backtesting Pipeline]]
- [[_COMMUNITY_ML Lifecycle Components|ML Lifecycle Components]]
- [[_COMMUNITY_Core Signal Components|Core Signal Components]]
- [[_COMMUNITY_Position Sizing Components|Position Sizing Components]]
- [[_COMMUNITY_Moving Averages & Oscillators|Moving Averages & Oscillators]]
- [[_COMMUNITY_Backtesting Report Tests|Backtesting Report Tests]]
- [[_COMMUNITY_External Signal Tests|External Signal Tests]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 28|Community 28]]
- [[_COMMUNITY_Community 29|Community 29]]
- [[_COMMUNITY_Community 30|Community 30]]
- [[_COMMUNITY_Community 31|Community 31]]
- [[_COMMUNITY_Community 32|Community 32]]
- [[_COMMUNITY_Community 34|Community 34]]
- [[_COMMUNITY_Community 35|Community 35]]
- [[_COMMUNITY_Community 36|Community 36]]
- [[_COMMUNITY_Community 37|Community 37]]
- [[_COMMUNITY_Community 40|Community 40]]
- [[_COMMUNITY_Community 41|Community 41]]
- [[_COMMUNITY_Community 42|Community 42]]
- [[_COMMUNITY_Community 43|Community 43]]
- [[_COMMUNITY_Community 44|Community 44]]
- [[_COMMUNITY_Community 45|Community 45]]
- [[_COMMUNITY_Community 46|Community 46]]
- [[_COMMUNITY_Community 47|Community 47]]
- [[_COMMUNITY_Community 48|Community 48]]
- [[_COMMUNITY_Community 49|Community 49]]
- [[_COMMUNITY_Community 50|Community 50]]
- [[_COMMUNITY_Community 51|Community 51]]
- [[_COMMUNITY_Community 52|Community 52]]
- [[_COMMUNITY_Community 53|Community 53]]
- [[_COMMUNITY_Community 54|Community 54]]
- [[_COMMUNITY_Community 55|Community 55]]
- [[_COMMUNITY_Community 56|Community 56]]
- [[_COMMUNITY_Community 57|Community 57]]
- [[_COMMUNITY_Community 58|Community 58]]
- [[_COMMUNITY_Community 59|Community 59]]
- [[_COMMUNITY_Community 60|Community 60]]
- [[_COMMUNITY_Community 61|Community 61]]
- [[_COMMUNITY_Community 62|Community 62]]
- [[_COMMUNITY_Community 63|Community 63]]
- [[_COMMUNITY_Community 64|Community 64]]

## God Nodes (most connected - your core abstractions)
1. `BaseIndicator` - 162 edges
2. `BaseSignal` - 121 edges
3. `BacktestEngine` - 58 edges
4. `FixedPositionSizer` - 57 edges
5. `RSI` - 56 edges
6. `RiskBasedPositionSizer` - 56 edges
7. `SMA` - 55 edges
8. `MLStrategy` - 55 edges
9. `EMA` - 53 edges
10. `MACD` - 51 edges

## Surprising Connections (you probably didn't know these)
- `MLOptimizationResult` --semantically_similar_to--> `OptimizationResult`  [INFERRED] [semantically similar]
  C:\Users\lukas\Documents\RND\TradeLab_FULL\TradeLab\src\trade_lab\ml_optimization\result.py → C:\Users\lukas\Documents\RND\TradeLab_FULL\TradeLab\src\trade_lab\optimization\result.py
- `MLStrategyIntrospector` --semantically_similar_to--> `StrategyIntrospector`  [INFERRED] [semantically similar]
  C:\Users\lukas\Documents\RND\TradeLab_FULL\TradeLab\src\trade_lab\mql5_export\ml_introspector.py → C:\Users\lukas\Documents\RND\TradeLab_FULL\TradeLab\src\trade_lab\mql5_export\introspector.py
- `Generate an HTML backtest report with charts and metrics.      Parameters` --uses--> `BacktestResult`  [INFERRED]
  C:\Users\lukas\Documents\RND\TradeLab_FULL\TradeLab\src\trade_lab\backtesting\report.py → C:\Users\lukas\Documents\RND\TradeLab_FULL\TradeLab\src\trade_lab\backtesting\engine.py
- `Wraps an existing DataFrame column as a first-class ``BaseIndicator``.      De` --uses--> `BaseIndicator`  [INFERRED]
  C:\Users\lukas\Documents\RND\TradeLab_FULL\TradeLab\src\trade_lab\signals\external.py → C:\Users\lukas\Documents\RND\TradeLab_FULL\TradeLab\src\trade_lab\indicators\base.py
- `Write the (optionally normalised) source column to the output column.` --uses--> `BaseIndicator`  [INFERRED]
  C:\Users\lukas\Documents\RND\TradeLab_FULL\TradeLab\src\trade_lab\signals\external.py → C:\Users\lukas\Documents\RND\TradeLab_FULL\TradeLab\src\trade_lab\indicators\base.py

## Hyperedges (group relationships)
- **Backtesting Pipeline** — backtesting_engine, backtesting_result, backtesting_metrics, backtesting_report [EXTRACTED 1.00]
- **ML Lifecycle** — ml_trainer, ml_models, ml_targets, ml_optimization_engine [INFERRED 0.95]
- **MQL5 Export Pipeline** — code_generator_export_to_mql5, validators_validate_strategy, introspector_strategyintrospector, indicator_registry_indicator_registry [EXTRACTED 1.00]
- **Strategy Optimization Framework** — optimizer_optunaoptimizer, objective_objective, param_space_paramspace, result_optimizationresult [EXTRACTED 1.00]
- **Risk Management Base Interface** — base_basetakeprofit, base_basestoploss, base_basetrailingstop [EXTRACTED 1.00]
- **Risk Management Components** — stop_loss_FixedSL, stop_loss_SignalStrengthSL, stop_loss_MovingAverageSL, stop_loss_ParabolicSARSL, take_profit_FixedTP, take_profit_SignalStrengthTP, trailing_stop_FixedTS, trailing_stop_SignalStrengthTS, trailing_stop_MovingAverageTS, trailing_stop_ParabolicSARTS [EXTRACTED 1.00]
- **Signal Components** — signals_base_BaseSignal, signals_external_ExternalSignal, signals_signals_OHLC, signals_signals_HeikinAshi, signals_temporal_CyclicalTemporalSignal [EXTRACTED 1.00]
- **Position Sizing Components** — sizing_base_BasePositionSizer, sizing_fixed_FixedPositionSizer, sizing_risk_based_RiskBasedPositionSizer [EXTRACTED 1.00]
- **Strategy Subpackage** — trade_lab_strategies_base_BaseStrategy, trade_lab_strategies_standard_StandardStrategy, trade_lab_strategies_ml_strategy_MLStrategy [EXTRACTED 0.90]
- **Indicator Verification Tests** — tests_test_indicators_extended, tests_test_indicators_statistical [INFERRED 0.80]
- **Test Suite for v0.8.0** — test_risk_management, test_mql5_export, test_ml_optimization [INFERRED 0.80]

## Communities

### Community 0 - "Base Signal & Position Sizing"
Cohesion: 0.06
Nodes (110): BaseSignal, Map indicator values to [-1.0, 1.0].          Reads from ``self.output_columns, Visualise indicator output on given axes., Column names written by ``_compute()`` — no lag suffix.          Used by the b, Final column names after lag is applied.          When ``lag == 0`` this is id, BasePositionSizer, BaseSignal, FixedPositionSizer (+102 more)

### Community 1 - "Strategy & Backtesting Engine"
Cohesion: 0.04
Nodes (76): BaseStrategy, Input:  Indicators (already computed), position sizing config.     Output: DataF, BaseStrategy, BacktestEngine, BacktestResult, _compute_atr(), Download and return the OHLCV DataFrame without running the backtest., Container for backtest output. (+68 more)

### Community 2 - "Indicators & Feature Matrix"
Cohesion: 0.05
Nodes (64): BaseIndicator, Input:  One or more Signals (computed sequentially before indicator runs)., Compute all upstream signals sequentially., Compute indicator, apply lag shift, and return df.          Subclasses must im, Compute indicator values and write them to ``_raw_output_columns``.          M, Output column name before lag suffix is applied., FeatureMatrix, Feature matrix construction from indicators with built-in lag support.  ``Featur (+56 more)

### Community 3 - "Monte Carlo Analysis"
Cohesion: 0.04
Nodes (65): MonteCarloAnalysis, Statistical analysis of Monte Carlo simulation results.  Provides raw distribu, Percentile rank of a specific value within the simulated distribution., List of metric names present in the results., Return finite values for a metric, raising KeyError if not found., Compute descriptive statistics for a list of metric values., Statistical analysis of Monte Carlo simulation results.      All methods retur, Descriptive statistics for every metric across all simulations.          Param (+57 more)

### Community 4 - "Technical Indicators (Bollinger, CCI)"
Cohesion: 0.03
Nodes (32): Resolve upstream signal chain before computing., BaseIndicator, BollingerBands, CCI, DeMarker, DPO, Bollinger Bands indicator.      Computes an SMA middle band and upper/lower ba, Commodity Channel Index oscillator.      Measures how far price has deviated f (+24 more)

### Community 5 - "ML Models & Targets"
Cohesion: 0.04
Nodes (58): BaseTarget, _DummyStandardScaler, _default_display_names(), _make_function_name(), _make_input_prefix(), dense_model(), directional_loss(), KerasModelWrapper (+50 more)

### Community 6 - "Risk Management Base Interface"
Cohesion: 0.04
Nodes (44): ABC, BasePositionSizer, BaseStopLoss, BaseTakeProfit, BaseTrailingStop, _compute(), output_columns(), plot() (+36 more)

### Community 7 - "External Signals & Indicators"
Cohesion: 0.05
Nodes (47): Compute signal, apply lag shift, and return df.          Subclasses must imple, Compute signal values and write them to ``_raw_output_columns``.          Must, ExternalSignal, Write the (optionally normalised) source column to the output column., Wraps an existing DataFrame column as a first-class ``BaseIndicator``.      De, Not supported — ``ExternalSignal`` is for ``MLStrategy`` only.          Raises, _sample_external_df(), test_external_signal_compute_applies_lag_and_renames_output_column() (+39 more)

### Community 8 - "MQL5 Export Infrastructure"
Cohesion: 0.18
Nodes (40): PriceSource, MQL5ExportResult, MQL5 code generator — orchestrates introspection, rendering and file output.  Ti, Create a Jinja2 environment pointing at the templates directory., Return a flat dict merging IndicatorConfig fields with registry metadata.      T, Format a one-line indicator description for MQL5ExportResult., Format per-layer architecture summaries for MQL5ExportResult.      Parameters, Export a ``StandardStrategy`` to a MetaTrader 5 Expert Advisor ``.mq5`` file. (+32 more)

### Community 9 - "Visualization & Performance Metrics"
Cohesion: 0.11
Nodes (14): _DummyFigure, Plot the output column as a line chart.          Parameters         ---------, compute_metrics(), _direction_stats(), Compute performance metrics from backtest results.      Parameters     ------, Return (win_rate, avg_win, avg_loss) for a direction., _format_metrics_table(), _format_metrics_tables() (+6 more)

### Community 10 - "MQL5 Code Generation & ONNX"
Cohesion: 0.11
Nodes (38): _enrich_indicator(), export_ml_to_mql5(), export_ml_to_mql5_onnx(), export_to_mql5(), _format_indicator_summary(), _format_ml_summary(), _make_jinja_env(), validate_ml_strategy() (+30 more)

### Community 11 - "ML Optimization (Optuna/Pruning)"
Cohesion: 0.11
Nodes (9): _dummy_create_study(), _DummyKerasPruningCallback, _DummyMedianPruner, _DummyScatter, _DummyStudy, _DummyTPESampler, _Logging, _TrialPruned (+1 more)

### Community 12 - "Strategy Implementation Examples"
Cohesion: 0.2
Nodes (14): build_parser(), default_dates(), main(), print_summary(), run_backtest(), BaseIndicator, BaseStopLoss, BaseTakeProfit (+6 more)

### Community 13 - "Lag-Reduced Moving Averages (DEMA/TEMA)"
Cohesion: 0.15
Nodes (4): DEMA, Double Exponential Moving Average indicator.      Reduces the lag of a standard, Triple Exponential Moving Average indicator.      Further reduces EMA lag by app, TEMA

### Community 14 - "Backtesting Pipeline"
Cohesion: 0.33
Nodes (6): BacktestEngine, compute_metrics, generate_report, BacktestResult, MLOptimizer, FeatureMatrix

### Community 15 - "ML Lifecycle Components"
Cohesion: 0.4
Nodes (5): BaseIndicator, ML Models, ModelPruner, ML Targets, MLTrainer

### Community 17 - "Core Signal Components"
Cohesion: 0.5
Nodes (4): Base Signal, Heikin-Ashi Signal, OHLC Log Returns, Cyclical Temporal Signal

### Community 18 - "Position Sizing Components"
Cohesion: 0.67
Nodes (3): Base Position Sizer, Fixed Position Sizer, Risk Based Position Sizer

### Community 20 - "Moving Averages & Oscillators"
Cohesion: 1.0
Nodes (2): Moving Averages, Oscillators

### Community 21 - "Backtesting Report Tests"
Cohesion: 1.0
Nodes (2): Extra Backtesting Report Tests, generate_report

### Community 22 - "External Signal Tests"
Cohesion: 1.0
Nodes (2): External Signal Tests, ExternalSignal

### Community 25 - "Community 25"
Cohesion: 1.0
Nodes (1): Load a previously saved model + metadata.

### Community 26 - "Community 26"
Cohesion: 1.0
Nodes (1): Return a Series of target values aligned with ``df.index``.

### Community 28 - "Community 28"
Cohesion: 1.0
Nodes (1): Generate one synthetic OHLCV scenario.          Parameters         ----------

### Community 29 - "Community 29"
Cohesion: 1.0
Nodes (1): Return the absolute take-profit price for the new position.

### Community 30 - "Community 30"
Cohesion: 1.0
Nodes (1): Return the absolute stop-loss price for the new position.

### Community 31 - "Community 31"
Cohesion: 1.0
Nodes (1): Return the initial trailing-stop price when the trade is opened.

### Community 32 - "Community 32"
Cohesion: 1.0
Nodes (1): Return the updated trailing-stop price for the current bar.

### Community 34 - "Community 34"
Cohesion: 1.0
Nodes (1): Final column names after lag is applied.          When ``lag == 0`` this is id

### Community 35 - "Community 35"
Cohesion: 1.0
Nodes (1): Visualise this signal's output columns.

### Community 36 - "Community 36"
Cohesion: 1.0
Nodes (1): Column names written by ``_compute()`` — no lag suffix.          Used by the b

### Community 37 - "Community 37"
Cohesion: 1.0
Nodes (1): Returns position size in number of units/shares.

### Community 40 - "Community 40"
Cohesion: 1.0
Nodes (1): Kernel Regression

### Community 41 - "Community 41"
Cohesion: 1.0
Nodes (1): Price Source Enum

### Community 42 - "Community 42"
Cohesion: 1.0
Nodes (1): Trade Lab Sizing Module

### Community 43 - "Community 43"
Cohesion: 1.0
Nodes (1): Tests Conftest

### Community 44 - "Community 44"
Cohesion: 1.0
Nodes (1): Backtesting Reporting Tests

### Community 45 - "Community 45"
Cohesion: 1.0
Nodes (1): Extended Indicator Tests

### Community 46 - "Community 46"
Cohesion: 1.0
Nodes (1): Statistical Indicator Tests

### Community 47 - "Community 47"
Cohesion: 1.0
Nodes (1): Test ML Module

### Community 48 - "Community 48"
Cohesion: 1.0
Nodes (1): Test ML Optimization

### Community 49 - "Community 49"
Cohesion: 1.0
Nodes (1): Test Monte Carlo Generators

### Community 50 - "Community 50"
Cohesion: 1.0
Nodes (1): Test Monte Carlo Runner Analysis

### Community 51 - "Community 51"
Cohesion: 1.0
Nodes (1): Test MQL5 Export

### Community 52 - "Community 52"
Cohesion: 1.0
Nodes (1): Test MQL5 Export ML

### Community 53 - "Community 53"
Cohesion: 1.0
Nodes (1): Test MQL5 Export ONNX Exporter

### Community 54 - "Community 54"
Cohesion: 1.0
Nodes (1): Test MQL5 Indicator Registry

### Community 55 - "Community 55"
Cohesion: 1.0
Nodes (1): Test Optimization

### Community 56 - "Community 56"
Cohesion: 1.0
Nodes (1): Test Risk Management

### Community 57 - "Community 57"
Cohesion: 1.0
Nodes (1): Test Signals Indicators Sizing

### Community 58 - "Community 58"
Cohesion: 1.0
Nodes (1): Test Strategies

### Community 59 - "Community 59"
Cohesion: 1.0
Nodes (1): Test Temporal Signal

### Community 60 - "Community 60"
Cohesion: 1.0
Nodes (1): Changelog

### Community 61 - "Community 61"
Cohesion: 1.0
Nodes (1): README

### Community 62 - "Community 62"
Cohesion: 1.0
Nodes (1): Test Probe

### Community 63 - "Community 63"
Cohesion: 1.0
Nodes (1): FRED API Config

### Community 64 - "Community 64"
Cohesion: 1.0
Nodes (1): Instructions Prompt

## Knowledge Gaps
- **126 isolated node(s):** `Average True Range for position sizing volatility input.`, `Return (win_rate, avg_win, avg_loss) for a direction.`, `Compute performance metrics from backtest results.      Parameters     ------`, `Final column names after lag is applied.          When ``lag == 0`` this is id`, `Map indicator values to [-1.0, 1.0].          Reads from ``self.output_columns` (+121 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Moving Averages & Oscillators`** (2 nodes): `Moving Averages`, `Oscillators`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Backtesting Report Tests`** (2 nodes): `Extra Backtesting Report Tests`, `generate_report`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `External Signal Tests`** (2 nodes): `External Signal Tests`, `ExternalSignal`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 25`** (1 nodes): `Load a previously saved model + metadata.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 26`** (1 nodes): `Return a Series of target values aligned with ``df.index``.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 28`** (1 nodes): `Generate one synthetic OHLCV scenario.          Parameters         ----------`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 29`** (1 nodes): `Return the absolute take-profit price for the new position.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 30`** (1 nodes): `Return the absolute stop-loss price for the new position.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 31`** (1 nodes): `Return the initial trailing-stop price when the trade is opened.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 32`** (1 nodes): `Return the updated trailing-stop price for the current bar.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 34`** (1 nodes): `Final column names after lag is applied.          When ``lag == 0`` this is id`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 35`** (1 nodes): `Visualise this signal's output columns.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 36`** (1 nodes): `Column names written by ``_compute()`` — no lag suffix.          Used by the b`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 37`** (1 nodes): `Returns position size in number of units/shares.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 40`** (1 nodes): `Kernel Regression`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 41`** (1 nodes): `Price Source Enum`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 42`** (1 nodes): `Trade Lab Sizing Module`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 43`** (1 nodes): `Tests Conftest`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 44`** (1 nodes): `Backtesting Reporting Tests`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 45`** (1 nodes): `Extended Indicator Tests`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 46`** (1 nodes): `Statistical Indicator Tests`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 47`** (1 nodes): `Test ML Module`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 48`** (1 nodes): `Test ML Optimization`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 49`** (1 nodes): `Test Monte Carlo Generators`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 50`** (1 nodes): `Test Monte Carlo Runner Analysis`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 51`** (1 nodes): `Test MQL5 Export`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 52`** (1 nodes): `Test MQL5 Export ML`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 53`** (1 nodes): `Test MQL5 Export ONNX Exporter`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 54`** (1 nodes): `Test MQL5 Indicator Registry`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 55`** (1 nodes): `Test Optimization`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 56`** (1 nodes): `Test Risk Management`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 57`** (1 nodes): `Test Signals Indicators Sizing`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 58`** (1 nodes): `Test Strategies`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 59`** (1 nodes): `Test Temporal Signal`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 60`** (1 nodes): `Changelog`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 61`** (1 nodes): `README`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 62`** (1 nodes): `Test Probe`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 63`** (1 nodes): `FRED API Config`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 64`** (1 nodes): `Instructions Prompt`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `BaseIndicator` connect `Indicators & Feature Matrix` to `Base Signal & Position Sizing`, `Strategy & Backtesting Engine`, `Technical Indicators (Bollinger, CCI)`, `ML Models & Targets`, `Risk Management Base Interface`, `External Signals & Indicators`, `MQL5 Export Infrastructure`, `Visualization & Performance Metrics`, `Lag-Reduced Moving Averages (DEMA/TEMA)`?**
  _High betweenness centrality (0.285) - this node is a cross-community bridge._
- **Why does `BaseSignal` connect `Base Signal & Position Sizing` to `Strategy & Backtesting Engine`, `Indicators & Feature Matrix`, `Technical Indicators (Bollinger, CCI)`, `Risk Management Base Interface`, `External Signals & Indicators`, `MQL5 Export Infrastructure`, `Lag-Reduced Moving Averages (DEMA/TEMA)`?**
  _High betweenness centrality (0.109) - this node is a cross-community bridge._
- **Why does `BacktestEngine` connect `Strategy & Backtesting Engine` to `Indicators & Feature Matrix`, `Monte Carlo Analysis`, `Strategy Implementation Examples`, `Risk Management Base Interface`?**
  _High betweenness centrality (0.097) - this node is a cross-community bridge._
- **Are the 155 inferred relationships involving `BaseIndicator` (e.g. with `BaseSignal` and `BaseMA`) actually correct?**
  _`BaseIndicator` has 155 INFERRED edges - model-reasoned connections that need verification._
- **Are the 115 inferred relationships involving `BaseSignal` (e.g. with `BaseIndicator` and `Input:  One or more Signals (computed sequentially before indicator runs).`) actually correct?**
  _`BaseSignal` has 115 INFERRED edges - model-reasoned connections that need verification._
- **Are the 51 inferred relationships involving `BacktestEngine` (e.g. with `BaseStrategy` and `MLObjective`) actually correct?**
  _`BacktestEngine` has 51 INFERRED edges - model-reasoned connections that need verification._
- **Are the 52 inferred relationships involving `FixedPositionSizer` (e.g. with `SignalConfig` and `IndicatorConfig`) actually correct?**
  _`FixedPositionSizer` has 52 INFERRED edges - model-reasoned connections that need verification._