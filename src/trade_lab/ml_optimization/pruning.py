"""Post-training weight pruning and dead feature detection.

``ModelPruner`` zeroes small weights in a trained Keras model, detects
features that have become fully dead (all weights in the first Dense layer
zeroed), and rebuilds the strategy with a leaner model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from trade_lab.ml_optimization.result import MLOptimizationResult


@dataclass
class PruningReport:
    """Summary of a single pruning pass."""

    zero_fraction: float
    dead_features: list[str]
    surviving_features: list[str]


class ModelPruner:
    """Post-training weight pruning for Keras Dense models.

    Two modes (select via constructor):

    * **Global threshold** — one percentile cutoff across *all* layer weights.
    * **Per-layer percentile** — bottom ``percentile``% zeroed independently
      per Dense layer.

    Parameters
    ----------
    percentile : float
        Percentage of weights to zero (0–100).  E.g. ``20`` zeros the
        smallest 20% of weights by absolute value.
    per_layer : bool
        If ``True``, apply the percentile independently to each Dense layer.
        If ``False`` (default), compute a single global threshold.
    """

    def __init__(self, percentile: float = 10.0, per_layer: bool = False) -> None:
        if not 0 <= percentile <= 100:
            raise ValueError(f"percentile must be in [0, 100], got {percentile}")
        self.percentile = percentile
        self.per_layer = per_layer

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prune_model(
        self,
        model: Any,
        feature_names: list[str],
    ) -> tuple[Any, dict]:
        """Zero small weights in a Keras model and detect dead features.

        Parameters
        ----------
        model : keras.Model
            Trained model to prune in-place.
        feature_names : list[str]
            Feature column names matching the first Dense layer's input dim.

        Returns
        -------
        tuple[keras.Model, dict]
            ``(pruned_model, report)`` where report contains
            ``zero_fraction``, ``dead_features``, and ``surviving_features``.
        """
        import keras

        total_weights = 0
        total_zeroed = 0
        first_dense_kernel: np.ndarray | None = None
        first_dense_layer: Any = None

        if not self.per_layer:
            # Collect all kernel weights for global threshold
            all_kernels: list[np.ndarray] = []
            for layer in model.layers:
                if isinstance(layer, keras.layers.Dense):
                    all_kernels.append(layer.get_weights()[0].ravel())
            if all_kernels:
                global_vals = np.concatenate(all_kernels)
                threshold = np.percentile(np.abs(global_vals), self.percentile)
            else:
                threshold = 0.0

        for layer in model.layers:
            if not isinstance(layer, keras.layers.Dense):
                continue
            ws = layer.get_weights()
            kernel = ws[0]

            if self.per_layer:
                threshold = np.percentile(np.abs(kernel), self.percentile)

            mask = np.abs(kernel) >= threshold
            zeroed = (~mask).sum()
            total_weights += kernel.size
            total_zeroed += zeroed
            kernel[~mask] = 0.0
            ws[0] = kernel
            layer.set_weights(ws)

            if first_dense_layer is None:
                first_dense_kernel = kernel
                first_dense_layer = layer

        # Dead feature detection — features where every weight in the first
        # Dense layer kernel row is zero after pruning.
        dead_features: list[str] = []
        surviving_features: list[str] = list(feature_names)
        if first_dense_kernel is not None and len(feature_names) == first_dense_kernel.shape[0]:
            for i, name in enumerate(feature_names):
                if np.all(first_dense_kernel[i, :] == 0.0):
                    dead_features.append(name)
            surviving_features = [f for f in feature_names if f not in dead_features]

        report = {
            'zero_fraction': total_zeroed / total_weights if total_weights else 0.0,
            'dead_features': dead_features,
            'surviving_features': surviving_features,
        }
        return model, report

    def prune_result(
        self,
        result: MLOptimizationResult,
        fine_tune_epochs: int = 5,
    ) -> tuple[MLOptimizationResult, dict]:
        """Prune the best model from an ``MLOptimizationResult``.

        Prunes the best model, filters dead-feature indicators, refits the
        scaler on training data, fine-tunes surviving weights, and rebuilds
        the strategy.

        Parameters
        ----------
        result : MLOptimizationResult
            Completed optimisation result to prune.
        fine_tune_epochs : int
            Number of epochs for fine-tuning after pruning.

        Returns
        -------
        tuple[MLOptimizationResult, dict]
            ``(updated_result, report)`` with pruned model, filtered feature
            spec, refitted scaler, and rebuilt strategy.
        """
        import copy

        import keras

        from trade_lab.indicators.base import BaseIndicator
        from trade_lab.ml_optimization.feature_builder import FeatureMatrix
        from trade_lab.ml_optimization.objective import _wrap_model
        from trade_lab.strategies.ml_strategy import MLStrategy

        # Step 1 — Prune model
        pruned_model, report = self.prune_model(
            result.best_model, result.feature_names,
        )
        dead_set = set(report['dead_features'])

        # Step 2 — Filter: drop indicators whose output columns are all dead.
        #           Partially alive indicators are kept.
        filtered_indicators: list[BaseIndicator] = []
        for ind in result.best_feature_spec:
            alive_cols = [c for c in ind.output_columns if c not in dead_set]
            if alive_cols:
                filtered_indicators.append(ind)

        # Step 3 — Rebuild FeatureMatrix and refit scaler on training data
        feature_matrix = FeatureMatrix(filtered_indicators)
        X_train, y_train = feature_matrix.build(result.train_df, fit_scaler=True)

        # Step 4 — Slice surviving feature weights from first Dense layer
        original_names = result.feature_names
        surviving_names = feature_matrix.feature_names
        surviving_indices = [
            original_names.index(name)
            for name in surviving_names
            if name in original_names
        ]

        dense_layers_info: list[tuple[dict, list[np.ndarray]]] = []
        first_dense = True
        for layer in pruned_model.layers:
            if isinstance(layer, keras.layers.Dense):
                config = layer.get_config()
                ws = layer.get_weights()
                if first_dense and surviving_indices:
                    kernel = ws[0]
                    ws = [kernel[surviving_indices, :]] + ws[1:]
                    first_dense = False
                else:
                    first_dense = False
                dense_layers_info.append((config, ws))

        # Step 5 — Build fine-tune model and transfer weights
        inputs = [keras.Input(name=name, shape=(1,)) for name in surviving_names]
        x = inputs[0] if len(inputs) == 1 else keras.layers.Concatenate()(inputs)
        for config, weights in dense_layers_info:
            layer = keras.layers.Dense.from_config(config)
            x = layer(x)
            layer.set_weights(weights)

        fine_tuned_model = keras.Model(inputs=inputs, outputs=x)
        fine_tuned_model.compile(optimizer='adam', loss='mse')
        fine_tuned_model.fit(X_train, y_train, epochs=fine_tune_epochs, verbose=0)

        # Step 6 — Wrap and rebuild strategy
        wrapped_model = _wrap_model(fine_tuned_model, surviving_names)
        strategy = MLStrategy(
            model=wrapped_model,
            indicators=filtered_indicators,
            allow_long=True,
            allow_short=True,
        )

        updated_result = copy.copy(result)
        updated_result.best_model = wrapped_model
        updated_result.best_strategy = strategy
        updated_result.best_feature_spec = filtered_indicators
        updated_result.feature_names = surviving_names
        updated_result.scaler = feature_matrix.scaler

        return updated_result, report