"""Post-training weight pruning for Keras models.

``ModelPruner`` supports two pruning strategies — global threshold and
per-layer percentile — and includes dead feature detection on the first
Dense layer. The ``prune_result`` convenience method operates directly on
an ``MLOptimizationResult``, handling feature spec filtering, scaler refit,
and fine-tuning.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from trade_lab.ml_optimization.result import MLOptimizationResult


class ModelPruner:
    """Prune small weights from a trained Keras model.

    Exactly one of ``threshold`` or ``percentile`` must be provided.

    Parameters
    ----------
    threshold : float | None
        Global absolute threshold. Weights with ``abs(w) < threshold`` are
        zeroed across all Dense layers.
    percentile : float | None
        Per-layer percentile. In each Dense layer, the bottom
        ``percentile``% of weights (by absolute magnitude) are zeroed.
        E.g. ``percentile=20`` zeros the smallest 20%.
    """

    def __init__(
        self,
        threshold: float | None = None,
        percentile: float | None = None,
    ) -> None:
        if (threshold is None) == (percentile is None):
            raise ValueError(
                "Exactly one of 'threshold' or 'percentile' must be provided."
            )
        self.threshold = threshold
        self.percentile = percentile

    def prune_model(
        self,
        model: Any,
        feature_names: list[str] | None = None,
    ) -> tuple[Any, dict]:
        """Prune weights from a Keras model.

        Creates a clone of the model and zeroes out small weights in all
        Dense layers (biases are not pruned). Optionally detects dead
        features in the first Dense layer.

        Parameters
        ----------
        model : keras.Model
            Trained model to prune. Not modified — a clone is returned.
        feature_names : list[str] | None
            If provided, enables dead feature detection on the first Dense
            layer. A feature is "dead" if all its outgoing weights are zero.

        Returns
        -------
        tuple[keras.Model, dict]
            ``(pruned_model, report)`` where ``report`` contains pruning
            statistics and dead feature list.
        """
        import keras

        # Clone the model to avoid modifying the original
        pruned_model = keras.models.clone_model(model)
        pruned_model.set_weights(model.get_weights())

        total_weights = 0
        zeroed_weights = 0
        layer_reports: dict[str, dict] = {}
        first_dense_found = False

        for layer in pruned_model.layers:
            if not isinstance(layer, keras.layers.Dense):
                continue

            weights = layer.get_weights()
            if not weights:
                continue

            # weights[0] = kernel, weights[1] = bias (if present)
            kernel = weights[0]
            layer_total = kernel.size

            if self.threshold is not None:
                mask = np.abs(kernel) < self.threshold
            else:
                # Per-layer percentile
                abs_vals = np.abs(kernel).ravel()
                cutoff = np.percentile(abs_vals, self.percentile)
                mask = np.abs(kernel) < cutoff

            kernel[mask] = 0.0
            layer_zeroed = int(np.sum(mask))

            # Set weights back (kernel modified, bias untouched)
            weights[0] = kernel
            layer.set_weights(weights)

            total_weights += layer_total
            zeroed_weights += layer_zeroed
            layer_reports[layer.name] = {
                'total': layer_total,
                'zeroed': layer_zeroed,
                'fraction': layer_zeroed / layer_total if layer_total > 0 else 0.0,
            }

            # Dead feature detection on the first Dense layer
            if not first_dense_found:
                first_dense_found = True
                first_kernel = kernel

        # Build dead features list
        dead_features: list[str] = []
        if feature_names is not None and first_dense_found:
            for i, name in enumerate(feature_names):
                if i < first_kernel.shape[0] and np.all(first_kernel[i, :] == 0):
                    dead_features.append(name)

        report = {
            'total_weights': total_weights,
            'zeroed_weights': zeroed_weights,
            'zero_fraction': (
                zeroed_weights / total_weights if total_weights > 0 else 0.0
            ),
            'layers': layer_reports,
            'dead_features': dead_features,
        }

        return pruned_model, report

    def prune_result(
        self,
        result: MLOptimizationResult,
        fine_tune_epochs: int = 10,
    ) -> tuple[MLOptimizationResult, dict]:
        """Prune an ``MLOptimizationResult`` end-to-end.

        Prunes the best model, filters dead features and their indicators,
        refits the scaler on training data, fine-tunes surviving weights,
        and rebuilds the strategy.

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

        from trade_lab.backtesting.engine import BacktestEngine
        from trade_lab.ml_optimization.feature_builder import (
            FeatureMatrix,
            LaggedIndicator,
        )
        from trade_lab.ml_optimization.objective import _wrap_model
        from trade_lab.strategies.ml_strategy import MLStrategy

        # Step 1 — Prune the model
        pruned_model, report = self.prune_model(
            result.best_model, result.feature_names,
        )

        # Step 2 — Identify dead features
        dead_set = set(report['dead_features'])

        # Step 3 — Filter feature spec: remove indicators whose columns
        # are all dead. Keep partially alive indicators.
        filtered_spec: list[LaggedIndicator] = []
        for li in result.best_feature_spec:
            alive_cols = [c for c in li.output_columns if c not in dead_set]
            if alive_cols:
                filtered_spec.append(li)

        # Step 4 — Rebuild FeatureMatrix and refit scaler on training data
        feature_matrix = FeatureMatrix(filtered_spec)
        X_train, y_train = feature_matrix.build(
            result.train_df, fit_scaler=True,
        )

        # Step 5 — Build a fresh model for the filtered features and fine-tune
        n_features = X_train.shape[1]
        # We need to create a new model matching the filtered feature count.
        # Transfer weights from the pruned model for surviving features.
        import keras

        # Build a simple model matching the architecture for fine-tuning
        # Extract surviving feature indices from the original feature set
        original_names = result.feature_names
        surviving_names = feature_matrix.feature_names
        surviving_indices = [
            original_names.index(name)
            for name in surviving_names
            if name in original_names
        ]

        # Get the pruned model's first Dense layer kernel and slice it
        dense_layers_info: list[tuple[dict, list[np.ndarray]]] = []
        first_dense = True
        for layer in pruned_model.layers:
            if isinstance(layer, keras.layers.Dense):
                config = layer.get_config()
                ws = layer.get_weights()
                if first_dense and surviving_indices:
                    # Slice the kernel to keep only surviving feature rows
                    kernel = ws[0]
                    sliced_kernel = kernel[surviving_indices, :]
                    ws = [sliced_kernel] + ws[1:]
                    first_dense = False
                else:
                    first_dense = False
                dense_layers_info.append((config, ws))

        # Build new functional model with named inputs
        inputs = [
            keras.Input(name=name, shape=(1,))
            for name in surviving_names
        ]
        if len(inputs) == 1:
            x = inputs[0]
        else:
            x = keras.layers.Concatenate()(inputs)

        for config, weights in dense_layers_info:
            layer = keras.layers.Dense.from_config(config)
            x = layer(x)
            layer.set_weights(weights)

        fine_tuned_model = keras.Model(inputs=inputs, outputs=x)
        fine_tuned_model.compile(
            optimizer='adam',
            loss='mse',
        )

        # Fine-tune
        # Build validation features for fine-tuning if possible
        fine_tuned_model.fit(
            X_train, y_train,
            epochs=fine_tune_epochs,
            verbose=0,
        )

        # Step 6 — Wrap model and rebuild strategy
        wrapped_model = _wrap_model(fine_tuned_model, surviving_names)
        strategy = MLStrategy(
            model=wrapped_model,
            indicators=[li.indicator for li in filtered_spec],
            allow_long=True,
            allow_short=True,
        )

        # Build updated result
        updated_result = copy.copy(result)
        updated_result.best_model = wrapped_model
        updated_result.best_strategy = strategy
        updated_result.best_feature_spec = filtered_spec
        updated_result.feature_names = surviving_names
        updated_result.scaler = feature_matrix.scaler

        return updated_result, report
