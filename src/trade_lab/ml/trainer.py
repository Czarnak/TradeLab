from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from trade_lab.indicators.base import BaseIndicator
from trade_lab.ml.models import KerasModelWrapper, wrap_with_scaling
from trade_lab.ml.preprocessing import FeatureScaler, prepare_features
from trade_lab.ml.targets import BaseTarget
from trade_lab.ml.validation import WalkForwardResult, WalkForwardSplit


class MLTrainer:
    """Orchestrates ML model training using the signal/indicator pipeline.

    Workflow:
        1. Run indicators (which run their upstream signals) to build features.
        2. Generate target labels from raw OHLCV via a ``BaseTarget``.
        3. Align features + target, drop NaN rows.
        4. Optionally scale features.
        5. Train a Keras model.
        6. Wrap in ``KerasModelWrapper`` for use with ``MLStrategy``.

    Parameters
    ----------
    indicators : list[BaseIndicator]
        Indicators that provide features.  Each indicator's ``compute()``
        is called sequentially — upstream signals are resolved automatically.
    target : BaseTarget
        Generates training labels from the DataFrame.
    model_builder : Callable[[int], keras.Model]
        Factory that creates a fresh, compiled Keras model given ``input_dim``.
        Use ``dense_model()`` from ``trade_lab.ml.models``.
    scaler : FeatureScaler | None
        If provided, features are scaled.  Fitted on training data only
        to prevent leakage.
    """

    def __init__(
        self,
        indicators: list[BaseIndicator],
        target: BaseTarget,
        model_builder: Callable,
        scaler: FeatureScaler | None = None,
    ):
        self.indicators = indicators
        self.target = target
        self.model_builder = model_builder
        self.scaler = scaler

    # ------------------------------------------------------------------
    # Dataset construction
    # ------------------------------------------------------------------

    def build_dataset(
        self,
        df: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, list[str], pd.DatetimeIndex]:
        """Run the full feature pipeline and return clean arrays.

        Returns
        -------
        X : np.ndarray
        y : np.ndarray
        feature_columns : list[str]
        index : pd.DatetimeIndex
        """
        df = self._compute_features(df)
        target = self.target.generate(df)
        feature_columns = self._collect_feature_columns(df)
        X, y, index = prepare_features(df, feature_columns, target)
        return X, y, feature_columns, index

    def _compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run all indicators (and their upstream signals)."""
        for indicator in self.indicators:
            df = indicator.compute(df)
        return df

    @staticmethod
    def _collect_feature_columns(df: pd.DataFrame) -> list[str]:
        """Discover feature columns using the naming convention."""
        return [
            col for col in df.columns if col.startswith(("signal__", "indicator__"))
        ]

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        df: pd.DataFrame,
        epochs: int = 100,
        batch_size: int = 32,
        validation_split: float = 0.2,
        verbose: int = 1,
    ) -> KerasModelWrapper:
        """Train a model on the full dataset.

        Validation is split **chronologically** (last N% of data),
        NOT randomly like Keras's default ``validation_split``.

        Returns
        -------
        KerasModelWrapper
            Trained model ready for ``MLStrategy``.
        """
        X, y, feature_columns, _ = self.build_dataset(df)

        if self.scaler:
            X = self.scaler.fit_transform(X, feature_columns)

        # Chronological split
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        model = self.model_builder(input_dim=X.shape[1])
        model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
        )

        # Fold the fitted scaler into the model so inference (and export) feed
        # raw features — see wrap_with_scaling. Without this the model is trained
        # on scaled inputs but served raw, silently degrading every prediction.
        return wrap_with_scaling(model, feature_columns, self.scaler)

    # ------------------------------------------------------------------
    # Walk-forward validation
    # ------------------------------------------------------------------

    def walk_forward(
        self,
        df: pd.DataFrame,
        n_splits: int = 5,
        expanding: bool = True,
        initial_train_ratio: float = 0.6,
        epochs: int = 100,
        batch_size: int = 32,
        verbose: int = 0,
    ) -> list[WalkForwardResult]:
        """Walk-forward cross-validation.

        For each fold:
            1. Create a fresh model via ``model_builder``.
            2. Fit on training window (expanding or sliding).
            3. Predict on test window.
            4. Record results.

        Returns
        -------
        list[WalkForwardResult]
            One result per fold, containing predictions, targets,
            and the trained model.
        """
        X, y, feature_columns, index = self.build_dataset(df)

        splitter = WalkForwardSplit(
            n_splits=n_splits,
            initial_train_ratio=initial_train_ratio,
            expanding=expanding,
        )
        folds = splitter.split(len(X))

        results: list[WalkForwardResult] = []

        for i, fold in enumerate(folds):
            X_train = X[fold.train_idx]
            y_train = y[fold.train_idx]
            X_test = X[fold.test_idx]
            y_test = y[fold.test_idx]

            # Fit scaler on training data only
            scaler_copy = None
            if self.scaler:
                scaler_copy = FeatureScaler(method=self.scaler.method)
                X_train = scaler_copy.fit_transform(X_train, feature_columns)
                X_test = scaler_copy.transform(X_test)

            model = self.model_builder(input_dim=X.shape[1])
            history = model.fit(
                X_train,
                y_train,
                epochs=epochs,
                batch_size=batch_size,
                verbose=verbose,
            )

            preds = model.predict(X_test, verbose=0).flatten()
            train_loss = history.history["loss"][-1]

            results.append(
                WalkForwardResult(
                    fold=i,
                    train_loss=train_loss,
                    test_predictions=pd.Series(
                        preds, index=index[fold.test_idx], name="prediction"
                    ),
                    test_targets=pd.Series(
                        y_test, index=index[fold.test_idx], name="target"
                    ),
                    # preds above were computed on the scaled X_test with the
                    # unfolded model; the stored model folds the fold's scaler so
                    # it is raw-feature-native for deployment.
                    model=wrap_with_scaling(model, feature_columns, scaler_copy),
                )
            )

        return results
