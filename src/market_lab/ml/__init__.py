"""ML model building, training, and optimization with Keras.

Public API:
- ``input_definitions``: Feature builders (close, open, return, range, volume lags)
- ``dataset_builder``: Compose features → X/y with train/val split
- ``model_builder``: Keras Dense model construction
- ``trainer``: Training loop, equity curves, report saving
- ``optimizer``: Optuna hyperparameter search
"""
