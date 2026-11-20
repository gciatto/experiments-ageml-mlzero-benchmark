import pandas as pd
import numpy as np
import autogluon.core as ag
from autogluon.tabular import TabularPredictor
import os
import uuid

# Set global seed for reproducibility across the entire process as requested.
# This influences AutoGluon's internal random operations.
ag.utils.set_random_seed(42)

class AutoGluonModelWrapper:
    """
    A wrapper class for AutoGluon's TabularPredictor to conform to the
    fit/predict model contract.
    """
    def __init__(self):
        self.predictor = None
        # AutoGluon will manage its own temporary artifact path during fit
        # if `path` is not explicitly provided to the TabularPredictor constructor.

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fits the AutoGluon TabularPredictor on the provided training data.
        All preprocessing and model training are handled internally by AutoGluon.

        Args:
            X (pd.DataFrame): Training features.
            y (pd.Series): Training labels.
        """
        # Combine X and y into a single DataFrame for AutoGluon.
        # The target column name is inferred from y.name.
        train_data = X.copy()
        train_data[y.name] = y

        # Initialize TabularPredictor.
        # By setting `path=None`, AutoGluon will create a new temporary directory
        # for each call to `fit`, satisfying the "fresh artifact directories per fold"
        # requirement.
        self.predictor = TabularPredictor(
            label=y.name,
            problem_type='classification',
            eval_metric='balanced_accuracy',
            path=None  # AutoGluon will create a temporary directory for artifacts
        )

        # Set a time limit for fitting to ensure all ten folds complete within the
        # shared deadline. 120 seconds per fold is a reasonable starting point
        # given a total of 1502 seconds for 10 folds plus overhead.
        time_limit_seconds = 120

        self.predictor.fit(
            train_data,
            time_limit=time_limit_seconds,
            presets='best_quality',  # Use 'best_quality' for potentially better performance
            # AutoGluon handles internal reproducibility through its own mechanisms,
            # influenced by the global seed set earlier.
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generates predictions for the given input features.

        Args:
            X (pd.DataFrame): Features for which to generate predictions.

        Returns:
            np.ndarray: A 1-dimensional NumPy array of predicted integer class labels.
        """
        if self.predictor is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit(X, y) first.")
        
        # AutoGluon's predict method returns a pandas Series.
        # Convert it to a NumPy array of integers as required.
        predictions = self.predictor.predict(X)
        return predictions.to_numpy().astype(int)

def build_model(X_train: pd.DataFrame, y_train: pd.Series) -> AutoGluonModelWrapper:
    """
    Factory function to build a new, unfitted AutoGluon model wrapper.

    This function adheres to the contract by returning a fresh, unfitted model
    instance that exposes `fit(X, y)` and `predict(X)` methods.
    The `X_train` and `y_train` arguments are provided by the harness for
    contract compliance but are not used to fit the model at this stage.

    Args:
        X_train (pd.DataFrame): Training features (not used for building, but for contract).
        y_train (pd.Series): Training labels (not used for building, but for contract).

    Returns:
        AutoGluonModelWrapper: An unfitted model wrapper ready for training.
    """
    return AutoGluonModelWrapper()