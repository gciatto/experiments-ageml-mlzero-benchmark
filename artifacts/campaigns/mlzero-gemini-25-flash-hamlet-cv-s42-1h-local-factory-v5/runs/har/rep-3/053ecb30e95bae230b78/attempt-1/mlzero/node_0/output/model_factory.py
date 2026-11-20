import os
import random
import uuid
import numpy as np
import pandas as pd
from pathlib import Path
from autogluon.tabular import TabularPredictor

# --- Module-level constants and configurations ---
SEED = 42
TARGET_COLUMN = 'Class'
EVAL_METRIC = 'balanced_accuracy'
# The total shared deadline for ALL ten fits is 3215 seconds.
# We allocate 250 seconds per fold for fitting to ensure all fits finish
# within the deadline, allowing ample time for overhead (data loading,
# model initialization, saving, etc.) across the 10 folds.
TIME_LIMIT_PER_FOLD = 250

# --- Wrapper Class ---
class AutoGluonTabularWrapper:
    """
    A wrapper class for AutoGluon's TabularPredictor to conform to the
    fit/predict interface required by the benchmark harness.
    """
    def __init__(self, seed, time_limit, eval_metric, target_column):
        self.seed = seed
        self.time_limit = time_limit
        self.eval_metric = eval_metric
        self.target_column = target_column
        self.predictor = None
        # Create a unique path for AutoGluon artifacts for each model instance.
        # This ensures "fresh artifact directories per fold" as required by the contract.
        # AutoGluon will create this directory if it doesn't exist.
        self.save_path = Path(f"AG_models_{uuid.uuid4()}")

    def fit(self, X, y):
        """
        Fits the AutoGluon TabularPredictor model using the provided training data.
        All preprocessing and model fitting are performed within this method.

        Args:
            X (pd.DataFrame): Training features.
            y (pd.Series or np.ndarray): Training labels.
        """
        # Set seeds for reproducibility for general Python/Numpy operations.
        # AutoGluon's internal randomness is primarily controlled by its 'seed' parameter.
        random.seed(self.seed)
        np.random.seed(self.seed)

        # Combine X and y into a single DataFrame for AutoGluon.
        # Ensure y is a Series with the correct name and index.
        if isinstance(y, pd.Series):
            y_series = y.rename(self.target_column)
        else: # Assume y is a numpy array or list
            y_series = pd.Series(y, name=self.target_column, index=X.index)

        train_data = pd.concat([X, y_series], axis=1)

        # Initialize TabularPredictor.
        # verbosity=0 to suppress excessive output during benchmark runs.
        self.predictor = TabularPredictor(
            label=self.target_column,
            eval_metric=self.eval_metric,
            path=str(self.save_path), # AutoGluon expects a string path
            verbosity=0
        )

        # Fit the predictor.
        # Using presets='extreme' as requested by the guidance ("try to use presets='extreme'").
        # Note: 'extreme' preset often requires a GPU and is optimized for datasets <30K samples.
        # If the environment lacks a GPU or the dataset is large, 'best_quality' is a robust
        # alternative for high accuracy.
        self.predictor.fit(
            train_data,
            presets='extreme', # As per guidance
            time_limit=self.time_limit,
            seed=self.seed,
            # AutoGluon automatically handles internal validation split.
            # Do not manually split a validation set from the training data.
        )

    def predict(self, X):
        """
        Makes predictions using the fitted AutoGluon TabularPredictor model.

        Args:
            X (pd.DataFrame): Features to predict on.

        Returns:
            pd.Series: Predicted class labels.
        """
        if self.predictor is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit(X, y) first.")
        return self.predictor.predict(X)

# --- Factory Function ---
def build_model(X_train, y_train):
    """
    Factory function to build a new, unfitted AutoGluon Tabular model wrapper.

    This function adheres to the contract by returning a fresh, unfitted model
    instance with `fit(X,y)` and `predict(X)` methods.

    Args:
        X_train (pd.DataFrame): Training features. Provided for context (e.g., column names, problem type).
                                The model returned is unfitted.
        y_train (pd.Series): Training labels. Provided for context.

    Returns:
        AutoGluonTabularWrapper: An unfitted instance of AutoGluonTabularWrapper
                                 with `fit(X,y)` and `predict(X)` methods.
    """
    # X_train and y_train are provided for context, but the model is returned unfitted.
    # The target column name is already known from the problem description.
    return AutoGluonTabularWrapper(
        seed=SEED,
        time_limit=TIME_LIMIT_PER_FOLD,
        eval_metric=EVAL_METRIC,
        target_column=TARGET_COLUMN
    )