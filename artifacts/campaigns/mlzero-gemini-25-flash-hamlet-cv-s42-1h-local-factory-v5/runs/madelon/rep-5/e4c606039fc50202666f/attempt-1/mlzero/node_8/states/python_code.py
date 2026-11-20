import pandas as pd
import numpy as np
import os
import time
import shutil
from autogluon.tabular import TabularPredictor
import autogluon.core as ag

# Module-level constants for configuration.
# These are initialized once when the module is loaded.
RANDOM_SEED = 42
TARGET_COLUMN_NAME = "Class"

class AutoGluonModelWrapper:
    """
    A wrapper class for AutoGluon's TabularPredictor to conform to the
    fit/predict interface required by the benchmark harness.
    """
    def __init__(self, target_column_name: str, random_seed: int):
        """
        Initializes the wrapper. The AutoGluon predictor itself is initialized
        during the fit call to ensure it's fresh for each fold.

        Args:
            target_column_name (str): The name of the target column.
            random_seed (int): Seed for reproducibility.
        """
        self.predictor: TabularPredictor = None
        self.target_column_name = target_column_name
        self.random_seed = random_seed
        self.model_path: str = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Fits the AutoGluon TabularPredictor model.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training labels.
        """
        # Set seed for AutoGluon for reproducibility within this fit call.
        # This addresses previous errors regarding `random_seed` not being accepted
        # by the TabularPredictor constructor or fit method directly.
        ag.set_seed(self.random_seed)

        # Combine features and target into a single DataFrame as required by AutoGluon
        train_data = X_train.copy()
        train_data[self.target_column_name] = y_train

        # Create a unique path for AutoGluon artifacts for this specific fit call.
        # This ensures isolation between different folds/runs and satisfies the
        # "fresh artifact directories per fold" requirement.
        # Using a timestamp to guarantee uniqueness.
        timestamp = int(time.time() * 1000)
        self.model_path = f"autogluon_model_{timestamp}"
        
        # Defensive: remove directory if it somehow exists from a previous run in the same path.
        # AutoGluon will create it if it doesn't exist.
        if os.path.exists(self.model_path):
            shutil.rmtree(self.model_path)
        
        # Initialize TabularPredictor.
        # The `random_seed` argument is not supported in the constructor for this AG version.
        self.predictor = TabularPredictor(
            label=self.target_column_name,
            path=self.model_path,
            eval_metric='balanced_accuracy',  # Metric specified in README.md
            verbosity=0  # Suppress excessive logging
        )

        # Fit the model with a time limit and presets for balanced performance.
        # The total remaining time is 1505 seconds. For 10 folds, a time_limit of 120 seconds
        # per fold results in 1200 seconds for fitting, leaving 305 seconds for overhead.
        # This should ensure all fits complete within the shared deadline.
        self.predictor.fit(
            train_data=train_data,
            presets='good_quality',  # A reasonable preset for general tabular classification
            time_limit=120,  # Adjusted time limit for safety across 10 folds
        )

    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        """
        Makes predictions using the fitted AutoGluon model.

        Args:
            X_test (pd.DataFrame): Test features.

        Returns:
            pd.Series: Predicted class labels.
        """
        if self.predictor is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        
        # AutoGluon's predict method directly takes the feature DataFrame.
        predictions = self.predictor.predict(X_test)
        return predictions

def build_model(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Factory function to generate a new, unfitted AutoGluon model wrapper.

    Args:
        X_train (pd.DataFrame): Training features (used for signature, not for model initialization).
        y_train (pd.Series): Training labels (used for signature, not for model initialization).

    Returns:
        AutoGluonModelWrapper: An unfitted model wrapper instance.
    """
    # Returns a fresh, unfitted instance of our wrapper.
    # The target column name and seed are passed from module-level constants.
    return AutoGluonModelWrapper(target_column_name=TARGET_COLUMN_NAME, random_seed=RANDOM_SEED)