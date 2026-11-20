import os
import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor

# Module-level constants and configurations
# The target column name as specified in the README.md
TARGET_COLUMN = 'class'

# Seed for reproducibility across AutoGluon's internal processes
RANDOM_SEED = 42

# Total time for all ten fits and overhead is 3475 seconds.
# We allocate 300 seconds per fold for fitting, leaving a buffer for overhead.
# This ensures ALL ten fits finish inside the remaining shared deadline.
TIME_LIMIT_PER_FIT_SECONDS = 300

class AutoGluonTabularWrapper:
    """
    A wrapper for AutoGluon TabularPredictor to conform to the MLZero factory contract.
    This wrapper provides `fit` and `predict` methods.
    """
    def __init__(self, label_column: str, random_seed: int, time_limit_per_fit: int):
        """
        Initializes the wrapper. The AutoGluon TabularPredictor itself is not
        initialized or fitted until the `fit` method is called.

        Args:
            label_column (str): The name of the target column.
            random_seed (int): Seed for reproducibility.
            time_limit_per_fit (int): Maximum time in seconds allowed for each fit operation.
        """
        self.label_column = label_column
        self.random_seed = random_seed
        self.time_limit_per_fit = time_limit_per_fit
        self.predictor: TabularPredictor = None
        
        # AutoGluon will create a subdirectory named 'AutogluonModels' within the
        # current working directory. The harness ensures the current working directory
        # is unique for each fold, thus providing "fresh artifact directories per fold".
        self.model_save_path: str = './AutogluonModels'

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fits the AutoGluon TabularPredictor model using the provided training data.
        All learned preprocessing is fitted within this method.

        Args:
            X (pd.DataFrame): Training features for the current fold.
            y (pd.Series): Training labels for the current fold.
        """
        # Combine X and y into a single DataFrame as required by AutoGluon
        train_data = X.copy()
        train_data[self.label_column] = y

        # Initialize TabularPredictor
        # - `label`: Specifies the target column.
        # - `eval_metric`: 'balanced_accuracy' is specified in the README.md.
        # - `path`: Ensures artifacts are saved in a unique location per fold.
        # - `verbosity=0`: Suppresses verbose output during fitting for cleaner logs.
        self.predictor = TabularPredictor(
            label=self.label_column,
            eval_metric='balanced_accuracy',
            path=self.model_save_path,
            verbosity=0,
        )

        # Fit the predictor with specified presets and time limit
        # - `presets='best_quality'`: Recommended for high accuracy, balances performance and time.
        # - `time_limit`: Crucial for adhering to the shared deadline.
        # - `random_state`: Ensures reproducibility of the training process.
        self.predictor.fit(
            train_data,
            presets='best_quality',
            time_limit=self.time_limit_per_fit,
            random_state=self.random_seed,
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Makes predictions using the fitted AutoGluon TabularPredictor.

        Args:
            X (pd.DataFrame): Features for prediction.

        Returns:
            np.ndarray: A 1-dimensional array of predicted class labels.
        """
        if self.predictor is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit(X, y) first.")
        
        # AutoGluon's predict method returns class labels directly for classification tasks.
        predictions = self.predictor.predict(X)
        
        # Ensure the output is a 1-dimensional numpy array as required by the contract.
        return np.asarray(predictions)


def build_model(X_train: pd.DataFrame, y_train: pd.Series) -> AutoGluonTabularWrapper:
    """
    Factory function to build a new, unfitted model wrapper for AutoGluon TabularPredictor.

    Args:
        X_train (pd.DataFrame): Training features. This argument is provided by the harness
                                for initialization but is not directly used by this specific
                                wrapper's __init__ as AutoGluon infers details during .fit().
        y_train (pd.Series): Training labels. Similar to X_train, used for initialization
                             context but not directly consumed by the wrapper's __init__.

    Returns:
        AutoGluonTabularWrapper: An unfitted model wrapper with `fit` and `predict` methods.
    """
    # Return a fresh, unfitted instance of our wrapper.
    # The actual AutoGluon TabularPredictor will be initialized and fitted
    # inside the wrapper's `fit` method when it is called by the harness.
    return AutoGluonTabularWrapper(
        label_column=TARGET_COLUMN,
        random_seed=RANDOM_SEED,
        time_limit_per_fit=TIME_LIMIT_PER_FIT_SECONDS
    )