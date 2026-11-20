import pandas as pd
import numpy as np
import os
import tempfile
import shutil
import logging

# Configure logging to suppress excessive AutoGluon output if desired.
# For this benchmark, we'll keep it at a reasonable level.
logging.getLogger('autogluon').setLevel(logging.WARNING)

# Define constants for the task as allowed by module scope initialization.
TARGET_COLUMN = 'median_house_value'
RANDOM_SEED = 42
# Total time for 10 fits is 782 seconds. Aim for ~70 seconds per fold
# to allow for overhead (data loading, Python startup, etc.).
TIME_LIMIT_PER_FOLD = 70 # seconds

# AutoGluon is allowed, so we import its TabularPredictor.
from autogluon.tabular import TabularPredictor

class AutoGluonTabularModelWrapper:
    """
    A wrapper class for AutoGluon's TabularPredictor.
    It conforms to the `fit(X, y)` and `predict(X)` interface required by the benchmark.
    This class handles the lifecycle of AutoGluon models, including temporary
    directory management for artifacts.
    """
    def __init__(self):
        self.predictor = None
        self.temp_path = None # Path to store AutoGluon artifacts for this model instance

    def fit(self, X, y):
        """
        Fits the AutoGluon TabularPredictor using the provided training data.
        All preprocessing and model training are encapsulated within this method,
        using only the supplied X and y for the current fold.
        """
        # Combine features (X) and target (y) into a single DataFrame,
        # as AutoGluon's TabularPredictor expects the target column to be
        # present in the training data.
        train_data = X.copy()
        train_data[TARGET_COLUMN] = y

        # Create a unique temporary directory for AutoGluon artifacts for this fold.
        # This ensures "fresh artifact directories per fold" and prevents conflicts
        # between different model instances or folds.
        self.temp_path = tempfile.mkdtemp()

        # Initialize AutoGluon's TabularPredictor.
        # - `label`: Specifies the target column.
        # - `path`: Points to the temporary directory for storing models and data.
        # - `problem_type`: Explicitly set to 'regression' for this task.
        # - `eval_metric`: Set to 'rmse' as specified by the problem description.
        # Note: `random_state` is NOT passed to the constructor, as per previous error.
        self.predictor = TabularPredictor(
            label=TARGET_COLUMN,
            path=self.temp_path,
            problem_type='regression',
            eval_metric='rmse'
        )

        # Fit the predictor.
        # - `time_limit`: Crucial for bounding the training time for each fold,
        #   ensuring all ten fits complete within the shared deadline.
        # - `presets='best_quality'`: Aims for the best possible performance.
        # - `seed=RANDOM_SEED`: Ensures reproducibility for stochastic processes
        #   within AutoGluon, as required by the benchmark.
        self.predictor.fit(
            train_data,
            time_limit=TIME_LIMIT_PER_FOLD,
            presets='best_quality',
            seed=RANDOM_SEED
        )

    def predict(self, X):
        """
        Generates numerical predictions using the fitted AutoGluon TabularPredictor.
        """
        if self.predictor is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit(X, y) first.")

        # AutoGluon's predict method expects a DataFrame for features.
        # It returns a pandas Series, which we convert to a numpy array
        # to match typical scikit-learn prediction output formats.
        predictions = self.predictor.predict(X)
        return predictions.values

    def __del__(self):
        """
        Cleans up the temporary directory created for AutoGluon artifacts.
        This method is called when the object is garbage collected.
        """
        if self.temp_path and os.path.exists(self.temp_path):
            try:
                shutil.rmtree(self.temp_path)
            except OSError as e:
                # Log or handle errors during cleanup if necessary.
                # For a benchmark, simply passing might be acceptable if cleanup
                # is not strictly critical (e.g., OS cleans /tmp periodically).
                logging.warning(f"Error removing temporary directory {self.temp_path}: {e}")


def build_model(X_train, y_train):
    """
    Factory function required by the benchmark harness.
    It returns a new, unfitted model wrapper instance.

    Args:
        X_train (pd.DataFrame): Training features (not used for fitting here).
        y_train (pd.Series): Training target (not used for fitting here).

    Returns:
        AutoGluonTabularModelWrapper: An unfitted model instance ready for training.
    """
    # X_train and y_train are provided as per the contract, but they are not
    # used within this function, as the contract requires returning an *unfitted* model.
    # The actual fitting and use of X_train/y_train occurs within the
    # AutoGluonTabularModelWrapper's .fit() method.
    return AutoGluonTabularModelWrapper()