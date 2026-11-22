import pandas as pd
import numpy as np
import os
import tempfile
import shutil
import random # For setting global seed if needed, though AutoGluon manages its own.

from autogluon.tabular import TabularPredictor, TabularDataset

# --- Module-level Constants and Configuration ---
# Seed for reproducibility. AutoGluon's TabularPredictor manages its own internal
# randomness for models, but setting a global seed can influence overall behavior
# if not explicitly overridden by the library. The benchmark harness is expected
# to set the global seed for the execution environment.
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Evaluation metric for AutoGluon. 'balanced_accuracy' is directly supported.
EVAL_METRIC = 'balanced_accuracy'

# Time limit per fold for AutoGluon's fit method.
# The total shared deadline is 3232 seconds for 10 folds, averaging ~323 seconds per fold.
# Setting a conservative time limit of 250 seconds per fit to allow for overhead
# and ensure all folds complete within the deadline.
TIME_LIMIT_PER_FIT = 250

# AutoGluon presets. 'good' offers a balance of speed and accuracy, suitable
# for time-constrained benchmarks. 'best' or 'extreme' might exceed the time limit.
AUTOGLUON_PRESET = 'good'


class AutoGluonTabularWrapper:
    """
    A wrapper class for autogluon.tabular.TabularPredictor to conform to the
    fit/predict interface required by the benchmark harness.
    """
    def __init__(self):
        self.predictor = None
        self.label_column_name = 'target'  # Default name for the target column
        self.save_path = None

        # Create a temporary directory for AutoGluon artifacts.
        # This directory will be unique for each instance of the wrapper, i.e., per fold.
        self.save_path = tempfile.mkdtemp(prefix="autogluon_model_")
        # print(f"AutoGluon model artifacts for this fold will be saved to: {self.save_path}") # Debugging

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fits the AutoGluon TabularPredictor model.

        Args:
            X (pd.DataFrame): Training features.
            y (pd.Series): Training target labels.
        """
        # Ensure y has a name for the target column in the combined DataFrame.
        # AutoGluon requires the target column to be part of the training DataFrame.
        if y.name is None:
            y.name = self.label_column_name
        else:
            self.label_column_name = y.name

        # Combine X and y into a single DataFrame as required by TabularPredictor.
        train_data = pd.concat([X, y], axis=1)
        train_dataset = TabularDataset(train_data)

        # Initialize TabularPredictor.
        # verbosity=0 suppresses most AutoGluon output for cleaner logs.
        self.predictor = TabularPredictor(
            label=self.label_column_name,
            eval_metric=EVAL_METRIC,
            path=self.save_path,
            verbosity=0
        )

        # Fit the predictor.
        # The `presets` parameter controls the trade-off between training time and model quality.
        # `time_limit` ensures the fitting process respects the per-fold deadline.
        self.predictor.fit(
            train_dataset,
            time_limit=TIME_LIMIT_PER_FIT,
            presets=AUTOGLUON_PRESET,
            # AutoGluon's internal models are configured by `presets`.
            # For specific model randomness, `hyperparameters` or `ag_args_fit` would be used.
            # The global `RANDOM_SEED` set at the module level provides overall reproducibility.
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Makes predictions using the fitted AutoGluon TabularPredictor model.

        Args:
            X (pd.DataFrame): Features to predict on.

        Returns:
            np.ndarray: Predicted integer class labels.
        """
        if self.predictor is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit(X, y) first.")

        # AutoGluon's predict method returns a pandas Series, convert to a NumPy array
        # to match the required output format.
        predictions = self.predictor.predict(X).to_numpy()
        return predictions

    def __del__(self):
        """
        Clean up the temporary directory created for AutoGluon artifacts.
        This is a best-effort cleanup. The benchmark harness is also expected
        to clean up the overall output directory.
        """
        if self.save_path and os.path.exists(self.save_path):
            try:
                shutil.rmtree(self.save_path)
                # print(f"Cleaned up AutoGluon model artifacts from: {self.save_path}") # Debugging
            except OSError as e:
                # print(f"Error removing AutoGluon model directory {self.save_path}: {e}") # Debugging
                pass  # Ignore errors during cleanup


def build_model(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Factory function to build a new, unfitted model wrapper.
    This function is called once per cross-validation fold by the harness.

    Args:
        X_train (pd.DataFrame): Example training features. This is provided
                                for context on the expected input format, but
                                not used for fitting at this stage.
        y_train (pd.Series): Example training target. Provided for context,
                             not used for fitting.

    Returns:
        AutoGluonTabularWrapper: An unfitted model wrapper instance, ready
                                 to have its `fit` method called.
    """
    # The `X_train` and `y_train` arguments are for defining the contract
    # and providing type hints; the model returned is unfitted.
    return AutoGluonTabularWrapper()