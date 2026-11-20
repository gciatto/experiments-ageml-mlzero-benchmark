import pandas as pd
import autogluon.tabular as ag
import tempfile
import shutil
import os
from autogluon.common.utils.utils import set_seed

# Module-level constants for reproducibility and resource management.
# These are initialized once when the module is loaded.
RANDOM_SEED = 42
# Allocate a conservative time limit per fold.
# The total shared deadline is 1403 seconds for 10 folds.
# An average of 140.3 seconds per fold is available.
# Setting it to 120 seconds per fit leaves room for overhead and other folds
# to potentially take longer, ensuring all fits finish within the deadline.
TIME_LIMIT_PER_FOLD = 120 # seconds

class AutoGluonModelWrapper:
    """
    A wrapper class for AutoGluon's TabularPredictor to conform to the
    `fit(X, y)` and `predict(X)` interface required by the harness.
    """
    def __init__(self):
        self.predictor = None
        self._temp_path = None # Stores the path to AutoGluon's temporary artifacts

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fits the AutoGluon TabularPredictor model.

        All learned preprocessing and model training are performed within this method,
        using only the provided training data (X, y) for the current fold.

        Args:
            X (pd.DataFrame): Training features for the current fold.
            y (pd.Series): Training target labels for the current fold.
        """
        # Set global seed for AutoGluon for reproducibility within this fit call.
        # This ensures that internal AutoGluon operations are deterministic.
        set_seed(RANDOM_SEED)

        # Create a unique temporary directory for AutoGluon artifacts for this fold.
        # This fulfills the requirement for "fresh artifact directories per fold".
        # AutoGluon will store its models, logs, and intermediate data here.
        self._temp_path = tempfile.mkdtemp()

        # AutoGluon's TabularPredictor expects a single DataFrame for training.
        # Combine features (X) and target (y) into one DataFrame.
        train_df = X.copy()
        # The target column name is specified as 'Class' in the problem description.
        # `y.name` will correctly provide this if `y` is a pandas Series with a name.
        train_df[y.name] = y

        # Initialize TabularPredictor.
        # `label` is the name of the target column.
        # `path` specifies where AutoGluon stores its models and data.
        # `eval_metric` is set to 'balanced_accuracy' as specified in the README.
        self.predictor = ag.TabularPredictor(
            label=y.name,
            path=self._temp_path,
            eval_metric='balanced_accuracy',
            # verbosity=0 # Uncomment to reduce AutoGluon's output during fitting
        )

        # Fit the predictor.
        # `time_limit` ensures that the fitting process respects the overall deadline.
        # `random_seed` ensures reproducibility of the model training process.
        # `presets='best_quality'` aims for high performance within the given time limit.
        self.predictor.fit(
            train_data=train_df,
            time_limit=TIME_LIMIT_PER_FOLD,
            random_seed=RANDOM_SEED, # Pass seed to fit method for specific model training
            presets='best_quality',
        )

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Generates predictions for the given input features X.

        Args:
            X (pd.DataFrame): Features for which to generate predictions.

        Returns:
            pd.Series: Predicted target values.
        """
        if self.predictor is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit(X, y) first.")
        return self.predictor.predict(X)

    def __del__(self):
        """
        Attempts to clean up the temporary directory created for AutoGluon artifacts
        when the AutoGluonModelWrapper object is garbage collected.
        Note: The `__del__` method is not guaranteed to be called reliably or promptly.
        """
        if self._temp_path and os.path.exists(self._temp_path):
            try:
                shutil.rmtree(self._temp_path)
                # print(f"Cleaned up temporary directory: {self._temp_path}") # For debugging
            except OSError as e:
                # print(f"Error cleaning up temporary directory {self._temp_path}: {e}") # For debugging
                pass # Ignore errors during cleanup, as it's a best-effort cleanup.

def build_model(X_train: pd.DataFrame, y_train: pd.Series) -> AutoGluonModelWrapper:
    """
    Factory function as required by the harness.
    Returns a new, unfitted model wrapper instance.

    X_train and y_train are provided for context (e.g., to infer data types or
    target column name) but are NOT used to fit the model at this stage.
    The model is fitted later via its `.fit(X, y)` method, which will receive
    the actual training data for a specific fold.

    Args:
        X_train (pd.DataFrame): Example training features (not used for fitting here).
        y_train (pd.Series): Example training target (not used for fitting here).

    Returns:
        AutoGluonModelWrapper: A new, unfitted instance of the model wrapper.
    """
    return AutoGluonModelWrapper()