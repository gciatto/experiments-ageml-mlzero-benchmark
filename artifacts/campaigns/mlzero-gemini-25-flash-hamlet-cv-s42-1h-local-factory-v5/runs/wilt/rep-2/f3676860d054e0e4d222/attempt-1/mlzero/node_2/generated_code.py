import pandas as pd
import numpy as np
import os
import tempfile
import shutil
from autogluon.tabular import TabularPredictor

# Seed for reproducibility across AutoGluon runs
SEED = 42

class AutoGluonModelWrapper:
    """
    A wrapper class for AutoGluon TabularPredictor to conform to the
    fit/predict interface required by the harness.
    """
    def __init__(self, time_limit: int = None):
        """
        Initializes the wrapper. The AutoGluon predictor itself is
        initialized and fitted only when the .fit() method is called.

        Args:
            time_limit (int, optional): Maximum time in seconds for AutoGluon's
                                        fit process. Defaults to None (no limit).
        """
        self.predictor: TabularPredictor = None
        self.time_limit = time_limit
        self._temp_path: str = None # Path for AutoGluon artifacts

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fits the AutoGluon TabularPredictor using the provided training data.
        All preprocessing and model training happens within this method,
        using only the training rows for the current fold.

        Args:
            X (pd.DataFrame): Training features.
            y (pd.Series): Training target labels.
        """
        # Create a temporary directory for AutoGluon artifacts for this fold.
        # This ensures fresh artifact directories per fold as required by the contract.
        # AutoGluon will store models and intermediate files here.
        self._temp_path = tempfile.mkdtemp()

        # Ensure y is a pandas Series with the correct name 'class' for AutoGluon.
        # If y is a numpy array, convert it to a Series.
        if not isinstance(y, pd.Series):
            y_series = pd.Series(y, name='class', index=X.index)
        else:
            y_series = y.copy() # Avoid modifying original Series in place
            y_series.name = 'class' # Ensure target column name is 'class'

        # Combine features and target into a single DataFrame for AutoGluon.
        train_data = pd.concat([X, y_series], axis=1)

        # Initialize TabularPredictor.
        # The 'label' argument specifies the name of the target column.
        # The 'path' argument specifies where AutoGluon should store its artifacts.
        self.predictor = TabularPredictor(
            label='class',
            path=self._temp_path,
            # Note: random_seed is passed to .fit() method, not constructor,
            # as per previous error analysis.
        )

        # Fit the predictor.
        # 'presets='fast_inference'' is chosen to balance performance and speed,
        # which is crucial given the shared deadline for all ten folds.
        # 'time_limit' ensures that each fit operation respects the overall time budget.
        # 'random_seed' ensures reproducibility of the training process.
        self.predictor.fit(
            train_data,
            presets='fast_inference',
            time_limit=self.time_limit,
            random_seed=SEED,
            # Other AutoGluon options (e.g., specific hyperparameters) could be
            # added here if further tuning is desired.
        )

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Makes predictions using the fitted AutoGluon TabularPredictor.

        Args:
            X (pd.DataFrame): Features for which to make predictions.

        Returns:
            pd.Series: Predicted class labels.
        """
        if self.predictor is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        predictions = self.predictor.predict(X)
        return predictions

    def __del__(self):
        """
        Cleans up the temporary directory created by AutoGluon when the
        wrapper object is garbage collected. This helps prevent accumulation
        of temporary files across multiple folds/runs.
        """
        if self._temp_path and os.path.exists(self._temp_path):
            try:
                shutil.rmtree(self._temp_path)
            except OSError as e:
                # In a production environment, this might be logged.
                # For a benchmark, simply ignoring or a passive print is often sufficient.
                # print(f"Warning: Could not remove AutoGluon temp directory {self._temp_path}: {e}")
                pass


def build_model(X_train: pd.DataFrame, y_train: pd.Series) -> AutoGluonModelWrapper:
    """
    Factory function to build a new, unfitted model wrapper.
    This function is called by the harness for each fold.

    Args:
        X_train (pd.DataFrame): Training features (used here for context, not fitting).
        y_train (pd.Series): Training target labels (used here for context, not fitting).

    Returns:
        AutoGluonModelWrapper: An unfitted model wrapper instance ready to be fitted.
    """
    # The total time for all ten fits is 2229 seconds.
    # To ensure all fits finish within the deadline, we allocate a time limit
    # for each individual AutoGluon fit operation.
    # 200 seconds per fold * 10 folds = 2000 seconds, leaving a buffer of 229 seconds
    # for data loading, Python startup, and other overheads.
    per_fold_time_limit = 200
    return AutoGluonModelWrapper(time_limit=per_fold_time_limit)