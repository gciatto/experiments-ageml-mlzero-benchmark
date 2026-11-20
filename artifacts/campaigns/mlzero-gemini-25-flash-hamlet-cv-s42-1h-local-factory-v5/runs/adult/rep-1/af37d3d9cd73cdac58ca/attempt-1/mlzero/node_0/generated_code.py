import pandas as pd
import numpy as np
import tempfile
import shutil
import os
from autogluon.tabular import TabularPredictor, TabularDataset

# --- Module-level constants ---
# Target column name as specified in the problem description.
TARGET_COLUMN = 'class'

# Problem type for AutoGluon. 'binary' for two classes, 'multiclass' for more.
# The 'adult' dataset is typically binary classification. AutoGluon can infer this,
# but being explicit can sometimes help.
PROBLEM_TYPE = 'binary'

# Random state for reproducibility of stochastic elements in AutoGluon.
RANDOM_STATE = 42

# Evaluation metric name. AutoGluon supports 'balanced_accuracy' as a string.
EVAL_METRIC_NAME = 'balanced_accuracy'

# Time limit for each fold's fit operation in seconds.
# The total remaining time is 3469 seconds for 10 folds, averaging ~346 seconds per fold.
# Setting a conservative limit to ensure all fits complete within the shared deadline.
FIT_TIME_LIMIT_SECONDS = 300 # 5 minutes per fold

class AutoGluonModel:
    """
    A wrapper class for AutoGluon's TabularPredictor to conform to the
    scikit-learn estimator interface (fit/predict).
    """
    def __init__(self):
        self.predictor = None
        self.temp_path = None # Stores the path to AutoGluon's temporary directory for cleanup

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fits the AutoGluon TabularPredictor model.

        All preprocessing and model fitting are encapsulated here, using only
        the provided training data (X, y) for the current fold.

        Args:
            X (pd.DataFrame): Training features for the current fold.
            y (pd.Series): Training target labels for the current fold.
        """
        # Combine X and y into a single DataFrame for AutoGluon.
        # Ensure y is a Series with the correct name, handling cases where y might be an array.
        if not isinstance(y, pd.Series):
            y_series = pd.Series(y, name=TARGET_COLUMN, index=X.index)
        else:
            y_series = y.copy() # Work on a copy to avoid modifying original Series name
            y_series.name = TARGET_COLUMN # Ensure the target column name is correct

        train_data = X.copy()
        train_data[TARGET_COLUMN] = y_series

        # Create a unique temporary directory for AutoGluon artifacts for this fold.
        # This fulfills the requirement for "fresh artifact directories per fold".
        self.temp_path = tempfile.mkdtemp(prefix="autogluon_model_")
        
        try:
            # Initialize TabularPredictor.
            # AutoGluon will infer feature types and handle preprocessing automatically.
            self.predictor = TabularPredictor(
                label=TARGET_COLUMN,
                problem_type=PROBLEM_TYPE,
                eval_metric=EVAL_METRIC_NAME,
                path=self.temp_path,
            )

            # Fit the predictor.
            # 'presets="best_quality"' is chosen for high accuracy, balancing with time_limit.
            # It enables advanced techniques like stacking and bagging.
            self.predictor.fit(
                train_data=TabularDataset(train_data),
                time_limit=FIT_TIME_LIMIT_SECONDS,
                presets='best_quality',
                random_state=RANDOM_STATE, # Ensures reproducibility for stochastic models
            )
        except Exception as e:
            # If an error occurs during fitting, attempt to clean up the temporary directory.
            if self.temp_path and os.path.exists(self.temp_path):
                shutil.rmtree(self.temp_path)
            raise e

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Makes predictions using the fitted AutoGluon model.

        Args:
            X (pd.DataFrame): Features to make predictions on.

        Returns:
            np.ndarray: A 1D array of predicted class labels, corresponding to the input rows.
        """
        if self.predictor is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit(X, y) first.")

        # AutoGluon's predict method returns a pandas Series; convert to a numpy array
        # to match the required output format (1D array).
        predictions = self.predictor.predict(TabularDataset(X))
        return predictions.to_numpy()

    def __del__(self):
        """
        Attempts to clean up the temporary directory created by AutoGluon
        when the AutoGluonModel object is garbage collected.
        """
        if self.temp_path and os.path.exists(self.temp_path):
            try:
                shutil.rmtree(self.temp_path)
            except OSError as e:
                # Log or print an error if cleanup fails (e.g., directory in use, permissions).
                # In a benchmark environment, the harness might also handle cleanup.
                print(f"Warning: Could not clean up AutoGluon temp directory {self.temp_path}: {e}")


def build_model(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Factory function as required by the evaluation harness.
    It returns a new, unfitted model object that exposes `fit(X,y)` and `predict(X)` methods.

    Args:
        X_train (pd.DataFrame): Training features. This argument is provided by the harness
                                but is not used for fitting within this function, as the
                                returned model is unfitted. It can be used for metadata
                                inference if a custom model's __init__ required it.
        y_train (pd.Series): Training target labels. Similar to X_train, not used for
                                fitting here.

    Returns:
        AutoGluonModel: An unfitted model object ready to be fitted.
    """
    # The AutoGluonModel constructor does not require X_train or y_train,
    # as AutoGluon's TabularPredictor infers necessary metadata during its .fit() call.
    return AutoGluonModel()