import pandas as pd
from autogluon.tabular import TabularPredictor
import os
import tempfile
import shutil
import logging

# Configure logging to suppress AutoGluon's verbose output if desired.
# For a benchmark, less verbose output might be preferred to keep logs clean.
# logging.getLogger('autogluon').setLevel(logging.WARNING) # Example: set to WARNING or ERROR

# Module-level constants
SEED = 42
# Time limit for each fold's fit operation.
# The total remaining search time is 3171 seconds for 10 folds.
# This averages to approximately 317 seconds per fold.
# Setting a conservative time limit to account for overhead,
# potential variance in training times, and to ensure all fits complete
# within the shared deadline.
TIME_LIMIT_PER_FIT = 280 # seconds

class AutoGluonModelWrapper:
    """
    A wrapper class for AutoGluon's TabularPredictor, providing a scikit-learn-like
    fit/predict interface. This wrapper handles model initialization, fitting,
    prediction, and management of AutoGluon's temporary artifact directories.
    """
    def __init__(self):
        self.predictor = None
        self.artifact_path = None
        # The target column name as specified in the problem description
        self.label_column = 'class'

    def fit(self, X, y):
        """
        Fits the AutoGluon TabularPredictor model using the provided training data.
        All preprocessing and model learning are performed within this method,
        using only the data (X, y) supplied for the current fold.

        Args:
            X (pd.DataFrame): Training features for the current fold.
            y (pd.Series or np.ndarray): Training labels for the current fold.
        """
        # Ensure y is a Series with the correct name for concatenation with X.
        # This is crucial for AutoGluon to correctly identify the label column.
        if isinstance(y, pd.Series):
            y_series = y.copy()
            y_series.name = self.label_column # Ensure the name is correct
        else:
            # If y is a numpy array, convert it to a Series with the correct name and index
            y_series = pd.Series(y, name=self.label_column, index=X.index)

        # Combine X and y into a single DataFrame, which is the expected input format for AutoGluon's fit method.
        train_data = pd.concat([X, y_series], axis=1)

        # Determine the problem type (binary or multiclass classification) based on the number of unique labels.
        num_classes = y_series.nunique()
        if num_classes == 2:
            problem_type = 'binary'
        elif num_classes > 2:
            problem_type = 'multiclass'
        else:
            # This case indicates an issue with the target labels (e.g., only one unique class),
            # which is not suitable for classification.
            raise ValueError(f"Invalid number of classes for classification: {num_classes}. Expected 2 or more.")

        # Create a unique temporary directory for AutoGluon artifacts for this specific fold.
        # This ensures "fresh artifact directories per fold" as required by the contract.
        # The directory will be cleaned up when the wrapper object is deleted.
        self.artifact_path = tempfile.mkdtemp(prefix="autogluon_artifacts_")

        # Initialize TabularPredictor.
        # The `random_state` parameter is used for reproducibility across AutoGluon's internal processes.
        # The previous error regarding passing 'seed' to `predictor.fit()` is avoided here.
        self.predictor = TabularPredictor(
            label=self.label_column,
            problem_type=problem_type,
            path=self.artifact_path,
            random_state=SEED,
            # verbosity=0 # Uncomment to suppress most AutoGluon output during fitting
        )

        # Fit the predictor with the specified training data and time limit.
        # The `time_limit` parameter is crucial for bounding each fit operation
        # to ensure all ten fits finish within the shared deadline.
        self.predictor.fit(
            train_data=train_data,
            time_limit=TIME_LIMIT_PER_FIT,
            # AutoGluon's default presets ('medium_quality') are often a good balance
            # between performance and training time. Other presets like 'high_quality'
            # or 'best_quality' could be used if more time were available.
        )

    def predict(self, X):
        """
        Generates class label predictions for the input features using the fitted model.

        Args:
            X (pd.DataFrame): Features to predict on.

        Returns:
            np.ndarray: A 1-dimensional array of predicted class labels,
                        matching the length of the input rows.
        """
        if self.predictor is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit(X, y) first.")

        # AutoGluon's predict method typically returns a pandas Series.
        # Convert it to a 1-dimensional numpy array as required by the prediction output format.
        predictions = self.predictor.predict(X)
        return predictions.to_numpy()

    def __del__(self):
        """
        Cleans up the temporary AutoGluon artifact directory when the wrapper object
        is garbage collected. This helps manage disk space.
        """
        if self.artifact_path and os.path.exists(self.artifact_path):
            try:
                shutil.rmtree(self.artifact_path)
            except OSError as e:
                # Log a warning if cleanup fails, but do not prevent program execution.
                # In some environments, temporary directories might be cleaned by the system.
                print(f"Warning: Could not remove AutoGluon artifact directory {self.artifact_path}: {e}")


def build_model(X_train, y_train):
    """
    Factory function to create a new, unfitted AutoGluon model wrapper.

    This function adheres to the contract by returning an object with `fit(X,y)`
    and `predict(X)` methods. The `X_train` and `y_train` arguments are provided
    for context by the harness but are not directly used to initialize the
    unfitted wrapper in this specific implementation. The actual training data
    for each fold is passed to the wrapper's `fit` method.

    Args:
        X_train (pd.DataFrame): Training features (for context, e.g., to infer feature names or types).
        y_train (pd.Series or np.ndarray): Training labels (for context, e.g., to infer problem type).

    Returns:
        AutoGluonModelWrapper: A fresh, unfitted model wrapper instance ready for training.
    """
    return AutoGluonModelWrapper()