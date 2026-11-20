import os
import numpy as np
import pandas as pd
from pathlib import Path
import logging

# Suppress AutoGluon logging verbosity to reduce console output.
# This is particularly useful in automated evaluation environments.
logging.getLogger('autogluon').setLevel(logging.WARNING)
logging.getLogger('autogluon.core').setLevel(logging.WARNING)
logging.getLogger('autogluon.tabular').setLevel(logging.WARNING)

from autogluon.tabular import TabularPredictor
# Correct import for set_seed based on the error message and AutoGluon documentation
from autogluon.common.utils.utils import set_seed

# --- Module-level constants and setup ---
# Define the target column name as specified in the task description.
TARGET_COLUMN = 'class'

# Define the problem type. The task is regression.
PROBLEM_TYPE = 'regression'

# Set a fixed random seed for reproducibility across runs and folds.
RANDOM_SEED = 42

# Specify the evaluation metric. The task uses Root Mean Squared Error (RMSE).
# AutoGluon has a built-in metric for this: 'root_mean_squared_error'.
EVAL_METRIC = 'root_mean_squared_error'

# Estimated time limit per fold in seconds.
# The total shared deadline is 2547 seconds for 10 folds, which is approximately
# 254 seconds per fold. We allocate a conservative amount for the actual model
# fitting to account for Python startup, data loading, and other harness overhead.
# This value might need adjustment if models consistently time out or finish too quickly.
# Using 250 seconds per fit to leave some buffer for other operations.
FIT_TIME_LIMIT_SECONDS = 250


class AutoGluonTabularWrapper:
    """
    A wrapper for AutoGluon's TabularPredictor to conform to the scikit-learn-like
    fit/predict interface required by the evaluation harness.
    """

    def __init__(self, path: str, time_limit: int, seed: int = RANDOM_SEED):
        """
        Initializes the wrapper. The AutoGluon predictor itself is not fitted here.

        Args:
            path (str): The directory path where AutoGluon will save its artifacts.
                        This path must be unique for each fold to prevent conflicts.
            time_limit (int): The maximum time in seconds allowed for the fit operation.
            seed (int): Random seed for reproducibility.
        """
        self.path = path
        self.time_limit = time_limit
        self.seed = seed
        self.predictor = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Fits the AutoGluon TabularPredictor model using the provided training data.
        All preprocessing and model learning occur within this method.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target values.
        """
        # Set the global random seed for AutoGluon operations within this fit call.
        # This addresses the previous errors regarding `random_seed` parameter in constructor/fit.
        set_seed(self.seed)

        # AutoGluon's TabularPredictor expects the target column to be part of the DataFrame
        # passed to its fit method. We combine X_train and y_train for this purpose.
        train_data = X_train.copy()
        train_data[TARGET_COLUMN] = y_train

        # Initialize TabularPredictor.
        # - 'label': Specifies the target column.
        # - 'problem_type': Explicitly sets the task type to regression.
        # - 'eval_metric': Sets the metric for internal model selection and hyperparameter tuning.
        # - 'path': Directs AutoGluon to save its models and artifacts to the unique
        #           directory provided for the current fold.
        # - 'verbosity=0': Suppresses most of AutoGluon's extensive logging during fitting.
        # The `random_seed` parameter is not accepted by the TabularPredictor constructor
        # in the installed AutoGluon version, as indicated by the error.
        self.predictor = TabularPredictor(
            label=TARGET_COLUMN,
            problem_type=PROBLEM_TYPE,
            eval_metric=EVAL_METRIC,
            path=self.path,
            verbosity=0
        )

        # Fit the predictor.
        # - 'time_limit': Ensures the fit operation respects the overall deadline.
        # - 'presets='best_quality'': This preset is chosen for high predictive accuracy.
        #   It automatically enables advanced techniques like stacking and bagging,
        #   which are crucial for AutoGluon's strong performance.
        # The `random_seed` parameter is not accepted by the `fit` method
        # in the installed AutoGluon version, as indicated by the error.
        # Global seeding is handled by `set_seed(self.seed)` call above.
        self.predictor.fit(
            train_data,
            time_limit=self.time_limit,
            presets='best_quality',
        )

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Makes predictions using the fitted AutoGluon TabularPredictor model.

        Args:
            X_test (pd.DataFrame): Features for which to make predictions.

        Returns:
            np.ndarray: A 1-dimensional array of floating-point predictions.
        """
        if self.predictor is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # AutoGluon's predict method directly accepts a pandas DataFrame.
        predictions = self.predictor.predict(X_test)

        # Convert the predictions (which might be a pandas Series) to a NumPy array
        # to match the required output format (1-dimensional array of floats).
        return predictions.to_numpy()


def build_model(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Factory function to build a new, unfitted AutoGluon Tabular model wrapper.

    This function is called by the evaluation harness for each cross-validation fold.
    It must return a fresh, unfitted model instance.

    Args:
        X_train (pd.DataFrame): Training features. This argument is provided by the
                                harness but should not be used to fit the model
                                within this function. It can be used to infer
                                metadata like column names or types if needed.
        y_train (pd.Series): Training target. Similar to X_train, not for fitting here.

    Returns:
        AutoGluonTabularWrapper: An unfitted model wrapper with `fit(X,y)` and `predict(X)` methods.
    """
    # The harness guarantees that `os.getcwd()` points to a unique, writable
    # directory for each fold's execution. We create a subdirectory within it
    # to store AutoGluon's models and temporary artifacts, ensuring isolation.
    artifact_dir = Path(os.getcwd()) / "autogluon_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # Instantiate the wrapper with the unique path and the allocated time limit.
    # The model is unfitted at this point. Its `.fit()` method will be called later
    # by the harness with the actual training data for the current fold.
    return AutoGluonTabularWrapper(
        path=str(artifact_dir),
        time_limit=FIT_TIME_LIMIT_SECONDS,
        seed=RANDOM_SEED
    )