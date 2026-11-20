import pandas as pd
import numpy as np
import os
import uuid
import shutil
import random # For global seed

from autogluon.tabular import TabularPredictor, TabularDataset

# Attempt to import set_random_seed from autogluon.common.utils.utils.
# This is a common location in recent AutoGluon versions.
# A try-except block handles potential API changes or different AutoGluon versions.
try:
    from autogluon.common.utils.utils import set_random_seed
except ImportError:
    # Fallback if set_random_seed is not found in the expected location.
    # In such cases, we'll rely on numpy and python's random seed,
    # and TabularPredictor's 'random_state' parameter for reproducibility.
    print("Warning: autogluon.common.utils.utils.set_random_seed not found. "
          "Relying on numpy/random seeds and TabularPredictor's random_state.")
    set_random_seed = None # Mark as not available

# Set a global seed for reproducibility across AutoGluon's internal operations
# and other libraries that might be used.
SEED = 42
if set_random_seed:
    set_random_seed(SEED)
else:
    np.random.seed(SEED)
    random.seed(SEED)


class AutoGluonTabularWrapper:
    """
    A wrapper class for AutoGluon's TabularPredictor to conform to the
    build_model factory contract (fit/predict methods).
    """
    def __init__(self, label_column: str, problem_type: str = 'regression'):
        self.label_column = label_column
        self.problem_type = problem_type
        self.predictor = None
        self.model_path = None

        # Generate a unique path for each model instance to store AutoGluon artifacts.
        # This ensures isolation between different folds/model instances.
        # The harness will call build_model for each fold, so each instance needs a unique path.
        self.model_path = f"AutogluonModels_{uuid.uuid4().hex}"
        
        # Note: Global seed is set at module level. TabularPredictor also accepts random_state
        # in its constructor, which is the most direct way to control its internal randomness.

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Fits the AutoGluon TabularPredictor model.

        All learned preprocessing is fitted within this method, using only the
        provided training rows (X_train, y_train).

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target.
        """
        # Combine features and target into a single DataFrame as required by AutoGluon.
        # AutoGluon handles feature type inference and preprocessing automatically.
        train_data = X_train.copy()
        train_data[self.label_column] = y_train

        # Initialize TabularPredictor with specified label, problem type, and evaluation metric.
        # The 'path' argument ensures that model artifacts are stored in a unique directory
        # for this specific fold/model instance.
        # Pass random_state to the constructor for reproducibility of internal splits/initializations.
        self.predictor = TabularPredictor(
            label=self.label_column,
            problem_type=self.problem_type,
            eval_metric='root_mean_squared_error', # The required metric for this regression task
            path=self.model_path,
            random_state=SEED # Pass seed for reproducibility
        )

        # Fit the predictor.
        # time_limit: Bounds the training time for this fold.
        #             Given a total deadline of ~3200 seconds for 10 folds,
        #             300 seconds (5 minutes) per fold is a conservative limit
        #             to ensure all fits complete within the shared deadline.
        # presets='extreme': As per guidance, aims for cutting-edge accuracy.
        #                    Note: This preset can be resource-intensive and may require a GPU.
        # verbosity=0: Suppresses most of AutoGluon's console output during fitting.
        # The 'seed' argument is removed from fit as it's not supported by TabularPredictor.fit().
        self.predictor.fit(
            train_data=TabularDataset(train_data),
            time_limit=300, # Bound training time per fold
            presets='extreme', # Using 'extreme' preset as per guidance
            verbosity=0
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Makes predictions using the fitted AutoGluon TabularPredictor model.

        Args:
            X (pd.DataFrame): Features for which to make predictions.

        Returns:
            np.ndarray: A 1-dimensional array of numerical regression predictions.
                        The length of the array matches the number of input rows.
        """
        if self.predictor is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Make predictions on the input features.
        predictions = self.predictor.predict(TabularDataset(X))

        # AutoGluon's predict method typically returns a Pandas Series.
        # Convert it to a 1-dimensional NumPy array as required by the contract.
        return predictions.values.flatten()

    def __del__(self):
        """
        Destructor to attempt cleaning up the AutoGluon model directory.
        This helps manage disk space, especially in environments with many folds.
        """
        if self.model_path and os.path.exists(self.model_path):
            try:
                shutil.rmtree(self.model_path)
            except OSError as e:
                # Ignore errors during cleanup, as it's not critical for task execution.
                # In a production environment, logging this warning might be useful.
                pass


def build_model(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Factory function to generate a new, unfitted model wrapper for AutoGluon Tabular.

    This function is called by the evaluation harness for each cross-validation fold.
    It returns an object that exposes `fit(X, y)` and `predict(X)` methods.

    Args:
        X_train (pd.DataFrame): Training features (used here primarily for context,
                                but not for fitting at this stage).
        y_train (pd.Series): Training target (used here primarily for context,
                              but not for fitting at this stage).

    Returns:
        AutoGluonTabularWrapper: A fresh, unfitted model wrapper instance.
    """
    # The target column name is explicitly 'class' as per the problem description.
    label_column = 'class'
    return AutoGluonTabularWrapper(label_column=label_column, problem_type='regression')