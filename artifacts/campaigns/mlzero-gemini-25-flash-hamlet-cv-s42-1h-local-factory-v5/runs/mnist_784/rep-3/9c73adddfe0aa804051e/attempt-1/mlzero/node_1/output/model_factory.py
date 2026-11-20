import os
import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
import logging
import random

# Set up logging to suppress excessive AutoGluon output.
# AutoGluon can be very verbose. Setting its logger level to WARNING or ERROR
# can reduce output during evaluation, making logs cleaner.
# Uncomment the following lines if verbose AutoGluon logging is an issue.
# logging.getLogger('autogluon').setLevel(logging.WARNING)
# logging.getLogger('autogluon.core').setLevel(logging.WARNING)
# logging.getLogger('autogluon.tabular').setLevel(logging.WARNING)

# Seed for reproducibility across runs.
# This seed will be passed to AutoGluon's fit method.
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

class AutoGluonModelWrapper:
    """
    A wrapper for AutoGluon's TabularPredictor to conform to the fit/predict interface
    required by the evaluation harness.
    """
    def __init__(self, time_limit_per_fold: int):
        """
        Initializes the AutoGluonModelWrapper.

        Args:
            time_limit_per_fold (int): The maximum time in seconds AutoGluon is allowed
                                       to train for a single fold.
        """
        self.predictor = None
        self.time_limit_per_fold = time_limit_per_fold
        self.target_column = 'class' # As specified in the problem description

        # AutoGluon will create its model artifacts in a subdirectory
        # named 'AutogluonModels' within the current working directory.
        # The evaluation harness guarantees a fresh artifact directory per fold,
        # meaning os.getcwd() will be unique for each fold's execution.
        # Thus, a simple relative path is sufficient and safe for AutoGluon's output.
        self.autogluon_path = 'AutogluonModels'

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fits the AutoGluon TabularPredictor using the provided training data.
        All preprocessing and model training are handled by AutoGluon internally.

        Args:
            X (pd.DataFrame): Training features for the current fold.
            y (pd.Series): Training target labels for the current fold.
        """
        # AutoGluon expects X and y to be combined into a single DataFrame.
        # Ensure y is a Series with the correct name ('class') and index.
        if isinstance(y, np.ndarray):
            y = pd.Series(y, name=self.target_column, index=X.index)
        elif isinstance(y, pd.Series):
            y.name = self.target_column
        else: # Handle other iterables like lists
            y = pd.Series(y, name=self.target_column, index=X.index)

        train_data = pd.concat([X, y], axis=1)

        # Initialize TabularPredictor.
        # The problem is a multiclass classification task (MNIST digits).
        # The evaluation metric is 'balanced_accuracy'.
        self.predictor = TabularPredictor(
            label=self.target_column,
            path=self.autogluon_path,
            eval_metric='balanced_accuracy',
            problem_type='multiclass'
        )

        # Fit the model with a time limit and presets for faster training.
        # 'medium_quality_faster_inference' preset offers a good balance
        # between training speed and model performance.
        self.predictor.fit(
            train_data,
            presets='medium_quality_faster_inference',
            time_limit=self.time_limit_per_fold,
            seed=RANDOM_SEED # Pass the global seed for reproducibility
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predicts class labels for the given input features using the fitted model.

        Args:
            X (pd.DataFrame): Features to predict on.

        Returns:
            np.ndarray: A 1-dimensional array of predicted class labels.
        """
        if self.predictor is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit(X, y) first.")
        # AutoGluon's predict method returns a Series, convert to numpy array as required.
        return self.predictor.predict(X).values

def build_model(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Generates a fresh, unfitted AutoGluon model wrapper instance.

    This function serves as the factory for creating new model instances
    for each cross-validation fold.

    Args:
        X_train (pd.DataFrame): Training features. This is provided by the harness
                                for schema inference but is not used directly by
                                the AutoGluonModelWrapper's constructor.
        y_train (pd.Series): Training target. Similar to X_train, not used by
                             the constructor.

    Returns:
        AutoGluonModelWrapper: An unfitted model wrapper instance ready for fitting.
    """
    # The total remaining time for all ten fits is 278 seconds.
    # To ensure all fits complete within the deadline, we allocate a time limit
    # for each individual AutoGluon fit. 25 seconds per fold (250 seconds total)
    # leaves a buffer for harness overhead and other operations.
    per_fold_time_limit = 25
    return AutoGluonModelWrapper(time_limit_per_fold=per_fold_time_limit)