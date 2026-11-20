import pandas as pd
from autogluon.tabular import TabularPredictor
import os
import uuid
import random
import numpy as np

# Global seed for reproducibility across the module.
# This ensures that any internal randomness not explicitly controlled by AutoGluon's
# 'seed' parameter in .fit() is also reproducible.
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Time limit per fold in seconds.
# The total deadline for all ten fits is 3240 seconds.
# To ensure all fits complete within the shared deadline, we allocate a conservative
# time limit per fold, accounting for overhead (data loading, model saving, etc.).
# 3240 seconds / 10 folds = 324 seconds per fold on average.
# Setting FIT_TIME_LIMIT to 200 seconds leaves 124 seconds per fold for other operations.
FIT_TIME_LIMIT = 200


class AutoGluonTabularWrapper:
    """
    A wrapper for AutoGluon TabularPredictor to conform to the build_model contract.
    This class provides `fit(X, y)` and `predict(X)` methods.
    """

    def __init__(self, label_column: str):
        """
        Initializes the wrapper. The AutoGluon predictor itself is not fitted here.

        Args:
            label_column (str): The name of the target column.
        """
        self.label_column = label_column
        self.predictor = None
        self.model_path = None  # Stores the path where the predictor artifacts are saved

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fits the AutoGluon TabularPredictor using the provided training data.

        All preprocessing and model fitting are performed within this method,
        using only the supplied training rows. A fresh artifact directory
        is created for each fit call.

        Args:
            X (pd.DataFrame): Training features.
            y (pd.Series): Training labels.
        """
        # Generate a fresh, unique directory for each fit call.
        # This is crucial for cross-validation to prevent artifact conflicts
        # and ensure isolation between folds. The path is relative to the
        # current working directory, which is expected to be within the
        # node output folder managed by the harness.
        self.model_path = f"autogluon_models_{uuid.uuid4()}"

        # Combine X and y into a single DataFrame, as AutoGluon expects
        # the label column to be part of the training data.
        train_data = X.copy()
        train_data[self.label_column] = y

        # Initialize TabularPredictor.
        # - `label`: Specifies the target column.
        # - `eval_metric`: Set to 'balanced_accuracy' as required by the task.
        # - `path`: Directs AutoGluon to save model artifacts to the unique directory.
        self.predictor = TabularPredictor(
            label=self.label_column,
            eval_metric='balanced_accuracy',
            path=self.model_path,
        )

        # Fit the predictor.
        # - `train_data`: The combined DataFrame for training.
        # - `time_limit`: Bounds the training time to ensure all folds complete.
        # - `presets='best'`: Configures AutoGluon for high accuracy, as suggested
        #   in the documentation for competition-winning quality.
        # - `seed`: Ensures reproducibility of the AutoGluon training process.
        # - `calibrate_decision_threshold=True`: Optimizes the decision threshold
        #   for binary classification, which is beneficial for metrics like
        #   balanced accuracy, especially with imbalanced classes.
        self.predictor.fit(
            train_data=train_data,
            time_limit=FIT_TIME_LIMIT,
            presets='best',
            seed=SEED,
            calibrate_decision_threshold=True,
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generates predictions for the given input features.

        Args:
            X (pd.DataFrame): Features for which to generate predictions.

        Returns:
            np.ndarray: A 1-dimensional array of predicted class labels.
        """
        if self.predictor is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit(X, y) first.")

        # AutoGluon's predict method returns a pandas Series.
        # Convert it to a NumPy array as required by the prediction output format.
        predictions = self.predictor.predict(X)
        return predictions.to_numpy()


def build_model(X_train: pd.DataFrame, y_train: pd.Series) -> AutoGluonTabularWrapper:
    """
    Factory function to build a NEW, unfitted AutoGluon Tabular classification model.

    This function is called by the harness to obtain an unfitted model wrapper.
    The wrapper's `fit` method will then be called with the actual training data
    for each cross-validation fold.

    Args:
        X_train (pd.DataFrame): Training features (used to infer target column name).
        y_train (pd.Series): Training labels (used to infer target column name).

    Returns:
        AutoGluonTabularWrapper: An unfitted model wrapper with `fit(X,y)`
                                 and `predict(X)` methods.
    """
    # The target column name is derived from the 'name' attribute of the y_train Series.
    label_column = y_train.name
    if label_column is None:
        # According to the problem description, the target column is 'class'.
        # If y_train.name is unexpectedly None, this indicates an issue with
        # how the Series was created or passed. A robust contract expects y_train.name
        # to be present.
        raise ValueError("y_train Series must have a 'name' attribute to identify the label column.")

    return AutoGluonTabularWrapper(label_column=label_column)