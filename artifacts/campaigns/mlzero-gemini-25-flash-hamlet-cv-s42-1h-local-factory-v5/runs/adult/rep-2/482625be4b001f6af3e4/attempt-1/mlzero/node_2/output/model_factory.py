import pandas as pd
import numpy as np
import random
import uuid
import shutil
from pathlib import Path

from autogluon.multimodal import MultiModalPredictor

class AutoGluonMultiModalWrapper:
    """
    A wrapper class for AutoGluon MultiModalPredictor to conform to the
    build_model factory contract. It encapsulates the MultiModalPredictor
    and provides `fit` and `predict` methods.
    """
    def __init__(self, label_column: str, problem_type: str, eval_metric: str, path: str):
        """
        Initializes the wrapper with task-specific parameters and a unique path
        for model artifacts.

        Args:
            label_column: The name of the target column.
            problem_type: The type of machine learning problem (e.g., 'classification').
            eval_metric: The metric to optimize during training.
            path: A unique directory path where AutoGluon will store model artifacts.
        """
        self.label_column = label_column
        self.problem_type = problem_type
        self.eval_metric = eval_metric
        self.path = path  # Store as string, as MultiModalPredictor expects string
        self.predictor = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fits the AutoGluon MultiModalPredictor model.
        All preprocessing is handled internally by MultiModalPredictor and fitted
        solely on the provided training data (X, y).

        Args:
            X: Training features.
            y: Training target labels.
        """
        # Set seeds for reproducibility for stochastic processes within AutoGluon.
        # This ensures consistent results across runs for models that use randomness.
        np.random.seed(42)
        random.seed(42)

        # Combine X and y into a single DataFrame as required by MultiModalPredictor.fit.
        train_data = X.copy()
        train_data[self.label_column] = y

        # Initialize MultiModalPredictor.
        # The 'path' argument ensures artifacts are stored in a unique directory per fold,
        # fulfilling the "fresh artifact directories per fold" requirement.
        # 'verbosity=0' can be uncommented to reduce console output during fitting.
        self.predictor = MultiModalPredictor(
            label=self.label_column,
            problem_type=self.problem_type,
            eval_metric=self.eval_metric,
            path=self.path,
            # verbosity=0 # Uncomment to reduce output during training
        )

        # Fit the predictor with a time limit and a suitable preset.
        # 'medium_quality' preset balances performance and training time, suitable for benchmarks.
        # 'time_limit' is set conservatively (90 seconds) to ensure all 10 folds complete
        # within the shared deadline, allowing for data loading, preprocessing, and other overheads.
        self.predictor.fit(
            train_data=train_data,
            presets="medium_quality",  # A valid preset for MultiModalPredictor
            time_limit=90,  # Time limit in seconds for each fold's fit operation
            # num_gpus=1 # Uncomment if you want to explicitly limit GPU usage to 1
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Makes predictions using the fitted AutoGluon MultiModalPredictor.

        Args:
            X: Features for which to make predictions.

        Returns:
            A 1D NumPy array of predicted class labels, corresponding to the input rows.
        """
        if self.predictor is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit(X, y) first.")

        # MultiModalPredictor.predict typically returns a pandas Series for classification labels.
        predictions = self.predictor.predict(X)

        # Convert the predictions to a 1D NumPy array as required by the contract.
        return predictions.to_numpy()

def build_model(X_train: pd.DataFrame, y_train: pd.Series) -> AutoGluonMultiModalWrapper:
    """
    Factory function to create a new, unfitted AutoGluon MultiModal model wrapper.
    This function is called once per cross-validation fold by the evaluation harness.

    Args:
        X_train: Training features (used for context, not for fitting here).
        y_train: Training target labels (used for context, not for fitting here).

    Returns:
        A new, unfitted instance of AutoGluonMultiModalWrapper.
    """
    # Generate a unique path for each model instance (i.e., for each fold).
    # This ensures that each fold's model artifacts are stored separately,
    # preventing interference and ensuring "fresh artifact directories per fold".
    unique_model_path = f"./autogluon_multimodal_model_artifacts_{uuid.uuid4().hex}"

    # Define task-specific parameters as per the problem description.
    label_column = 'class'
    problem_type = 'classification'
    eval_metric = 'balanced_accuracy'  # The metric specified for evaluation

    # Return a new, unfitted instance of our wrapper.
    # The actual fitting will happen when the harness calls model.fit(X, y).
    return AutoGluonMultiModalWrapper(
        label_column=label_column,
        problem_type=problem_type,
        eval_metric=eval_metric,
        path=unique_model_path
    )