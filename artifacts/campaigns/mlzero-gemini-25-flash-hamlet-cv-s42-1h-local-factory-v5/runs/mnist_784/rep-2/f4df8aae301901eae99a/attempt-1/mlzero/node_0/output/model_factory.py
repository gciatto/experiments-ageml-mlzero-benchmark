import pandas as pd
import numpy as np
import random
import os
import tempfile
import shutil
from pathlib import Path
from autogluon.multimodal import MultiModalPredictor

# Set global seed for reproducibility across all stochastic operations.
# AutoGluon MultiModal also accepts a 'seed' parameter in its fit method.
SEED = 42
np.random.seed(SEED)
random.seed(SEED)


class AutoGluonMultiModalWrapper:
    """
    A wrapper class for AutoGluon MultiModalPredictor to conform to the
    fit/predict interface required by the benchmark.
    """

    def __init__(self, label_column_name: str, problem_type: str):
        self.predictor: MultiModalPredictor = None
        self.label_column_name = label_column_name
        self.problem_type = problem_type
        self.model_path: Path = None  # Path to the temporary directory for model artifacts

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fits the AutoGluon MultiModalPredictor model.

        All preprocessing and model fitting are performed within this method,
        using only the provided training data (X, y).

        Args:
            X (pd.DataFrame): Training features.
            y (pd.Series): Training target.
        """
        # Combine X and y into a single DataFrame as required by MultiModalPredictor.
        train_data = X.copy()
        train_data[self.label_column_name] = y

        # Create a temporary directory for model artifacts to ensure a fresh state
        # for each fold and to avoid conflicts across parallel runs or subsequent folds.
        self.model_path = Path(tempfile.mkdtemp())

        # Initialize MultiModalPredictor.
        # The 'path' argument directs AutoGluon to store all model checkpoints and
        # intermediate files within this temporary directory.
        self.predictor = MultiModalPredictor(
            label=self.label_column_name,
            problem_type=self.problem_type,
            path=str(self.model_path),  # Store artifacts in the temporary directory
            # verbosity=0, # Uncomment for silent training logs
        )

        # Fit the predictor.
        # The 'time_limit' parameter is crucial to ensure that each fold's training
        # finishes within a reasonable time, allowing all ten fits to complete
        # inside the shared deadline. 90 seconds per fold * 10 folds = 900 seconds,
        # which is well within the total remaining budget of ~3400 seconds,
        # leaving ample room for overhead.
        self.predictor.fit(
            train_data=train_data,
            time_limit=90,  # seconds per fold
            seed=SEED,      # Set seed for reproducibility of stochastic processes within AutoGluon
            # For tabular data like MNIST pixels (pixel1...pixel784), AutoGluon MultiModal
            # will automatically select appropriate models (e.g., tabular models, or potentially an MLP).
            # No explicit 'image_column' is needed as the features are raw pixel values
            # in tabular format, not image file paths.
        )

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Makes predictions using the fitted AutoGluon MultiModalPredictor model.

        Args:
            X (pd.DataFrame): Features to predict on.

        Returns:
            pd.Series: Predicted class labels.
        """
        if self.predictor is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit(X, y) first.")

        # MultiModalPredictor.predict expects a DataFrame.
        predictions = self.predictor.predict(X)
        return predictions

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Makes probability predictions using the fitted AutoGluon MultiModalPredictor model.

        Args:
            X (pd.DataFrame): Features to predict probabilities for.

        Returns:
            pd.DataFrame: Predicted class probabilities.
        """
        if self.predictor is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit(X, y) first.")

        # MultiModalPredictor.predict_proba expects a DataFrame.
        probabilities = self.predictor.predict_proba(X)
        return probabilities

    def __del__(self):
        """
        Cleans up the temporary directory created for model artifacts.
        This method is called when the object is garbage collected.
        """
        if self.model_path and self.model_path.exists():
            try:
                shutil.rmtree(self.model_path)
            except OSError as e:
                # Log or print a warning if cleanup fails, but do not block execution.
                # This might occur due to file locks or permission issues.
                print(f"Warning: Could not remove temporary directory {self.model_path}: {e}")


def build_model(X_train: pd.DataFrame, y_train: pd.Series) -> AutoGluonMultiModalWrapper:
    """
    Generates a new, unfitted model wrapper for AutoGluon MultiModalPredictor.

    This function acts as the factory for the model, returning an object that
    conforms to the `fit(X,y)` and `predict(X)` interface.

    Args:
        X_train (pd.DataFrame): Training features (used to infer column names/types).
        y_train (pd.Series): Training target (used to infer the label column name).

    Returns:
        AutoGluonMultiModalWrapper: An unfitted model wrapper exposing fit(X,y) and predict(X) methods.
    """
    # Infer the label column name from the y_train Series.
    # If y_train has no name, default to "class" as per problem description.
    label_column_name = y_train.name if y_train.name is not None else "class"

    # The task is classification, and MNIST has 10 distinct digits (0-9),
    # which makes it a multiclass classification problem.
    problem_type = "multiclass"

    return AutoGluonMultiModalWrapper(label_column_name=label_column_name, problem_type=problem_type)