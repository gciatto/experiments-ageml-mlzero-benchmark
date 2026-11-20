import os
import numpy as np
import pandas as pd
import tempfile
import shutil
import random
import logging
import warnings

# Suppress warnings from AutoGluon or other libraries that might be verbose.
# This helps keep the output clean during evaluation.
warnings.filterwarnings('ignore')

# Set global seeds for reproducibility.
# This affects numpy, Python's built-in random module, and potentially other
# libraries that rely on these for their random operations.
np.random.seed(42)
random.seed(42)

# Configure logging to avoid excessive output from AutoGluon and its dependencies
# (like PyTorch Lightning). Setting their logger levels to WARNING or ERROR
# reduces verbosity. This is a module-level initialization, which is allowed
# by the contract.
logging.getLogger('autogluon').setLevel(logging.WARNING)
logging.getLogger('lightning').setLevel(logging.WARNING)
logging.getLogger('transformers').setLevel(logging.WARNING) # Often used by AutoGluon MultiModal
logging.getLogger('torch').setLevel(logging.WARNING)

# Import MultiModalPredictor after setting up logging to ensure it respects the level.
from autogluon.multimodal import MultiModalPredictor


class AutoGluonMultiModalWrapper:
    """
    A wrapper for AutoGluon's MultiModalPredictor to conform to the required
    fit(X, y) and predict(X) interface.

    This wrapper handles the creation and cleanup of temporary directories
    for AutoGluon's model artifacts, ensuring "fresh artifact directories per fold".
    """

    def __init__(self, label_column_name: str):
        """
        Initializes the wrapper with the name of the target column.

        Args:
            label_column_name (str): The name of the target column for classification.
        """
        self.predictor = None
        self.label_column_name = label_column_name
        self.model_path = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fits the MultiModalPredictor model using the provided training data.

        All preprocessing and model fitting are performed within this method,
        using only the training rows for the current fold.

        Args:
            X (pd.DataFrame): Training features.
            y (pd.Series): Training labels.
        """
        # Ensure y is a Series with the correct name and index for AutoGluon.
        # If y is an ndarray, convert it to a Series.
        if isinstance(y, np.ndarray):
            y = pd.Series(y, name=self.label_column_name, index=X.index)
        elif isinstance(y, pd.Series):
            # Ensure the Series name matches the expected label_column_name
            y.name = self.label_column_name

        # Combine X and y into a single DataFrame, which MultiModalPredictor expects.
        train_data = X.copy()
        train_data[self.label_column_name] = y

        # Create a unique temporary directory for this fold's model artifacts.
        # This is crucial for the "fresh artifact directories per fold" requirement
        # and isolates each fold's training process.
        self.model_path = tempfile.mkdtemp(prefix="ag_mm_mnist_784_")

        # Initialize MultiModalPredictor.
        # - `label`: Specifies the target column.
        # - `path`: Points to the temporary directory where model artifacts will be saved.
        # - `problem_type`: Explicitly set to 'classification' as per the task.
        self.predictor = MultiModalPredictor(
            label=self.label_column_name,
            path=self.model_path,
            problem_type='classification',
        )

        # Fit the predictor.
        # - `time_limit`: Set to 300 seconds per fold. With 10 folds, this allows
        #   3000 seconds for training, leaving ~275 seconds for overhead within
        #   the total 3275-second deadline. This helps bound each fit.
        # - `presets='medium_quality'`: Balances model quality and training speed,
        #   suitable for tabular data and time constraints.
        # For purely tabular data like MNIST pixels, MultiModalPredictor will
        # automatically infer feature types (all pixel columns are numerical)
        # and use appropriate internal models (e.g., tabular backbones or MLPs).
        self.predictor.fit(
            train_data=train_data,
            time_limit=300,  # seconds per fold
            presets='medium_quality',
        )

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Makes predictions using the fitted MultiModalPredictor model.

        Args:
            X (pd.DataFrame): Features to predict on.

        Returns:
            np.ndarray: A 1-dimensional array of predicted class labels.
                        The length matches the number of input samples.
        """
        if self.predictor is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit(X, y) first.")

        # MultiModalPredictor expects a DataFrame for prediction.
        predictions = self.predictor.predict(X)

        # Ensure the output is a 1-dimensional NumPy array of predicted class labels,
        # as required by the contract.
        return np.asarray(predictions)

    def __del__(self):
        """
        Attempts to clean up the temporary directory created for model artifacts
        when the wrapper object is garbage collected.
        """
        if self.model_path and os.path.exists(self.model_path):
            try:
                shutil.rmtree(self.model_path)
            except OSError as e:
                # Log or ignore errors during cleanup, e.g., if the directory
                # is still in use or due to permission issues.
                # print(f"Error cleaning up temporary directory {self.model_path}: {e}")
                pass


def build_model(X_train: pd.DataFrame, y_train: pd.Series) -> AutoGluonMultiModalWrapper:
    """
    Factory function to build a new, unfitted model wrapper for the classification task.

    This function is called by the harness to obtain a fresh model instance
    for each cross-validation fold.

    Args:
        X_train (pd.DataFrame): Training features (provided for context, not for fitting here).
        y_train (pd.Series): Training labels (provided for context, used to infer target column name).

    Returns:
        AutoGluonMultiModalWrapper: An unfitted model wrapper instance, ready for fitting.
    """
    # The target column name is specified as 'class' in the README.md.
    # We can also infer it from y_train if it's a pandas Series.
    label_column_name = 'class'
    if isinstance(y_train, pd.Series) and y_train.name is not None:
        label_column_name = y_train.name

    return AutoGluonMultiModalWrapper(label_column_name=label_column_name)