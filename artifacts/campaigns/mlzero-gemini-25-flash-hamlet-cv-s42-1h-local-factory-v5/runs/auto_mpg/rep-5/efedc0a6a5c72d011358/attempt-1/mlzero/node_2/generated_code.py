import pandas as pd
import numpy as np
import os
import uuid
import logging
import shutil
import random

from autogluon.tabular import TabularPredictor

# --- Module-level constants and initialization ---
SEED = 42

# Set global seeds for reproducibility
np.random.seed(SEED)
random.seed(SEED) # For Python's built-in random module

# Configure logging to avoid excessive AutoGluon output during evaluation.
# Set to WARNING or ERROR to reduce verbosity.
logging.basicConfig(level=logging.WARNING)
logging.getLogger('autogluon').setLevel(logging.WARNING)
logging.getLogger('autogluon.core').setLevel(logging.WARNING)
logging.getLogger('autogluon.tabular').setLevel(logging.WARNING)

# Define the features based on the input format provided in the task description.
FEATURES = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model', 'origin']
# The target column is 'class' as specified in README.md.
TARGET_COLUMN = 'class'

# --- Wrapper Class for AutoGluon TabularPredictor ---
class AutoGluonRegressionWrapper:
    """
    A wrapper class for AutoGluon's TabularPredictor to conform to the
    fit/predict interface required by the benchmark harness.
    """
    def __init__(self, path_prefix="autogluon_models_artifacts"):
        """
        Initializes the wrapper.
        Args:
            path_prefix (str): A base directory name for AutoGluon's model artifacts.
                               A unique subdirectory will be created under this for each fit.
        """
        self.predictor = None
        self.path_prefix = path_prefix
        self.model_path = None # To store the unique path for this model instance's artifacts

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fits the AutoGluon TabularPredictor model.

        All preprocessing and model training are performed within this method,
        using only the provided training data (X, y).

        Args:
            X (pd.DataFrame): Training features.
            y (pd.Series): Training target.
        """
        # Create a copy of the input data to avoid modifying the original DataFrames/Series
        # and combine them into a single DataFrame as expected by AutoGluon.
        train_data = X.copy()
        train_data[TARGET_COLUMN] = y.copy()

        # Preprocessing for 'horsepower' column:
        # The 'horsepower' column often contains non-numeric values (e.g., '?') in the raw dataset.
        # Convert it to numeric, coercing errors will turn non-numeric entries into NaN.
        # AutoGluon's internal preprocessing will then handle these NaN values (e.g., by imputation).
        if 'horsepower' in train_data.columns:
            train_data['horsepower'] = pd.to_numeric(train_data['horsepower'], errors='coerce')

        # Create a unique path for AutoGluon artifacts for this specific fold's training.
        # This ensures that each fold's model training is isolated and does not interfere
        # with other folds or previous runs. The path is created relative to the current
        # working directory, which is typically the node's output folder.
        unique_id = str(uuid.uuid4())
        self.model_path = os.path.join(os.getcwd(), self.path_prefix, f"ag_model_{unique_id}")

        # Ensure the unique model directory is clean before AutoGluon uses it.
        # AutoGluon will create the directory if it doesn't exist.
        if os.path.exists(self.model_path):
            shutil.rmtree(self.model_path)

        # Initialize TabularPredictor for regression.
        # - label: Specifies the target column.
        # - path: Specifies the directory to save model artifacts.
        # - problem_type: Explicitly set to 'regression'.
        # - eval_metric: Set to 'rmse' as per the task description.
        # - verbosity: Set to 0 to suppress most AutoGluon output during training.
        self.predictor = TabularPredictor(
            label=TARGET_COLUMN,
            path=self.model_path,
            problem_type='regression',
            eval_metric='rmse',
            verbosity=0
        )

        # Fit the predictor.
        # - time_limit: Crucial for ensuring all ten fits finish within the shared deadline.
        #               Given a total of ~3300 seconds for 10 folds, 300 seconds per fold
        #               is a conservative estimate, allowing for overhead.
        # - presets: 'good_quality' balances model performance with training time.
        # - seed: Pass the global seed for internal reproducibility of AutoGluon's algorithms.
        self.predictor.fit(
            train_data=train_data,
            time_limit=300, # seconds
            presets='good_quality',
            seed=SEED
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generates predictions for the given input features.

        Args:
            X (pd.DataFrame): Features for which to generate predictions.

        Returns:
            np.ndarray: A 1-dimensional array of numerical regression predictions.
        """
        if self.predictor is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Create a copy of the input data for prediction.
        test_data = X.copy()

        # Apply the same preprocessing for 'horsepower' as during fitting.
        if 'horsepower' in test_data.columns:
            test_data['horsepower'] = pd.to_numeric(test_data['horsepower'], errors='coerce')

        # Generate predictions using the fitted predictor.
        predictions = self.predictor.predict(test_data)

        # Return predictions as a 1-dimensional NumPy array.
        return predictions.to_numpy()

# --- Factory Function ---
def build_model(X_train: pd.DataFrame, y_train: pd.Series) -> AutoGluonRegressionWrapper:
    """
    Factory function to create a new, unfitted model wrapper.

    This function is called by the harness to get a fresh model instance for each fold.
    It should return an unfitted model; actual training happens in the wrapper's .fit() method.

    Args:
        X_train (pd.DataFrame): Training features (provided for schema inference, not for fitting here).
        y_train (pd.Series): Training target (provided for schema inference, not for fitting here).

    Returns:
        AutoGluonRegressionWrapper: An unfitted model wrapper with fit/predict methods.
    """
    # X_train and y_train are provided at this stage, but they are not used for fitting here.
    # They can be used for schema inference if a model requires it before fitting,
    # but AutoGluon handles this dynamically within its .fit() method.
    return AutoGluonRegressionWrapper()