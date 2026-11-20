import pandas as pd
import numpy as np
import os
import tempfile
import shutil
import logging
import random
from autogluon.tabular import TabularPredictor

# Configure logging.
# Set the root logger to WARNING to suppress most of AutoGluon's internal verbose output.
# This is important to keep the output clean for the benchmark harness.
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
# Create a specific logger for this module to output INFO messages if needed.
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) # Set this module's logger to INFO

class AutoGluonTabularWrapper:
    """
    A wrapper class for autogluon.tabular.TabularPredictor that conforms to the
    fit/predict interface required by the benchmark harness.
    """
    def __init__(self):
        self.predictor = None
        self.artifact_path = None # Path for AutoGluon's temporary artifacts
        self.label_column = 'Class' # Target column name as per problem description

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Fits the AutoGluon TabularPredictor model.

        All preprocessing and model fitting are performed within this method,
        using only the provided training data. A fresh temporary directory
        is created for AutoGluon's artifacts for each fit call.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target labels.
        """
        # Set seeds for reproducibility at the start of the fit method.
        # This affects global numpy and python random states.
        np.random.seed(42)
        random.seed(42)

        # Create a unique temporary directory for this fold's AutoGluon artifacts.
        # This ensures isolation and prevents interference between different folds.
        # The benchmark harness should clean up the overall node output folder,
        # but this ensures AG doesn't write to a shared location across folds.
        self.artifact_path = tempfile.mkdtemp()
        logger.info(f"Using temporary path for AutoGluon artifacts: {self.artifact_path}")

        # Combine X_train and y_train into a single DataFrame as required by AutoGluon.
        train_data = X_train.copy()
        train_data[self.label_column] = y_train

        # Initialize TabularPredictor.
        # eval_metric is set to 'balanced_accuracy' as specified in the task.
        # verbosity=0 suppresses most of AutoGluon's extensive logging.
        # Removed 'random_state' from constructor as it's not supported by TabularPredictor.
        self.predictor = TabularPredictor(
            label=self.label_column,
            eval_metric='balanced_accuracy',
            path=self.artifact_path,
            verbosity=0, # Suppress AutoGluon's internal logging
        )

        # Fit the predictor.
        # The task guidance suggests "try to use presets='extreme'". However, 'extreme'
        # often requires a GPU and can be significantly slower, potentially exceeding
        # the per-fold time limit in a CPU-only environment.
        # To ensure ALL ten fits finish inside the remaining shared deadline (3149s total,
        # approx. 314s per fold), 'best_quality' is chosen as a more robust preset
        # that provides high accuracy within typical CPU time limits.
        # A time_limit of 300 seconds per fold is set to allow for overhead.
        time_limit_per_fold = 300 # seconds
        logger.info(f"Fitting model with time_limit={time_limit_per_fold}s and presets='best_quality'.")
        # Removed 'seed' from fit method as it's not a supported argument for TabularPredictor.fit().
        self.predictor.fit(
            train_data,
            presets='best_quality', # Chosen for robustness and time limit adherence.
            time_limit=time_limit_per_fold,
        )
        logger.info("Model fitting complete.")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Makes predictions on new data using the fitted AutoGluon model.

        Args:
            X (pd.DataFrame): Features for which to make predictions.

        Returns:
            np.ndarray: A 1-dimensional NumPy array of predicted class labels.
        """
        if self.predictor is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Make predictions. AutoGluon's predict method returns a Series,
        # convert to NumPy array as required by the contract.
        predictions = self.predictor.predict(X)
        return predictions.to_numpy()

    def __del__(self):
        """
        Attempts to clean up the temporary directory created for AutoGluon artifacts.
        This method is called when the object is garbage collected, but its execution
        is not guaranteed. The benchmark harness should ideally manage temporary
        directory cleanup.
        """
        if self.artifact_path and os.path.exists(self.artifact_path):
            try:
                shutil.rmtree(self.artifact_path)
                logger.info(f"Cleaned up temporary AutoGluon artifact path: {self.artifact_path}")
            except Exception as e:
                logger.warning(f"Error cleaning up AutoGluon artifact path {self.artifact_path}: {e}")


def build_model(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Factory function to build a new, unfitted AutoGluon Tabular model wrapper.

    This function is called by the benchmark harness for each fold to obtain
    a fresh model instance.

    Args:
        X_train (pd.DataFrame): Training features (provided by harness,
                                but not used for fitting or configuration in this factory).
        y_train (pd.Series): Training target (provided by harness,
                             but not used for fitting or configuration in this factory).

    Returns:
        AutoGluonTabularWrapper: An unfitted model wrapper with `fit(X,y)` and `predict(X)` methods.
    """
    # X_train and y_train are provided here to allow for dynamic model configuration
    # based on the data, if necessary. For this AutoGluon setup, the label name
    # is hardcoded to 'Class' as per the problem description, and the model
    # is configured during its `fit` method.
    return AutoGluonTabularWrapper()