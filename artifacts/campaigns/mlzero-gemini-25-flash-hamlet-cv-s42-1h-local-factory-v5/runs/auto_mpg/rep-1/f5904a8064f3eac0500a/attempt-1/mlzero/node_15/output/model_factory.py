import pandas as pd
import numpy as np
import os
import uuid
import logging
import shutil
import random

from autogluon.multimodal import MultiModalPredictor

# Module-level constants
SEED = 42
# The total remaining time for all ten fits and overhead is 137 seconds.
# To ensure all ten fits finish within this deadline, we set a conservative
# time limit per fold. 10 seconds per fold allows for 100 seconds of training
# and 37 seconds for overhead (data loading, model initialization, etc.).
TIME_LIMIT_PER_FOLD = 10

# Configure logging to provide informative messages during execution.
# The harness is expected to capture stdout/stderr.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AutoGluonMMWrapper:
    """
    A wrapper class for AutoGluon MultiModalPredictor to conform to the
    fit/predict interface required by the MLZero benchmark.
    """

    def __init__(self, output_dir_base):
        """
        Initializes the wrapper.

        Args:
            output_dir_base (str): The base directory where model artifacts for
                                   this wrapper instance will be stored.
        """
        self.predictor = None
        self.output_dir_base = output_dir_base
        self.is_fitted = False
        self.fold_output_dir = None  # To store the actual directory used for the current fit

    def fit(self, X, y):
        """
        Fits the AutoGluon MultiModalPredictor model using the provided training data.

        All learned preprocessing is performed within this method, using only
        the training rows for the current fold.

        Args:
            X (pd.DataFrame): Training features for the current fold.
            y (pd.Series): Training target for the current fold.
        """
        # Set the random seed for reproducibility for this specific training run.
        # AutoGluon MultiModalPredictor's fit method accepts a 'seed' argument.
        # Also set general Python and NumPy seeds for broader reproducibility.
        random.seed(SEED)
        np.random.seed(SEED)

        # Create a unique output directory for this specific fold's training artifacts.
        # This ensures "fresh artifact directories per fold" as required.
        self.fold_output_dir = os.path.join(self.output_dir_base, f"fold_model_{uuid.uuid4().hex}")
        os.makedirs(self.fold_output_dir, exist_ok=True)
        logger.info(f"Training AutoGluon MultiModal model for a fold in: {self.fold_output_dir}")

        # Combine X and y into a single DataFrame as required by MultiModalPredictor.
        # The target column name is 'class' as per the problem description.
        train_data = X.copy()
        train_data['class'] = y

        # Initialize MultiModalPredictor.
        # The task is regression, and the target column is 'class'.
        # All input features are numerical, so MultiModalPredictor will treat them as tabular.
        self.predictor = MultiModalPredictor(
            label='class',
            path=self.fold_output_dir,
            problem_type='regression',
            # Using default presets for a balance of speed and quality,
            # relying primarily on the `time_limit` for constraint.
            # No explicit presets are set to avoid issues with invalid preset names.
        )

        # Fit the predictor with the specified time limit and seed.
        # This helps ensure all ten fits finish within the shared deadline.
        self.predictor.fit(
            train_data=train_data,
            time_limit=TIME_LIMIT_PER_FOLD,
            seed=SEED,  # Pass seed directly to fit method
            # verbosity=2 # 0: silent, 1: warning, 2: info, 3: debug
        )
        self.is_fitted = True
        logger.info(f"AutoGluon MultiModal model fitted for a fold. Model saved to: {self.predictor.path}")

    def predict(self, X):
        """
        Makes predictions using the fitted AutoGluon MultiModalPredictor model.

        Args:
            X (pd.DataFrame): Features to make predictions on.

        Returns:
            np.ndarray: Predicted continuous numerical values.
        """
        if not self.is_fitted or self.predictor is None:
            raise RuntimeError("Model has not been fitted yet. Call fit(X, y) first.")

        # AutoGluon MultiModalPredictor expects a DataFrame for prediction.
        predictions = self.predictor.predict(X)

        # Return predictions as a NumPy array, as expected by scikit-learn-like interfaces.
        return predictions.values

    def __del__(self):
        """
        Clean up the temporary model directory when the wrapper object is deleted.
        This is a best-effort cleanup and might not always execute,
        but it's good practice for temporary artifacts.
        It cleans up the specific fold directory if it was created.
        """
        if hasattr(self, 'fold_output_dir') and self.fold_output_dir and os.path.exists(self.fold_output_dir):
            try:
                shutil.rmtree(self.fold_output_dir)
                logger.info(f"Cleaned up model artifacts directory for fold: {self.fold_output_dir}")
            except OSError as e:
                logger.warning(f"Error cleaning up model artifacts directory {self.fold_output_dir}: {e}")


def build_model(X_train, y_train):
    """
    Factory function to create a new, unfitted model wrapper for AutoGluon MultiModal.

    This function is called by the harness to obtain a fresh model instance.
    It does not perform any fitting; fitting occurs within the wrapper's `fit` method.

    Args:
        X_train (pd.DataFrame): Training features (provided for context, not used for fitting here).
        y_train (pd.Series): Training target (provided for context, not used for fitting here).

    Returns:
        AutoGluonMMWrapper: A fresh unfitted model wrapper with `fit` and `predict` methods.
    """
    # Create a unique base directory for all models created by this specific
    # factory instance. This ensures isolation between different calls to build_model
    # if the harness were to call it multiple times for different candidates.
    # Within this base, each fold's `fit` will create its own subdirectory.
    # The harness is expected to run this in a suitable output directory.
    # We create a subdirectory relative to the current working directory.
    # This base directory will contain subdirectories for each fold.
    base_model_output_dir = f"automm_factory_models_{uuid.uuid4().hex}"
    os.makedirs(base_model_output_dir, exist_ok=True)
    logger.info(f"Created base directory for AutoGluon MultiModal models: {base_model_output_dir}")

    return AutoGluonMMWrapper(output_dir_base=base_model_output_dir)