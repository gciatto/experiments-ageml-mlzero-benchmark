import os
import uuid
import pandas as pd
import numpy as np
from autogluon.multimodal import MultiModalPredictor

# Module-level constants
SEED = 42  # Retained as a constant, but not directly used with AutoGluon's removed seeding functions.
PROBLEM_TYPE = 'classification'
LABEL_COLUMN = 'Class'

# Base directory for AutoGluon artifacts. Each fold will get a unique sub-directory.
# Using /tmp is generally good for temporary files in containerized environments.
# The harness is expected to manage the lifecycle of these temporary directories.
AG_ARTIFACT_BASE_DIR = "/tmp/autogluon_mm_models"

# Time limit per fold for MultiModalPredictor.fit()
# The total shared deadline is 303 seconds for ALL ten fits and overhead.
# Allocating 20 seconds per fold for fitting (10 folds * 20s = 200s)
# leaves 303 - 200 = 103 seconds for environment setup, data loading,
# model initialization, and other harness overhead across all folds.
# This is a conservative estimate to ensure all fits complete within the deadline.
FIT_TIME_LIMIT_SECONDS = 20

class AutoGluonMultiModalWrapper:
    """
    A wrapper class for AutoGluon's MultiModalPredictor to conform to the
    fit/predict interface required by the MLZero benchmark.
    """
    def __init__(self):
        self.predictor = None
        self.model_path = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fits the AutoGluon MultiModalPredictor model.
        All learned preprocessing and model fitting occur within this method,
        using only the provided training data for the current fold.

        Args:
            X (pd.DataFrame): Training features.
            y (pd.Series): Training labels.
        """
        # Ensure a fresh artifact directory for each fit call to isolate folds
        # and prevent data leakage or interference between models.
        unique_id = uuid.uuid4().hex
        self.model_path = os.path.join(AG_ARTIFACT_BASE_DIR, unique_id)
        os.makedirs(self.model_path, exist_ok=True)

        # Combine X and y into a single DataFrame as expected by MultiModalPredictor.
        train_data = X.copy()
        train_data[LABEL_COLUMN] = y

        # Initialize MultiModalPredictor.
        # We explicitly set problem_type and path.
        # verbosity=0 reduces log output during training, which can be helpful
        # in environments with strict logging limits or for cleaner output.
        # No 'random_seed' or 'seed' argument is passed to the constructor or fit method
        # as per previous error fixes and AutoGluon API.
        self.predictor = MultiModalPredictor(
            label=LABEL_COLUMN,
            problem_type=PROBLEM_TYPE,
            path=self.model_path,
            verbosity=0,
        )

        # Fit the model with a predefined time limit.
        # For purely tabular data, MultiModalPredictor will leverage its tabular capabilities.
        # The problem statement indicates "Hyperparameter tuning is unnecessary because the presets is already configured."
        # So, we rely on AutoGluon's default presets for tabular data, which are designed
        # to balance performance and training time.
        self.predictor.fit(
            train_data=train_data,
            time_limit=FIT_TIME_LIMIT_SECONDS,
            # No explicit presets are set here, relying on AutoGluon's intelligent defaults.
            # If specific presets were desired (e.g., 'good_quality', 'medium_quality'),
            # they could be added as a 'presets' argument.
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Makes predictions using the fitted AutoGluon MultiModalPredictor model.

        Args:
            X (pd.DataFrame): Features to make predictions on.

        Returns:
            np.ndarray: Predicted class labels.
        """
        if self.predictor is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit(X, y) first.")
        
        # MultiModalPredictor.predict expects a DataFrame.
        predictions = self.predictor.predict(X)
        # Ensure returning a numpy array as per common ML interface expectations.
        return predictions.to_numpy()


def build_model(X_train: pd.DataFrame, y_train: pd.Series) -> AutoGluonMultiModalWrapper:
    """
    Factory function to build a new, unfitted AutoGluon MultiModal model wrapper.

    Args:
        X_train (pd.DataFrame): Training features (provided for context, not used for fitting here).
        y_train (pd.Series): Training labels (provided for context, not used for fitting here).

    Returns:
        AutoGluonMultiModalWrapper: An unfitted model wrapper with fit/predict methods.
    """
    # The X_train and y_train arguments are part of the contract but are not used
    # to initialize the model itself. The actual fitting happens inside the
    # AutoGluonMultiModalWrapper's .fit() method, ensuring that all preprocessing
    # and learning are done on the specific fold's training data.
    return AutoGluonMultiModalWrapper()