import pandas as pd
import numpy as np
import os
import uuid
import random
import shutil
from pathlib import Path

# AutoGluon MultiModal import
from autogluon.multimodal import MultiModalPredictor

# --- Module-level constants and initialization ---
GLOBAL_SEED = 42
TARGET_COLUMN_NAME = 'class'

# Total remaining search time for ALL ten fits is 201 seconds.
# To ensure all fits finish, we allocate approximately 20 seconds per fold.
# This leaves a small buffer for overhead.
FIT_TIME_LIMIT_PER_FOLD = 20  # seconds

# Set global seeds for reproducibility.
# AutoGluon's internal seeding mechanisms are often influenced by these
# global settings, though direct 'seed' arguments to fit methods are not
# always supported by MultiModalPredictor.
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)

class AutoGluonMultiModalWrapper:
    """
    A wrapper class for AutoGluon MultiModalPredictor to conform to the
    fit/predict interface required by the benchmark harness.
    """
    def __init__(self, seed: int = GLOBAL_SEED):
        self.predictor = None
        self.seed = seed
        self.temp_path = None
        # Re-set seeds for this instance's operations if any,
        # ensuring consistency for any non-AutoGluon random operations.
        random.seed(self.seed)
        np.random.seed(self.seed)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fits the AutoGluon MultiModalPredictor model.

        Args:
            X (pd.DataFrame): Training features.
            y (pd.Series): Training labels.
        """
        # Re-set seeds immediately before AutoGluon operations within fit for consistency.
        random.seed(self.seed)
        np.random.seed(self.seed)

        # Create a combined DataFrame for AutoGluon MultiModalPredictor.
        # The predictor expects the label column to be part of the training DataFrame.
        train_data = X.copy()
        train_data[TARGET_COLUMN_NAME] = y

        # Create a unique temporary directory for this fold's artifacts.
        # This ensures "fresh artifact directories per fold" as required.
        # Using pathlib.Path for robust directory management.
        self.temp_path = Path(f"./autogluon_mm_artifacts_{uuid.uuid4().hex}")
        self.temp_path.mkdir(parents=True, exist_ok=True)

        # Initialize MultiModalPredictor.
        # For tabular classification, problem_type='classification' is appropriate.
        # MultiModalPredictor will automatically detect numerical features as tabular.
        self.predictor = MultiModalPredictor(
            label=TARGET_COLUMN_NAME,
            problem_type='classification',
            path=str(self.temp_path),  # MultiModalPredictor expects a string path
            # verbosity=0, # Uncomment to suppress AutoGluon logs during training
        )

        # Fit the predictor.
        # Use 'medium_quality' presets for a balance between accuracy and training speed.
        # 'best_quality' might exceed the per-fold time limit.
        # The 'seed' argument is not directly supported by MultiModalPredictor.fit,
        # as indicated by previous errors. Reproducibility is primarily handled
        # by global seed settings and AG's internal mechanisms.
        self.predictor.fit(
            train_data=train_data,
            time_limit=FIT_TIME_LIMIT_PER_FOLD,
            presets='medium_quality',
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Makes predictions using the fitted AutoGluon MultiModalPredictor model.

        Args:
            X (pd.DataFrame): Features for which to make predictions.

        Returns:
            np.ndarray: A 1-dimensional array of predicted class labels.
        """
        if self.predictor is None:
            raise RuntimeError("Model has not been fitted yet.")
        
        # MultiModalPredictor expects a DataFrame for prediction.
        predictions = self.predictor.predict(X)
        
        # Ensure the output is a 1-dimensional NumPy array of predicted class labels.
        return predictions.to_numpy()

    def __del__(self):
        """
        Cleans up the temporary artifact directory when the object is garbage collected.
        This is a best-effort cleanup. Python's garbage collection timing is not guaranteed.
        """
        if self.temp_path and self.temp_path.exists():
            try:
                shutil.rmtree(self.temp_path)
            except OSError:
                # Log the error if necessary, but do not fail the program.
                # For example: import logging; logging.warning(f"Error removing directory {self.temp_path}: {e}")
                pass

def build_model(X_train: pd.DataFrame, y_train: pd.Series) -> AutoGluonMultiModalWrapper:
    """
    Factory function to build and return a new, unfitted model wrapper
    for tabular classification using AutoGluon MultiModal.

    Args:
        X_train (pd.DataFrame): Training features (provided for contract compliance,
                                not used for fitting at this stage).
        y_train (pd.Series): Training labels (provided for contract compliance,
                              not used for fitting at this stage).

    Returns:
        AutoGluonMultiModalWrapper: An unfitted model wrapper with fit(X,y) and predict(X) methods.
    """
    # Return a fresh, unfitted instance of the wrapper.
    # The actual fitting will happen when the harness calls its .fit() method.
    return AutoGluonMultiModalWrapper(seed=GLOBAL_SEED)