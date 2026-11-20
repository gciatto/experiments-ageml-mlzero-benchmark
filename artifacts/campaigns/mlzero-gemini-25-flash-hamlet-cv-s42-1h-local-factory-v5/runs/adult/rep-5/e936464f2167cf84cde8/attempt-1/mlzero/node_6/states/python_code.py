import pandas as pd
import numpy as np
import os
import shutil
import tempfile
import uuid
import random

from autogluon.multimodal import MultiModalPredictor

# Define a global random seed for reproducibility across all folds.
# Based on previous errors, MultiModalPredictor does NOT accept 'random_seed'
# directly in its constructor or fit method in the installed version.
# Therefore, we set global seeds for numpy and the standard 'random' module,
# which AutoGluon often respects for internal operations.
GLOBAL_RANDOM_SEED = 42

# Set global seeds for numpy and random for reproducibility.
np.random.seed(GLOBAL_RANDOM_SEED)
random.seed(GLOBAL_RANDOM_SEED)

class AutoGluonMultiModalWrapper:
    """
    A wrapper for AutoGluon's MultiModalPredictor to conform to the
    fit/predict interface required by the benchmark.
    """
    def __init__(self):
        self.predictor = None
        self.model_path = None
        # The target column name is 'class' as specified in the problem description.
        self.label_column = 'class'

    def fit(self, X, y):
        """
        Fits the AutoGluon MultiModalPredictor.
        All preprocessing and model training happens here, using only the
        provided training data (X, y) for the current fold.
        """
        # Combine X and y into a single DataFrame for AutoGluon.
        # Ensure the target column is named correctly.
        train_df = X.copy()
        train_df[self.label_column] = y

        # Create a unique temporary directory for this fold's model artifacts.
        # This ensures fresh artifact directories per fold, preventing data leakage
        # and managing disk space.
        self.model_path = tempfile.mkdtemp(prefix="ag_multimodal_model_")

        # Initialize MultiModalPredictor.
        # 'random_seed' is removed from the constructor as per previous errors.
        # Global numpy and random seeds are set at the module level for reproducibility.
        self.predictor = MultiModalPredictor(
            label=self.label_column,
            problem_type='classification',
            path=self.model_path,
        )

        # Determine time_limit for fitting.
        # The total remaining time is 2614 seconds for 10 folds.
        # To ensure all ten fits finish within the shared deadline, we set a
        # conservative time limit per fold. 180 seconds (3 minutes) per fold
        # allows for 30 minutes total training time, leaving a buffer for
        # overhead and potential variance in training duration.
        time_limit_per_fold = 180 # seconds

        # Fit the predictor.
        # 'presets' are used to balance model quality and training time.
        # 'good_quality' is a reasonable choice for tabular classification
        # within a limited time budget and to mitigate resource exhaustion
        # issues (like Ray C++ runtime crashes seen with 'best_quality').
        # 'random_seed' is removed from the fit method as it's not a valid argument.
        self.predictor.fit(
            train_data=train_df,
            time_limit=time_limit_per_fold,
            presets='good_quality',
        )

    def predict(self, X):
        """
        Makes predictions using the fitted AutoGluon MultiModalPredictor.
        Returns predicted class labels.
        """
        if self.predictor is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit(X, y) first.")

        # AutoGluon MultiModalPredictor's predict method expects a DataFrame.
        # It returns a pandas Series of predicted labels.
        predictions = self.predictor.predict(X)

        # Convert the pandas Series to a numpy array as typically expected by benchmarks.
        return predictions.values

    def __del__(self):
        """
        Clean up the temporary directory created for model artifacts when the
        wrapper object is deleted. This is crucial for managing disk space
        and ensuring clean state for subsequent folds.
        """
        if self.model_path and os.path.exists(self.model_path):
            try:
                shutil.rmtree(self.model_path)
            except OSError:
                # Ignore errors during cleanup, e.g., if the directory is already
                # gone or permissions issues. This prevents benchmark crashes
                # due to non-critical cleanup failures.
                pass

def build_model(X_train, y_train):
    """
    Factory function to build a new, unfitted AutoGluon MultiModal model wrapper.
    This function adheres to the contract by returning a fresh, unfitted estimator.
    The X_train and y_train arguments are provided by the harness to infer
    problem type or features, but are not used to fit the model at this stage.
    The actual fitting occurs within the wrapper's .fit(X, y) method.
    """
    return AutoGluonMultiModalWrapper()