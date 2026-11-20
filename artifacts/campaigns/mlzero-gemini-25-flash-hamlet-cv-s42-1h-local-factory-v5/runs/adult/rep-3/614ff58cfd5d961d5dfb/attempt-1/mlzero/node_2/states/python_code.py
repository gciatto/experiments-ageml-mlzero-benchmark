import pandas as pd
import numpy as np
import os
import logging
import random
import uuid

# AutoGluon imports
from autogluon.tabular import TabularPredictor
from autogluon.common.utils.utils import set_seed

# Set a global seed for reproducibility.
# This seed will be used by `autogluon.common.utils.utils.set_seed`
# within the `fit` method to ensure reproducibility for each fold's training.
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Configure logging to suppress excessive output from AutoGluon if desired.
# For this benchmark, it's often better to let AutoGluon log normally for debugging.
# If needed, uncomment and adjust:
# logging.basicConfig(level=logging.INFO) # Or WARNING, ERROR
# logging.getLogger('autogluon').setLevel(logging.WARNING)


class AutoGluonTabularWrapper:
    """
    A wrapper class for AutoGluon's TabularPredictor to conform to the
    fit/predict interface required by the benchmark.
    """
    def __init__(self):
        self.predictor = None
        # Base path for AutoGluon model artifacts.
        # Using /tmp ensures it's a writable temporary location.
        self.base_model_dir = "/tmp/autogluon_models"
        self.current_model_path = None # Path for the current fold's model

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fits the AutoGluon TabularPredictor model.
        All preprocessing and model training happens here, using only the provided X, y.
        """
        # Set seed for reproducibility for this specific fold's training.
        # This ensures that AutoGluon's internal random processes are seeded.
        set_seed(SEED)

        # AutoGluon expects the target column to be part of the DataFrame.
        train_data = X.copy()
        # The contract states `y` is a pandas Series containing the `class` column values,
        # so `y.name` should reliably be 'class'.
        label_column = y.name
        train_data[label_column] = y

        # Create a unique path for this predictor instance's artifacts.
        # This ensures "fresh artifact directories per fold" and prevents conflicts.
        self.current_model_path = os.path.join(self.base_model_dir, str(uuid.uuid4()))
        os.makedirs(self.current_model_path, exist_ok=True)

        # Determine problem type based on the number of unique classes.
        num_unique_classes = y.nunique()
        if num_unique_classes == 2:
            problem_type = 'binary'
        elif num_unique_classes > 2:
            problem_type = 'multiclass'
        else:
            # This case should ideally not happen with valid training data.
            raise ValueError(f"Invalid number of unique classes in target: {num_unique_classes}")

        # Initialize TabularPredictor.
        self.predictor = TabularPredictor(
            label=label_column,
            path=self.current_model_path,
            eval_metric='balanced_accuracy', # Optimize for balanced_accuracy as per task.
            problem_type=problem_type,
            # verbosity=0 # Uncomment to suppress most AutoGluon output.
        )

        # Fit the predictor with a time limit.
        # The total time is 3280 seconds for 10 folds, so approximately 328 seconds per fold.
        # Setting time_limit to 300 seconds leaves a buffer for overhead and other operations.
        self.predictor.fit(
            train_data,
            time_limit=300,
            presets='best_quality', # Aims for best performance, can be resource intensive.
            # ag_args_fit={'num_gpus': 0} # Explicitly disable GPU if not available or causing issues.
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Makes predictions using the fitted AutoGluon model.
        Returns a 1-dimensional numpy array of predicted class labels (strings).
        """
        if self.predictor is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

        predictions = self.predictor.predict(X)
        return predictions.values # Convert pandas Series to numpy array.


def build_model(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Factory function to create a new, unfitted model wrapper.
    X_train and y_train are provided for context by the harness,
    but the actual fitting happens when the wrapper's `fit` method is called.
    """
    return AutoGluonTabularWrapper()