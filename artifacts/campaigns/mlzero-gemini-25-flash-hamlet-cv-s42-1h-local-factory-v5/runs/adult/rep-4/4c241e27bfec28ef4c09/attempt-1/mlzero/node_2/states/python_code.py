import pandas as pd
import numpy as np
import os
import tempfile
import shutil
import logging
from autogluon.multimodal import MultiModalPredictor
from autogluon.common.utils.utils import set_seed

# Configure logging to avoid excessive output from AutoGluon during benchmark runs.
# This helps keep logs cleaner and focuses on essential information.
logging.basicConfig(level=logging.WARNING)  # Set default logging level to WARNING
logging.getLogger('autogluon').setLevel(logging.WARNING)
logging.getLogger('lightning').setLevel(logging.WARNING)  # AutoGluon uses PyTorch Lightning

# --- Module-level Constants and Configuration ---
RANDOM_SEED = 42
TARGET_COLUMN = 'class'  # The name of the target column as specified in the task.
# Conservative time limit per fold to ensure all 10 folds finish within the total deadline.
# The total remaining time is 3247 seconds. For 10 folds, this averages to ~324 seconds/fold.
# Setting to 240 seconds (4 minutes) leaves ample buffer for overhead and other folds.
TIME_LIMIT_PER_FOLD = 240

# --- Model Wrapper Class ---
class AutoGluonMultiModalWrapper:
    """
    A wrapper class for AutoGluon MultiModalPredictor to conform to the
    fit/predict interface required by the benchmark harness.
    """
    def __init__(self, label_column: str, random_seed: int):
        """
        Initializes the wrapper. The MultiModalPredictor itself is not
        instantiated until the fit method is called.

        Args:
            label_column (str): The name of the target column.
            random_seed (int): Seed for reproducibility.
        """
        self.label_column = label_column
        self.random_seed = random_seed
        self.predictor: MultiModalPredictor = None
        self.output_path: str = None  # Stores the path to the temporary directory

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fits the AutoGluon MultiModalPredictor.

        Args:
            X (pd.DataFrame): Training features.
            y (pd.Series): Training labels.
        """
        # Set the global random seed for reproducibility across AutoGluon components.
        # This should be called before MultiModalPredictor initialization and fit.
        set_seed(self.random_seed)

        # Combine features and labels into a single DataFrame as required by MultiModalPredictor.
        train_data = X.copy()
        train_data[self.label_column] = y

        # Create a unique temporary directory for this fold's model artifacts.
        # This ensures "fresh artifact directories per fold" and avoids conflicts
        # between different folds or runs.
        self.output_path = tempfile.mkdtemp(prefix="automm_model_")

        # Initialize MultiModalPredictor.
        # AutoGluon automatically infers problem_type (binary/multiclass classification)
        # and handles various data modalities (numerical, categorical, text, etc.).
        self.predictor = MultiModalPredictor(
            label=self.label_column,
            path=self.output_path,
            problem_type='classification',  # Explicitly set problem type for clarity.
            # verbosity=0 # Uncomment to suppress most AutoGluon output if needed.
        )

        # Fit the predictor with a time limit to manage overall benchmark duration.
        # AutoGluon's default presets are generally robust and provide a good balance.
        self.predictor.fit(
            train_data=train_data,
            time_limit=TIME_LIMIT_PER_FOLD,
            # Consider adding presets='best_quality' or 'high_quality' if more time is available
            # and higher accuracy is prioritized over speed. For this benchmark,
            # default settings with a time limit are a good balance.
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Makes predictions using the fitted AutoGluon MultiModalPredictor.

        Args:
            X (pd.DataFrame): Features for which to make predictions.

        Returns:
            np.ndarray: A 1D array of predicted class labels.
        """
        if self.predictor is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # MultiModalPredictor.predict returns a pandas Series or DataFrame.
        # Convert it to a 1D numpy array as required by the benchmark contract.
        predictions = self.predictor.predict(X)
        return predictions.to_numpy()

    def __del__(self):
        """
        Cleans up the temporary directory created for model artifacts.
        This is a best-effort cleanup and might not always execute reliably
        (e.g., if the program terminates unexpectedly or the object is not
        garbage collected in a timely manner).
        """
        if self.output_path and os.path.exists(self.output_path):
            try:
                shutil.rmtree(self.output_path)
                # logging.info(f"Cleaned up temporary directory: {self.output_path}") # For debugging
            except OSError as e:
                # Log the error but don't fail, as cleanup is secondary to model execution.
                logging.warning(f"Error removing temporary directory {self.output_path}: {e}")

# --- Factory Function ---
def build_model(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Factory function to create a new, unfitted model object.
    This model object must expose `fit(X,y)` and `predict(X)` methods.

    Args:
        X_train (pd.DataFrame): Training features. Used here primarily for
                                schema inference if needed, but not for fitting.
        y_train (pd.Series): Training labels. Used to determine the name of
                             the target column.

    Returns:
        AutoGluonMultiModalWrapper: An unfitted model object ready for training.
    """
    # Determine the label column name. Prefer y_train.name if available,
    # otherwise fall back to the predefined TARGET_COLUMN constant.
    label_column_name = y_train.name if y_train.name is not None else TARGET_COLUMN
    return AutoGluonMultiModalWrapper(label_column=label_column_name, random_seed=RANDOM_SEED)