import pandas as pd
import numpy as np
import tempfile
import shutil
import os

from autogluon.tabular import TabularPredictor
# Corrected import for set_seed based on the latest AutoGluon API changes
from autogluon.core.utils.utils import set_seed

# Module-level constants for reproducibility and resource management
RANDOM_SEED = 42
# The total remaining search time is 3140 seconds.
# For 10-fold cross-validation, a conservative estimate for each fit
# would be 3140 / 10 = 314 seconds per fold.
# We set a time_limit of 280 seconds per fit to allow for
# overheads like data loading, model initialization, and cleanup across folds.
TIME_LIMIT_PER_FIT = 280  # seconds

# The problem statement explicitly defines the target column name.
TARGET_COLUMN_NAME = 'class'

class AutoGluonModel:
    """
    A wrapper class for AutoGluon's TabularPredictor to conform to the
    build_model(X_train, y_train) -> unfitted_model_object contract.
    This class encapsulates the AutoGluon model, its fitting, prediction,
    and artifact management.
    """
    def __init__(self, output_dir: str):
        """
        Initializes the wrapper. The AutoGluon predictor itself is not
        instantiated until .fit() to ensure it's fresh for each fold.

        Args:
            output_dir (str): A unique temporary directory path where AutoGluon
                              will store its models and artifacts for this fold.
        """
        self.output_dir = output_dir
        self.predictor = None  # Will be initialized and fitted in .fit()

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fits the AutoGluon TabularPredictor.
        All preprocessing and model training happens here, using only X and y.

        Args:
            X (pd.DataFrame): Training features for the current fold.
            y (pd.Series): Training target labels for the current fold.
        """
        # Set the seed for reproducibility. This should be called before
        # any stochastic operations, including TabularPredictor initialization
        # and fitting, to ensure consistent results across runs for stochastic models.
        set_seed(RANDOM_SEED)

        # Combine X and y into a single DataFrame for AutoGluon.
        # AutoGluon expects the target column to be part of the DataFrame.
        train_data = X.copy()
        train_data[TARGET_COLUMN_NAME] = y

        # Initialize TabularPredictor with the specified path and metric.
        # AutoGluon will automatically infer the problem_type (binary/multiclass classification)
        # based on the target column's unique values.
        # The 'random_state' argument is not accepted by TabularPredictor constructor
        # in the installed version, nor is 'seed' accepted by .fit().
        # Reproducibility is handled by set_seed(RANDOM_SEED) called at the beginning of fit().
        self.predictor = TabularPredictor(
            label=TARGET_COLUMN_NAME,
            eval_metric='balanced_accuracy',  # Use the specified metric for evaluation
            path=self.output_dir,             # Use the unique path for this fold's artifacts
            # verbosity=0 # Uncomment to suppress AutoGluon output for cleaner logs
        )

        # Fit the predictor.
        # - time_limit: Bounds the training time to ensure all folds complete within the deadline.
        # - presets: 'best_quality' is chosen for high accuracy, balancing performance with the time limit.
        #            'extreme' is often recommended but typically requires a GPU and is best for smaller datasets (<30K samples).
        #            The adult dataset has ~48k rows, making 'best_quality' a safer choice without explicit GPU.
        # - calibrate_decision_threshold: Recommended for metrics like balanced_accuracy in binary classification.
        self.predictor.fit(
            train_data,
            time_limit=TIME_LIMIT_PER_FIT,
            presets='best_quality',
            calibrate_decision_threshold=True
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Makes predictions using the fitted AutoGluon TabularPredictor.

        Args:
            X (pd.DataFrame): Features for which to make predictions.

        Returns:
            np.ndarray: A 1D array of predicted class labels.
        """
        if self.predictor is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit(X, y) first.")
        
        # AutoGluon's predict method returns a pandas Series.
        # Convert it to a numpy array as required by the contract.
        return self.predictor.predict(X).to_numpy()

    def __del__(self):
        """
        Cleans up the temporary directory created for AutoGluon artifacts.
        This helps manage disk space, especially across multiple cross-validation folds.
        """
        # Ensure the output_dir exists and is a valid path before attempting to remove.
        # Check if self.output_dir is set and exists.
        if hasattr(self, 'output_dir') and self.output_dir and os.path.exists(self.output_dir):
            try:
                shutil.rmtree(self.output_dir)
            except OSError as e:
                # Log or print a warning if cleanup fails, but do not block execution.
                # This can happen if the directory is already removed or due to permissions.
                # In a production environment, consider using a proper logging mechanism.
                # print(f"Warning: Could not remove AutoGluon directory {self.output_dir}: {e}")
                pass

def build_model(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Factory function to build an unfitted model.
    This function is called once per cross-validation fold by the evaluation harness.

    Args:
        X_train (pd.DataFrame): Training features (provided for context, not used for fitting here).
        y_train (pd.Series): Training target labels (provided for context, not used for fitting here).

    Returns:
        AutoGluonModel: An unfitted instance of the AutoGluonModel wrapper.
    """
    # Create a unique temporary directory for AutoGluon artifacts for this specific fold.
    # This fulfills the requirement for "fresh artifact directories per fold".
    temp_dir = tempfile.mkdtemp(prefix="autogluon_model_")
    
    # Return an unfitted instance of our wrapper model.
    # The actual fitting will occur when the harness calls model.fit(X, y).
    return AutoGluonModel(output_dir=temp_dir)