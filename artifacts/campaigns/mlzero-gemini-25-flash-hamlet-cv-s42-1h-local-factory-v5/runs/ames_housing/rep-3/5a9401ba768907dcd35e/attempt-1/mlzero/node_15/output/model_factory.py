import pandas as pd
from autogluon.tabular import TabularPredictor
import tempfile
import shutil
import os
# Fix: The import path for set_seed has been a source of confusion due to AutoGluon API changes.
# Based on the latest error summaries, 'autogluon.common.utils.utils' is the likely correct path
# for the installed version (1.6.4b) as it was the suggested fix to move from 'core'.
from autogluon.common.utils.utils import set_seed

# --- Module-level constants ---
TARGET_COLUMN = 'SalePrice'
RANDOM_STATE = 42
# A reasonable time limit per fold for AutoGluon's fit method.
# The total remaining time is 2033 seconds for 10 folds, which is ~203 seconds per fold.
# Setting it to 180 seconds leaves a buffer for data loading, setup, and cleanup overhead.
AUTOGLUON_TIME_LIMIT_PER_FOLD = 180

# Call set_seed at the module level for global reproducibility.
set_seed(RANDOM_STATE)

# --- Wrapper Class for AutoGluon ---
class AutoGluonModelWrapper:
    """
    A wrapper class to encapsulate AutoGluon's TabularPredictor,
    providing fit and predict methods compatible with the evaluation harness.
    Handles temporary directory management for AutoGluon artifacts and
    ensures the 'Id' column is not used as a feature.
    """
    def __init__(self):
        self.predictor = None
        # Create a unique temporary directory for AutoGluon artifacts for each model instance.
        # This ensures "fresh artifact directories per fold" as required.
        self.output_dir = tempfile.mkdtemp(prefix="autogluon_model_")
        # print(f"AutoGluon output directory: {self.output_dir}") # Uncomment for debugging

    def fit(self, X, y):
        """
        Fits the AutoGluon TabularPredictor on the provided training data.
        All preprocessing and model fitting are performed within this method,
        using only the data (X, y) supplied for the current fold.
        """
        # Ensure X is a DataFrame for consistent operations, as AutoGluon expects it.
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Drop the 'Id' column if it exists, as it must not be used as a feature.
        X_processed = X.drop(columns=['Id'], errors='ignore')

        # Combine features and target into a single DataFrame for AutoGluon.
        train_data = X_processed.copy()
        train_data[TARGET_COLUMN] = y

        # Initialize TabularPredictor.
        # The 'path' argument directs AutoGluon to store its artifacts in our
        # unique temporary directory.
        # Fix: Removed 'random_state' from TabularPredictor constructor as it's not accepted.
        self.predictor = TabularPredictor(
            label=TARGET_COLUMN,
            problem_type='regression',
            eval_metric='rmse',  # Explicitly set RMSE as per task requirements.
            path=self.output_dir,
            verbosity=0  # Suppress AutoGluon output during fit to reduce log noise.
        )

        # Fit the predictor with a time limit and specified presets for efficiency.
        # Fix: Changed 'presets' from 'medium_quality_fast_inference' to 'medium_quality'
        # as the former was reported as an invalid preset.
        # Fix: Removed 'seed' argument directly from .fit() and instead passed 'random_state'
        # via 'ag_args_fit' as per the latest error suggestion for reproducibility.
        self.predictor.fit(
            train_data=train_data,
            time_limit=AUTOGLUON_TIME_LIMIT_PER_FOLD,
            presets='medium_quality',  # Balances speed and performance.
            ag_args_fit={'random_state': RANDOM_STATE}, # Ensures reproducibility for stochastic models.
            # num_cpus=... # Can be set if specific resource limits are known.
            # num_gpus=... # Uncomment and set if GPU is available and desired.
        )

    def predict(self, X):
        """
        Makes predictions using the fitted AutoGluon TabularPredictor.
        """
        # Ensure X is a DataFrame for consistent operations.
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Drop the 'Id' column if it exists, matching the preprocessing during fit.
        X_processed = X.drop(columns=['Id'], errors='ignore')

        # Make predictions on the processed data.
        predictions = self.predictor.predict(X_processed)
        return predictions

    def __del__(self):
        """
        Cleans up the temporary directory created for AutoGluon artifacts.
        This method is called when the object is garbage collected, helping to
        prevent disk space accumulation across multiple folds.
        """
        if os.path.exists(self.output_dir):
            try:
                shutil.rmtree(self.output_dir)
                # print(f"Cleaned up AutoGluon output directory: {self.output_dir}") # Uncomment for debugging
            except Exception as e:
                # print(f"Error cleaning up AutoGluon directory {self.output_dir}: {e}") # Uncomment for debugging
                # Ignore errors during cleanup, e.g., if the directory is already removed
                # or there are permissions issues.
                pass

# --- Factory function ---
def build_model(X_train, y_train):
    """
    Factory function as required by the contract.
    It returns a NEW, unfitted model wrapper instance.
    X_train and y_train are provided for type inference/initialization,
    but the actual fitting happens within the wrapper's fit method.
    """
    # For AutoGluon, the TabularPredictor itself handles type inference and
    # configuration during its .fit() call, so X_train and y_train are not
    # directly used at this stage for model initialization.
    return AutoGluonModelWrapper()