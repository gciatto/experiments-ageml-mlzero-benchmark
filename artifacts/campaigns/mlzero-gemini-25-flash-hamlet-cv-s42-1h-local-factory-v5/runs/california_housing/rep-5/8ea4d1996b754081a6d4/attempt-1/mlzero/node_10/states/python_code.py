import pandas as pd
import numpy as np
import os
import tempfile
import shutil
import sys
import logging
import random # Import random for setting global seed

# Suppress AutoGluon logging verbosity to reduce console output during evaluation.
# This can be adjusted if more detailed logs are needed for debugging.
logging.getLogger('autogluon').setLevel(logging.WARNING)

# AutoGluon is allowed, so import it.
try:
    from autogluon.tabular import TabularPredictor
    # Attempt to import set_seed from the most likely location based on recent AutoGluon versions.
    # If this fails, we'll fall back to numpy/random seeding.
    ag_set_seed = None
    try:
        from autogluon.common.utils.utils import set_seed as ag_set_seed
    except ImportError:
        try:
            from autogluon.core.utils.utils import set_seed as ag_set_seed
        except ImportError:
            # If set_seed is not found in common or core utils, it might be moved or removed.
            # We'll proceed without AutoGluon's specific seed setting, relying on global Python/Numpy seeds.
            print("Warning: AutoGluon's set_seed function not found. Relying on numpy/random seeds.", file=sys.stderr)

except ImportError:
    # Provide a helpful error message if AutoGluon is not installed.
    # The harness should ensure dependencies are met, but this is a robust check.
    raise ImportError(
        "AutoGluon is not installed. Please ensure it is available in the environment "
        "or install it using: pip install autogluon"
    )

# --- Module-level constants and configurations ---
# Seed for reproducibility.
RANDOM_SEED = 42
# Target column name as specified in the problem description.
TARGET_COLUMN = 'median_house_value'
# Time limit for each AutoGluon fit call in seconds.
# The total remaining time is 484 seconds. For 10 folds, 40 seconds per fold
# allows for 400 seconds of fitting, leaving 84 seconds for overhead.
# This is a reasonable heuristic to ensure all fits finish within the shared deadline.
FIT_TIME_LIMIT_SECONDS = 40

# Set global seeds for Python's random module and NumPy for general reproducibility.
# This is a fallback if AutoGluon's specific set_seed is not available or effective.
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if ag_set_seed:
    ag_set_seed(RANDOM_SEED)
    # print(f"AutoGluon specific seed set to {RANDOM_SEED}", file=sys.stderr) # Uncomment for debugging

# --- Wrapper Class for AutoGluon TabularPredictor ---
class AutoGluonTabularWrapper:
    """
    A wrapper class for AutoGluon's TabularPredictor to conform to the
    fit/predict interface required by the factory contract.
    """
    def __init__(self):
        self.predictor = None
        self.temp_path = None
        # A unique identifier for this instance's temporary directory,
        # useful for debugging if cleanup fails or for distinguishing multiple runs.
        self.instance_id = os.urandom(8).hex()

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fits the AutoGluon TabularPredictor model.

        All preprocessing and model fitting are performed within this method
        using only the provided training data (X, y). A unique temporary
        directory is created for AutoGluon artifacts for each fit call.

        Args:
            X (pd.DataFrame): Training features.
            y (pd.Series): Training target values.
        """
        # Ensure X and y are pandas DataFrames/Series, converting if they are numpy arrays.
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y, name=TARGET_COLUMN)

        # Combine features and target into a single DataFrame for AutoGluon.
        train_data = X.copy()
        train_data[TARGET_COLUMN] = y

        # Create a unique temporary directory for AutoGluon artifacts for this fold.
        # This ensures "fresh artifact directories per fold" and avoids conflicts
        # between different model instances or folds.
        self.temp_path = tempfile.mkdtemp(prefix=f"autogluon_model_{self.instance_id}_")
        # print(f"AutoGluon artifacts path for instance {self.instance_id}: {self.temp_path}", file=sys.stderr) # For debugging

        # Initialize TabularPredictor.
        # Based on previous errors, 'random_seed' is NOT accepted in the constructor for the installed version.
        # Global set_seed (if available) or numpy/random seeds are expected to handle reproducibility.
        self.predictor = TabularPredictor(
            label=TARGET_COLUMN,
            path=self.temp_path,
            problem_type='regression',
            # random_seed=RANDOM_SEED # Removed based on error feedback
        )

        # Fit the predictor.
        # 'time_limit' is crucial for bounding the fit time per fold, adhering to the contract.
        # 'seed' or 'random_seed' are NOT accepted by .fit() for the installed version.
        self.predictor.fit(
            train_data=train_data,
            time_limit=FIT_TIME_LIMIT_SECONDS,
            # presets='best_quality' can be used for potentially better performance
            # but might consume more time. Default presets are often a good balance.
            # verbosity=0 # Suppress AutoGluon output during fit if desired
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generates predictions for the given input features using the fitted model.

        Args:
            X (pd.DataFrame): Features for which to generate predictions.

        Returns:
            np.ndarray: A numpy array of predicted numerical (float) values.
        """
        if self.predictor is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit(X, y) first.")

        # Ensure X is a pandas DataFrame, converting if it's a numpy array.
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # AutoGluon's predict method returns a pandas Series; convert to numpy array.
        predictions = self.predictor.predict(X)
        return predictions.values

    def __del__(self):
        """
        Cleans up the temporary directory created by AutoGluon when the object
        is garbage collected. This helps manage disk space.
        """
        if self.temp_path and os.path.exists(self.temp_path):
            try:
                shutil.rmtree(self.temp_path)
                # print(f"Cleaned up AutoGluon artifacts path: {self.temp_path}", file=sys.stderr) # For debugging
            except OSError as e:
                # Log the error if cleanup fails, but do not block execution.
                # This can happen if files are still in use or due to permissions.
                print(f"Warning: Could not remove temporary directory {self.temp_path}: {e}", file=sys.stderr)

# --- Factory Function ---
def build_model(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Factory function to create a new, unfitted model wrapper.

    This function adheres to the contract by returning a fresh, unfitted
    model object that exposes .fit(X,y) and .predict(X) methods.
    X_train and y_train are provided for schema inference or context,
    but no fitting or preprocessing is performed at this stage.

    Args:
        X_train (pd.DataFrame): Training features (provided for context/schema, not used for fitting here).
        y_train (pd.Series): Training target (provided for context/schema, not used for fitting here).

    Returns:
        AutoGluonTabularWrapper: An unfitted model object with .fit() and .predict() methods.
    """
    # For AutoGluon, actual fitting and schema inference happen within the
    # wrapper's .fit() method. We simply return a fresh instance of our wrapper.
    return AutoGluonTabularWrapper()