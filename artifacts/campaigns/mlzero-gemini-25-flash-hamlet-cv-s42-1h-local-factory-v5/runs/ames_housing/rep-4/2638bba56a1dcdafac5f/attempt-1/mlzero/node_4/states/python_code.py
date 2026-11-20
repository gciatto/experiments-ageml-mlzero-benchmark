import os
import random
import numpy as np
import pandas as pd
import tempfile
import shutil
from autogluon.tabular import TabularPredictor

# Global seed for reproducibility
GLOBAL_SEED = 42
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
# AutoGluon specific seed environment variable for reproducibility
os.environ['AG_RANDOM_SEED'] = str(GLOBAL_SEED)

# Define a wrapper class for AutoGluon to conform to the fit/predict interface
class AutoGluonModelWrapper:
    """
    A wrapper class for AutoGluon's TabularPredictor to provide a scikit-learn-like
    fit/predict interface.
    """
    def __init__(self, target_column: str = 'SalePrice', id_column: str = 'Id'):
        self.target_column = target_column
        self.id_column = id_column
        self.predictor = None
        self.temp_path = None # Path for AutoGluon artifacts

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fits the AutoGluon TabularPredictor model.

        All preprocessing and model fitting are performed within this method,
        using only the provided training data. A unique temporary directory
        is created for AutoGluon artifacts for each fit call.

        Args:
            X: Training features.
            y: Training target values.
        """
        # Create a temporary directory for AutoGluon artifacts for this fold.
        # This ensures "fresh artifact directories per fold" as required.
        self.temp_path = tempfile.mkdtemp()
        
        # Make copies to avoid modifying original DataFrames/Series
        X_processed = X.copy()
        y_processed = y.copy()

        # Drop 'Id' column if it exists, as it's an identifier and not a feature.
        # This adheres to the "Never add row IDs as features" contract.
        if self.id_column in X_processed.columns:
            X_processed = X_processed.drop(columns=[self.id_column])

        # Combine X and y into a single DataFrame for AutoGluon
        train_data = X_processed
        train_data[self.target_column] = y_processed

        # Initialize TabularPredictor
        self.predictor = TabularPredictor(
            label=self.target_column,
            path=self.temp_path,
            eval_metric='rmse', # Explicitly set eval_metric for regression
            problem_type='regression' # Explicitly set problem type
        )

        # Fit the predictor.
        # A time_limit is set to ensure all ten fits finish within the shared deadline.
        # The total remaining time is 1376 seconds for 10 folds, so ~137 seconds per fold.
        # Setting a conservative time_limit of 120 seconds per fold.
        # The 'medium_quality_faster_inference_only_refit' preset balances model quality and training speed.
        # The AG_RANDOM_SEED environment variable handles reproducibility, so 'random_seed' is not passed to fit.
        self.predictor.fit(
            train_data=train_data,
            time_limit=120, # Adjusted time limit per fold to ensure all 10 fits finish within the total deadline
            presets='medium_quality_faster_inference_only_refit',
            # verbosity=0 # Uncomment to reduce AutoGluon log output
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generates predictions for the given input features.

        Args:
            X: Features for which to generate predictions.

        Returns:
            A 1D numpy array of predicted target values.
        """
        if self.predictor is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        
        # Make a copy to avoid modifying original DataFrame
        X_processed = X.copy()

        # Drop 'Id' column if it exists, consistent with fitting.
        if self.id_column in X_processed.columns:
            X_processed = X_processed.drop(columns=[self.id_column])

        predictions = self.predictor.predict(X_processed)
        
        # AutoGluon saves its models to the specified path.
        # Cleaning up the temporary directory immediately after predict might
        # prevent the harness from inspecting the saved model if needed.
        # We rely on the operating system or the harness to clean up temporary
        # directories created by tempfile.mkdtemp() when the process exits or
        # when the experiment run concludes.
        
        return predictions.values # Return as numpy array as per contract

    def __del__(self):
        """
        Attempt to clean up the temporary directory when the object is garbage collected.
        Note: __del__ is not guaranteed to be called.
        """
        if self.temp_path and os.path.exists(self.temp_path):
            try:
                shutil.rmtree(self.temp_path)
            except OSError:
                # Ignore errors during cleanup, e.g., if directory is already removed
                pass

def build_model(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Generates a fresh, unfitted model wrapper for the regression task.

    This function adheres to the factory contract by returning a new, unfitted
    model object that exposes `fit(X, y)` and `predict(X)` methods.
    The `X_train` and `y_train` arguments are provided for context and type hinting,
    but are not used to fit the model at this stage.

    Args:
        X_train: Training features (example for expected input format).
        y_train: Training target (example for expected input format).

    Returns:
        An unfitted model wrapper object with .fit(X, y) and .predict(X) methods.
    """
    return AutoGluonModelWrapper(target_column='SalePrice', id_column='Id')