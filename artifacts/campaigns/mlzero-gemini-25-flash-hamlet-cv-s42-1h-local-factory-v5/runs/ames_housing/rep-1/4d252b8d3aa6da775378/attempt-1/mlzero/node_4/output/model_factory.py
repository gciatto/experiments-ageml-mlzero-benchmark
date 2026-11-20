import os
import numpy as np
import pandas as pd
import tempfile
import shutil
import random
from autogluon.tabular import TabularDataset, TabularPredictor

# Module-level constants
TARGET_COLUMN = 'SalePrice'
ID_COLUMN = 'Id'
RANDOM_SEED = 42
# Total time for 10 folds is ~294 seconds.
# Allocate ~25 seconds per fold for fitting to ensure all fits finish
# within the shared deadline, leaving buffer for overhead.
FIT_TIME_LIMIT_SECONDS = 25 

class AutoGluonTabularRegressionWrapper:
    """
    A wrapper for AutoGluon's TabularPredictor to conform to the build_model contract.
    This wrapper handles model initialization, fitting, prediction, and cleanup
    of AutoGluon's temporary artifact directories.
    """
    def __init__(self):
        self.predictor = None
        self.model_path = None # To store the temporary path for AutoGluon artifacts

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Fits the AutoGluon TabularPredictor on the provided training data.
        All preprocessing and model training are encapsulated within this method,
        using only the training rows for the current fold.
        """
        # Create a temporary directory for AutoGluon artifacts for this fold.
        # This ensures "fresh artifact directories per fold" as required.
        self.model_path = tempfile.mkdtemp(prefix="autogluon_model_")

        # Combine X_train and y_train into a single DataFrame for AutoGluon.
        train_data = X_train.copy()
        train_data[TARGET_COLUMN] = y_train

        # Drop the 'Id' column if it exists, as it must not be used as a feature.
        if ID_COLUMN in train_data.columns:
            train_data = train_data.drop(columns=[ID_COLUMN])

        # Set global seeds for reproducibility for any non-AutoGluon components.
        # AutoGluon also has its own internal random state management, which is set below.
        np.random.seed(RANDOM_SEED)
        random.seed(RANDOM_SEED)

        # Initialize TabularPredictor.
        # eval_metric='root_mean_squared_error' corresponds to RMSE, where lower is better.
        # The 'random_state' argument is removed from the constructor as it's not accepted.
        self.predictor = TabularPredictor(
            label=TARGET_COLUMN,
            eval_metric='root_mean_squared_error',
            path=self.model_path,
        )
        
        # Fit the predictor.
        # The 'medium' preset is chosen for the tight time limit (~25s per fold).
        # The 'extreme' preset is too aggressive for this time constraint and may require GPU.
        # The time_limit is set to ensure all ten fits finish within the shared deadline.
        # 'seed' is passed directly to `fit` for reproducibility of stochastic models.
        self.predictor.fit(
            train_data=TabularDataset(train_data),
            time_limit=FIT_TIME_LIMIT_SECONDS,
            presets='medium', # Changed from 'extreme' to 'medium' due to tight time limit
            seed=RANDOM_SEED, # Changed from random_state to seed as per AutoGluon API
        )

    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        """
        Makes predictions using the fitted AutoGluon TabularPredictor.
        """
        if self.predictor is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Drop the 'Id' column if it exists in the test data.
        test_data = X_test.copy()
        if ID_COLUMN in test_data.columns:
            test_data = test_data.drop(columns=[ID_COLUMN])

        predictions = self.predictor.predict(TabularDataset(test_data))
        return predictions

    def __del__(self):
        """
        Clean up the temporary directory created by AutoGluon when the object is deleted.
        This is a best-effort cleanup to prevent accumulation of temporary files.
        """
        if self.model_path and os.path.exists(self.model_path):
            try:
                shutil.rmtree(self.model_path)
            except OSError:
                # Ignore errors during cleanup, e.g., if the directory is already gone
                # or due to permission issues. This prevents interrupting the evaluation.
                pass

def build_model(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Factory function to create a new, unfitted model wrapper.
    
    Args:
        X_train (pd.DataFrame): Training features (not used for fitting in this function,
                                but provided to conform to the contract).
        y_train (pd.Series): Training target (not used for fitting in this function,
                             but provided to conform to the contract).

    Returns:
        AutoGluonTabularRegressionWrapper: A fresh, unfitted model wrapper instance
                                           with .fit(X,y) and .predict(X) methods.
    """
    # X_train and y_train are provided to build_model but are not used for fitting
    # or initialization within this function. The actual fitting happens when
    # the returned wrapper's .fit() method is called by the evaluation harness.
    return AutoGluonTabularRegressionWrapper()