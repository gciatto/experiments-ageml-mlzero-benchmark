import pandas as pd
import numpy as np
import os
import tempfile
import shutil
import random # For global seed setting

from autogluon.tabular import TabularPredictor, TabularDataset

# --- Module-level constants and configurations ---
# The target column for this regression task.
TARGET_COLUMN = 'SalePrice'
# The problem type, explicitly set to 'regression'.
PROBLEM_TYPE = 'regression'
# Random seed for reproducibility across AutoGluon runs.
RANDOM_SEED = 42

# Total remaining time for 10 folds is 35 seconds.
# We allocate 3 seconds per fold for AutoGluon's fit method.
# This is an extremely tight limit, chosen to attempt completion within the
# very strict overall deadline. AutoGluon's 'medium' preset is used for speed.
FIT_TIME_LIMIT_PER_FOLD = 3 # 3 seconds per fold * 10 folds = 30 seconds. This leaves 5 seconds for overhead across all folds.

# --- Wrapper Class for AutoGluon TabularPredictor ---
class AutoGluonTabularWrapper:
    """
    A wrapper class for AutoGluon's TabularPredictor to conform to the
    fit/predict interface required by the evaluation harness.
    """
    def __init__(self):
        self.predictor = None
        self.temp_path = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Fits the AutoGluon TabularPredictor model.

        All preprocessing and model training are performed within this method,
        using only the provided X_train and y_train data.
        A fresh temporary directory is created for AutoGluon artifacts for each fit call.
        """
        # Set global seeds for reproducibility within this fold's training.
        # This helps ensure that any library or AutoGluon component relying on
        # global random states is consistent for this specific fold.
        np.random.seed(RANDOM_SEED)
        random.seed(RANDOM_SEED)

        # Create a unique temporary directory for AutoGluon artifacts for this specific fit.
        # This ensures isolation between different folds/fits and proper cleanup.
        self.temp_path = tempfile.mkdtemp(prefix='autogluon_model_')

        # Combine X_train and y_train into a single DataFrame as required by AutoGluon.
        # Drop the 'Id' column as it should not be used as a feature, as per task instructions.
        train_df = X_train.copy()
        if 'Id' in train_df.columns:
            train_df = train_df.drop(columns=['Id'])
        train_df[TARGET_COLUMN] = y_train

        # Initialize TabularPredictor.
        # - label: Specifies the target column.
        # - problem_type: Explicitly set to 'regression'.
        # - eval_metric: Set to 'root_mean_squared_error' as per task requirements (RMSE, lower is better).
        # - path: Points to the temporary directory where AutoGluon will save its models and artifacts.
        # - verbosity: Set to 0 to suppress AutoGluon's extensive output during training.
        self.predictor = TabularPredictor(
            label=TARGET_COLUMN,
            problem_type=PROBLEM_TYPE,
            eval_metric='root_mean_squared_error',
            path=self.temp_path,
            verbosity=0
        )

        # Fit the predictor.
        # - train_data: The combined training DataFrame. AutoGluon handles internal validation splits.
        # - time_limit: Bounds the training time to ensure all ten fits complete within the shared deadline.
        # - presets: 'medium' is chosen for speed, given the extremely tight time limit.
        #            'extreme' or 'best_quality' would be too slow.
        # - ag_args_fit: Used to pass random_state for reproducibility to AutoGluon's internal models,
        #                ensuring consistency across different runs of the same model type.
        self.predictor.fit(
            train_data=TabularDataset(train_df),
            time_limit=FIT_TIME_LIMIT_PER_FOLD,
            presets='medium', # Changed from 'extreme' to 'medium' for speed
            ag_args_fit={'random_state': RANDOM_SEED},
        )

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Generates predictions for the given test data using the fitted model.
        """
        if self.predictor is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Drop the 'Id' column from the test data before prediction, consistent with training.
        test_df = X_test.copy()
        if 'Id' in test_df.columns:
            test_df = test_df.drop(columns=['Id'])

        # Make predictions. AutoGluon's predict method returns a pandas Series,
        # which is then converted to a numpy array as required by the contract.
        predictions = self.predictor.predict(TabularDataset(test_df))
        return predictions.values

    def __del__(self):
        """
        Cleans up the temporary directory created for AutoGluon artifacts.
        This method is called when the object is garbage collected.
        """
        if self.temp_path and os.path.exists(self.temp_path):
            try:
                shutil.rmtree(self.temp_path)
            except OSError as e:
                # Log or print a warning if cleanup fails, but do not block program execution.
                # This might occur due to file locks or permission issues.
                print(f"Warning: Could not remove temporary directory {self.temp_path}: {e}")

# --- Factory Function ---
def build_model(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Factory function as required by the evaluation harness.
    It returns a fresh, unfitted instance of the AutoGluonTabularWrapper.

    Args:
        X_train (pd.DataFrame): Training features (not used for initialization, but for type hint).
        y_train (pd.Series): Training target (not used for initialization, but for type hint).

    Returns:
        AutoGluonTabularWrapper: An unfitted model wrapper with .fit() and .predict() methods.
    """
    # X_train and y_train are provided here by the harness but are not used for
    # initializing the model, as the model is returned in an unfitted state.
    # They will be used later when the .fit() method of the returned wrapper is called.
    return AutoGluonTabularWrapper()