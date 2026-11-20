import os
import time
import random
import numpy as np
import pandas as pd
from pathlib import Path
import shutil

# AutoGluon imports
from autogluon.tabular import TabularPredictor

# --- Module-level constants and initialization ---
SEED = 42
TARGET_COLUMN = 'Class'
EVAL_METRIC = 'balanced_accuracy'

# Set seeds for reproducibility
random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
# AutoGluon specific seed environment variable for internal reproducibility
os.environ['AG_TABULAR_RANDOM_SEED'] = str(SEED)

# Define a reasonable time limit per fold.
# The total shared deadline is 3222 seconds for ALL ten fits and overhead.
# A conservative estimate of 200 seconds per fold for fitting leaves
# 1222 seconds for data loading, preprocessing, prediction, evaluation,
# and any other overhead across 10 folds. This aims to ensure all ten fits
# finish within the remaining shared deadline.
FIT_TIME_LIMIT_SECONDS = 200

# --- Wrapper Class for AutoGluon TabularPredictor ---
class AutoGluonTabularWrapper:
    """
    A wrapper class for AutoGluon's TabularPredictor to conform to the
    fit/predict interface required by the harness.
    """
    def __init__(self, target_column: str, eval_metric: str, seed: int, time_limit: int):
        self.target_column = target_column
        self.eval_metric = eval_metric
        self.seed = seed
        self.time_limit = time_limit
        self.predictor = None
        self.model_path = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Fits the AutoGluon TabularPredictor model.
        A unique path is generated for each fit call to ensure fresh artifact directories per fold.
        All learned preprocessing is fitted within this method, using only the training rows.
        """
        # Create a unique path for this model instance to store AutoGluon artifacts.
        # Using PID and nanoseconds ensures high uniqueness across potential parallel runs and folds.
        self.model_path = Path(f"autogluon_model_dir_{os.getpid()}_{time.time_ns()}")
        
        # Ensure the directory is clean if it somehow existed (highly unlikely with time_ns)
        if self.model_path.exists():
            shutil.rmtree(self.model_path)

        # Combine X_train and y_train into a single DataFrame as required by AutoGluon.
        # Make a copy to avoid modifying the original DataFrames if they are views.
        train_data = X_train.copy()
        train_data[self.target_column] = y_train.copy()

        # Initialize TabularPredictor.
        # problem_type is inferred by AutoGluon, but can be explicitly set if known (e.g., 'binary', 'multiclass').
        self.predictor = TabularPredictor(
            label=self.target_column,
            eval_metric=self.eval_metric,
            path=str(self.model_path), # AutoGluon expects a string path for its model directory
            verbosity=0 # Suppress most AutoGluon output to keep logs clean
        )

        # Fit the predictor.
        # 'best_quality' preset is recommended for high accuracy.
        # calibrate_decision_threshold=True is crucial for metrics like balanced_accuracy
        # in binary classification, as it optimizes the decision threshold on AutoGluon's
        # internal validation set.
        self.predictor.fit(
            train_data=train_data,
            time_limit=self.time_limit,
            presets='best_quality',
            seed=self.seed, 
            calibrate_decision_threshold=True,
            # AutoGluon automatically handles internal validation splits.
            # Do not manually split a validation set from the training data.
        )

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Makes predictions using the fitted AutoGluon TabularPredictor.
        """
        if self.predictor is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit(X, y) first.")
        
        # AutoGluon's predict method handles feature transformation internally.
        predictions = self.predictor.predict(X_test)
        # Return as numpy array to conform to typical scikit-learn interface.
        return predictions.values 

# --- Factory function ---
def build_model(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Factory function to create a new, unfitted AutoGluonTabularWrapper model.
    
    Args:
        X_train (pd.DataFrame): Training features (not used for building, but for signature compliance).
        y_train (pd.Series): Training target (not used for building, but for signature compliance).
        
    Returns:
        AutoGluonTabularWrapper: An unfitted model wrapper instance.
    """
    # X_train and y_train are passed to build_model but should not be used to fit
    # or preprocess anything at this stage. They are here for signature compliance.
    # The actual fitting happens inside the wrapper's .fit() method.
    return AutoGluonTabularWrapper(
        target_column=TARGET_COLUMN,
        eval_metric=EVAL_METRIC,
        seed=SEED,
        time_limit=FIT_TIME_LIMIT_SECONDS
    )