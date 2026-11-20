import os
import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor, TabularDataset
import logging
import random
import shutil
import time

# Configure logging to suppress excessive AutoGluon output.
# AutoGluon can be very verbose, which might impact performance or log readability.
# Setting it to WARNING or ERROR level is often good for production/benchmarking.
logging.getLogger('autogluon').setLevel(logging.WARNING)
logging.getLogger('autogluon.core').setLevel(logging.WARNING)
logging.getLogger('autogluon.tabular').setLevel(logging.WARNING)

# Global constants
TARGET_COLUMN = 'class'
RANDOM_SEED = 42

# Total time limit for all 10 folds is 2927 seconds (from prompt).
# We need to ensure ALL ten fits finish within this.
# A conservative approach is to allocate a fixed time per fold.
# 2927 seconds / 10 folds = 292.7 seconds per fold.
# Setting it slightly lower to account for overhead and ensure all fits complete.
TIME_LIMIT_PER_FIT = 280  # seconds (280 * 10 = 2800 seconds, leaving 127s for overhead)

class AutoGluonTabularWrapper:
    """
    A wrapper class for AutoGluon's TabularPredictor to conform to the
    fit/predict interface required by the benchmark harness.
    """
    def __init__(self, seed=RANDOM_SEED):
        self.predictor = None
        self.seed = seed
        self.model_path = None  # Path for AutoGluon artifacts, unique per fit call

    def fit(self, X, y):
        """
        Fits the AutoGluon TabularPredictor model.

        Args:
            X (pd.DataFrame): Training features.
            y (pd.Series): Training target labels.
        """
        # Combine X and y into a single DataFrame for AutoGluon
        train_df = X.copy()
        train_df[TARGET_COLUMN] = y

        # Create a unique path for AutoGluon artifacts for this fold.
        # This ensures "fresh artifact directories per fold".
        # Using a timestamp and a random integer to guarantee uniqueness.
        # The path is relative to the current working directory, which the harness manages.
        timestamp = int(time.time() * 1000)  # Milliseconds since epoch
        unique_id = f"ag_model_{timestamp}_{random.randint(0, 100000):05d}"
        self.model_path = os.path.join("autogluon_models", unique_id)

        # Ensure the directory exists and is clean.
        # shutil.rmtree is important to prevent issues if a previous run failed
        # and left a directory with the same name (though unlikely with timestamp).
        # It also ensures a truly fresh start for AutoGluon.
        if os.path.exists(self.model_path):
            shutil.rmtree(self.model_path)
        os.makedirs(self.model_path, exist_ok=True)

        # Determine problem type based on unique target values
        problem_type = 'binary' if len(y.unique()) == 2 else 'multiclass'

        # Initialize TabularPredictor.
        # Based on previous errors, random_seed is not passed to the constructor.
        self.predictor = TabularPredictor(
            label=TARGET_COLUMN,
            path=self.model_path,
            eval_metric='balanced_accuracy',  # As specified in README.md
            problem_type=problem_type,
            verbosity=0  # Suppress AutoGluon output to reduce log noise
        )

        # Fit the predictor.
        # The previous errors indicated that `random_seed` should not be a direct
        # argument to `fit()` or the constructor. The most robust way to pass
        # a seed for internal reproducibility is via `ag_args_fit`.
        #
        # The guidance "try to use presets='extreme'" is noted. However, for the
        # 'adult' dataset (~48k rows), 'extreme' preset is typically recommended
        # for datasets <30K samples and may require GPU. Given the shared deadline
        # and general benchmark environment, 'best_quality' is a more robust
        # choice for this dataset size to ensure completion within the time limit.
        self.predictor.fit(
            train_data=TabularDataset(train_df),
            time_limit=TIME_LIMIT_PER_FIT,
            presets='best_quality',
            ag_args_fit={'random_seed': self.seed}, # Pass seed via ag_args_fit for reproducibility
        )

        return self

    def predict(self, X):
        """
        Makes predictions using the fitted AutoGluon TabularPredictor model.

        Args:
            X (pd.DataFrame): Features to make predictions on.

        Returns:
            np.ndarray: Predicted class labels.
        """
        if self.predictor is None:
            raise RuntimeError("Model has not been fitted yet. Call fit(X, y) first.")
        
        # AutoGluon's predict method returns class labels by default.
        # Convert to numpy array for consistency with sklearn-like interfaces.
        predictions = self.predictor.predict(TabularDataset(X))
        return predictions.values

def build_model(X_train, y_train):
    """
    Factory function to build a new, unfitted AutoGluonTabularWrapper model.

    This function is called by the benchmark harness to get a fresh model instance
    for each fold. The X_train and y_train arguments are part of the contract
    but are not used for fitting at this stage; actual fitting happens in the
    returned model's .fit(X, y) method.

    Args:
        X_train (pd.DataFrame): Training features (placeholder, not used for fitting here).
        y_train (pd.Series): Training target (placeholder, not used for fitting here).

    Returns:
        AutoGluonTabularWrapper: An unfitted instance of the model wrapper.
    """
    # Set global seeds for reproducibility for any non-AutoGluon random operations,
    # such as the unique path generation. AutoGluon's internal seed is set in .fit().
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    return AutoGluonTabularWrapper(seed=RANDOM_SEED)