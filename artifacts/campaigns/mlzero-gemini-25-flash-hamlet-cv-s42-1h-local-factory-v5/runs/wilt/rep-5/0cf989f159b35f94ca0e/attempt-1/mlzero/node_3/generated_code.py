import pandas as pd
from autogluon.tabular import TabularPredictor
import os
import uuid
import numpy as np
import random
from autogluon.core.utils.utils import set_seed # Import set_seed for global reproducibility

# --- Module-level constants and configurations ---
# The name of the target column as specified in the problem description.
TARGET_COLUMN = 'class'

# A fixed random seed for reproducibility across different runs and cross-validation folds.
RANDOM_SEED = 42

# Time limit for AutoGluon's fit method for each individual fold.
# The total shared deadline for all ten fits and associated overhead is 2974 seconds.
# Allocating 180 seconds (3 minutes) per fold for AutoGluon's training allows
# 2974 - (10 * 180) = 1174 seconds for data loading, evaluation, and other
# benchmark overheads. This strategy prioritizes completing all fits within
# the shared deadline over using the most computationally intensive presets.
TIME_LIMIT_PER_FIT = 180  # seconds

# --- Wrapper Class for AutoGluon TabularPredictor ---
class AutoGluonTabularWrapper:
    """
    A wrapper class designed to provide a scikit-learn-like `fit` and `predict`
    interface for AutoGluon's TabularPredictor. This allows AutoGluon to be
    integrated seamlessly into evaluation harnesses expecting this interface.
    """
    def __init__(self):
        self.predictor = None
        self.model_path = None

    def fit(self, X, y):
        """
        Fits the AutoGluon TabularPredictor model using the provided training data.

        All learned preprocessing, model selection, and hyperparameter tuning
        are performed internally by AutoGluon within this method, using only
        the supplied `X` and `y` training rows for the current fold.

        Args:
            X (pd.DataFrame): Training features for the current fold.
            y (pd.Series): Training target labels for the current fold.
        """
        # Set seeds for standard Python random, NumPy, and AutoGluon for reproducibility.
        np.random.seed(RANDOM_SEED)
        random.seed(RANDOM_SEED)
        set_seed(RANDOM_SEED) # Apply AutoGluon's global seed

        # AutoGluon's TabularPredictor expects the features and target to be
        # combined into a single pandas DataFrame for its `fit` method.
        train_data = X.copy()
        train_data[TARGET_COLUMN] = y

        # Generate a unique path for this model instance's artifacts.
        # This ensures "fresh artifact directories per fold" as required,
        # preventing conflicts if multiple models are trained (e.g., in parallel
        # or sequentially across different cross-validation folds).
        # Using a temporary directory ensures cleanup if the harness allows,
        # but AutoGluon manages its own path.
        self.model_path = f"autogluon_model_{uuid.uuid4()}"
        
        # Initialize TabularPredictor with specified parameters:
        # - label: The name of the target column.
        # - eval_metric: Set to 'balanced_accuracy' as per the problem description.
        # - path: The unique directory where AutoGluon will save its model artifacts.
        # - verbosity: Set to 0 to suppress verbose output during training,
        #              keeping the console clean.
        # Removed 'random_seed' from constructor as it's not supported in recent AG versions.
        self.predictor = TabularPredictor(
            label=TARGET_COLUMN,
            eval_metric='balanced_accuracy',
            path=self.model_path,
            verbosity=0
        )

        # Fit the predictor with specific configurations:
        # - presets: 'best_quality' is chosen to balance predictive performance
        #            with the strict time limit. While 'extreme' was suggested
        #            in the guidance, it often requires a GPU and significantly
        #            longer training times, which would likely exceed the shared
        #            deadline for all ten folds in a CPU-only environment.
        #            'best_quality' provides strong performance with a wider
        #            range of models and ensemble techniques.
        # - time_limit: Enforces a maximum training duration for this specific fold,
        #               crucial for meeting the overall shared deadline.
        # Removed 'random_seed' from fit method as it's not a valid argument.
        # Global seeding via set_seed() handles reproducibility.
        self.predictor.fit(
            train_data,
            presets='best_quality',
            time_limit=TIME_LIMIT_PER_FIT,
        )

    def predict(self, X):
        """
        Makes predictions using the fitted AutoGluon TabularPredictor model.

        Args:
            X (pd.DataFrame): Features on which to make predictions.

        Returns:
            pd.Series: Predicted class labels for each input row.
        """
        if self.predictor is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit(X, y) first.")
        
        # AutoGluon's predict method directly takes the feature DataFrame
        # and returns the predicted class labels.
        return self.predictor.predict(X)

# --- Factory Function ---
def build_model(X_train, y_train):
    """
    Factory function to generate a new, unfitted model wrapper for
    AutoGluon TabularPredictor.

    This function adheres to the contract by returning a fresh, unfitted
    model instance that exposes `fit(X, y)` and `predict(X)` methods.

    Args:
        X_train (pd.DataFrame): Training features (provided by the harness for context,
                                but not used for fitting within this function).
        y_train (pd.Series): Training target labels (provided for context,
                             but not used for fitting within this function).

    Returns:
        AutoGluonTabularWrapper: An unfitted instance of the model wrapper.
    """
    # The X_train and y_train arguments are provided by the harness for context,
    # but the model returned by this factory function must be unfitted.
    # The actual fitting process occurs when the .fit(X, y) method of the
    # returned AutoGluonTabularWrapper object is subsequently called by the harness.
    return AutoGluonTabularWrapper()