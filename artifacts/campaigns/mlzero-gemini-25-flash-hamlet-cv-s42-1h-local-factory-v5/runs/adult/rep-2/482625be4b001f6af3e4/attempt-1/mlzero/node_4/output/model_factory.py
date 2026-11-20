import os
import uuid
import pandas as pd
import numpy as np
import random
from autogluon.tabular import TabularPredictor, TabularDataset
from autogluon.core.metrics import make_scorer
from sklearn.metrics import balanced_accuracy_score

# --- Module-level constants and configurations ---
RANDOM_SEED = 42
TARGET_COLUMN = 'class'
EVAL_METRIC_NAME = 'balanced_accuracy'

# Remaining TOTAL search time, including all ten fits and overhead: 117 seconds.
# For 10 folds, this means roughly 11.7 seconds per fold.
# Allocating 10 seconds per fold for AutoGluon's internal fit,
# which leaves a small buffer for data loading, wrapper overhead, and other operations.
TIME_LIMIT_PER_FOLD_SECONDS = 10 # seconds

# Define a custom scorer for balanced_accuracy.
# This ensures AutoGluon optimizes directly for this metric and handles its specifics.
# balanced_accuracy_score takes true labels (y_true) and predicted labels (y_pred).
ag_balanced_accuracy_scorer = make_scorer(
    name=EVAL_METRIC_NAME,
    score_func=balanced_accuracy_score,
    optimum=1,
    greater_is_better=True,
    needs_class=True # Indicates the score_func expects class predictions, not probabilities.
)

class AutoGluonTabularModel:
    """
    A wrapper for AutoGluon TabularPredictor to conform to the build_model contract.
    This class encapsulates the AutoGluon model, its fitting, and prediction logic.
    """
    def __init__(self):
        self.predictor = None
        # Create a unique path for each predictor instance to ensure fresh artifact
        # directories per fold. This prevents models from different folds overwriting
        # each other's artifacts and ensures isolation.
        # Using /tmp is a common practice for temporary files in Linux environments.
        self.save_path = os.path.join('/tmp', f'autogluon_model_{uuid.uuid4().hex}')
        # AutoGluon will create this directory, so explicit os.makedirs is not strictly
        # necessary.

    def fit(self, X, y):
        """
        Fits the AutoGluon TabularPredictor model using the provided training data.
        All learned preprocessing is performed internally by AutoGluon and fitted
        solely on the input X and y.
        """
        # Set global random seeds for reproducibility.
        np.random.seed(RANDOM_SEED)
        random.seed(RANDOM_SEED)

        # AutoGluon's TabularPredictor expects the target column to be part of
        # the training DataFrame. We create a copy to avoid modifying the original X.
        train_data = X.copy()
        train_data[TARGET_COLUMN] = y

        # Initialize TabularPredictor with the target column, evaluation metric,
        # a unique save path for model artifacts, and random_state for reproducibility.
        # verbosity=0 suppresses AutoGluon's extensive logging during training.
        self.predictor = TabularPredictor(
            label=TARGET_COLUMN,
            eval_metric=ag_balanced_accuracy_scorer,
            path=self.save_path,
            verbosity=0,
            random_state=RANDOM_SEED # FIX: random_state moved to TabularPredictor constructor
        )

        # Fit the predictor.
        # - train_data: The combined features and target for training.
        # - time_limit: Bounds the training time to ensure all folds complete within the deadline.
        # - presets='best_quality': Aims for high predictive accuracy, suitable for competition.
        #   (FIX: Assuming 'best_quality' is a valid preset, as it was in the original code.
        #   The error message mentioned 'medium_quality_fast_inference' which was not in the provided code.)
        # - calibrate_decision_threshold=True: Crucial for optimizing metrics like
        #   balanced_accuracy in binary classification by adjusting the prediction threshold.
        self.predictor.fit(
            train_data=TabularDataset(train_data),
            time_limit=TIME_LIMIT_PER_FOLD_SECONDS, # FIX: Adjusted time limit for the shared deadline
            presets='best_quality',
            calibrate_decision_threshold=True
        )

    def predict(self, X):
        """
        Makes predictions using the fitted AutoGluon TabularPredictor.
        Returns a 1D array of predicted class labels.
        """
        if self.predictor is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit(X, y) first.")

        # AutoGluon's predict method typically returns a pandas Series.
        # Convert it to a NumPy array to match the required output format (1D array).
        predictions = self.predictor.predict(TabularDataset(X))
        return predictions.to_numpy()

def build_model(X_train, y_train):
    """
    Factory function as required by the evaluation harness.
    Returns a new, unfitted model object that exposes `fit(X,y)` and `predict(X)` methods.
    X_train and y_train are provided for context but are not used for fitting
    within this function; fitting occurs when the returned model's .fit() method is called.
    """
    return AutoGluonTabularModel()