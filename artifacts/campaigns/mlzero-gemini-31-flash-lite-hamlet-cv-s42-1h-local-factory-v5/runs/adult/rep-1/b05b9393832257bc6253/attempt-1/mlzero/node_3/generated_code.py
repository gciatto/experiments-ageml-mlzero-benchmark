import os
import shutil
import uuid
import pandas as pd
from autogluon.tabular import TabularPredictor

class AutoGluonWrapper:
    """
    A wrapper for AutoGluon TabularPredictor that handles fit/predict
    within a self-contained directory structure per fold.
    """
    def __init__(self, label='class'):
        self.label = label
        self.predictor = None
        # Create a unique path for this specific model instance to avoid collisions
        self.path = os.path.join(os.getcwd(), f"ag_model_{uuid.uuid4().hex}")

    def fit(self, X, y):
        # Combine X and y for AutoGluon
        train_data = X.copy()
        train_data[self.label] = y
        
        # Initialize and fit the predictor.
        # We use 'extreme' preset for high performance.
        # We set a time_limit to ensure the 10-fold CV completes within the 
        # shared remaining deadline. 40 seconds per fold is a safe allocation 
        # given the 527s total budget.
        self.predictor = TabularPredictor(
            label=self.label,
            path=self.path,
            eval_metric='balanced_accuracy'
        ).fit(
            train_data,
            presets='extreme',
            time_limit=40,
            verbosity=0
        )

    def predict(self, X):
        if self.predictor is None:
            raise ValueError("Model has not been fitted yet.")
        return self.predictor.predict(X)

    def __del__(self):
        # Cleanup the artifact directory after use to save disk space
        if self.path and os.path.exists(self.path):
            try:
                shutil.rmtree(self.path, ignore_errors=True)
            except Exception:
                pass

def build_model(X_train, y_train):
    """
    Factory function to return a fresh, unfitted AutoGluonWrapper.
    The target column is 'class' based on the dataset description.
    """
    return AutoGluonWrapper(label='class')