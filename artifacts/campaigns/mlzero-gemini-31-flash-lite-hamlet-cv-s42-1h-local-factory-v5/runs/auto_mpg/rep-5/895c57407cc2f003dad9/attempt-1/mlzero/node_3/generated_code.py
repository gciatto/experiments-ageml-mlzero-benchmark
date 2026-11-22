import os
import shutil
import uuid
import pandas as pd
from autogluon.tabular import TabularPredictor

class AutoGluonRegressorWrapper:
    """
    A wrapper for AutoGluon TabularPredictor that handles the lifecycle
    of a model within a cross-validation fold.
    """
    def __init__(self):
        # Generate a unique path for this fold's artifacts to avoid collisions
        # Using a unique ID ensures that parallel or sequential folds do not interfere
        self.path = os.path.join(os.getcwd(), f"ag_model_{uuid.uuid4().hex}")
        self.predictor = None
        self.label = 'class'

    def fit(self, X, y):
        """
        Fits the AutoGluon predictor. 
        X and y are provided by the CV harness.
        """
        # Combine X and y into a single DataFrame for AutoGluon
        train_data = X.copy()
        train_data[self.label] = y
        
        # Initialize and fit the predictor.
        # We use 'best_quality' to enable stacking/bagging which is robust for regression.
        # We set a time_limit to ensure the total 10-fold process fits within the 
        # remaining shared deadline (approx 328 seconds per fold).
        self.predictor = TabularPredictor(
            label=self.label,
            path=self.path,
            eval_metric='rmse'
        ).fit(
            train_data,
            presets='best_quality',
            time_limit=300,
            verbosity=1
        )
        return self

    def predict(self, X):
        """
        Predicts using the fitted AutoGluon predictor.
        """
        if self.predictor is None:
            raise ValueError("Model has not been fitted yet.")
        return self.predictor.predict(X)

    def __del__(self):
        """
        Cleanup artifacts after the fold evaluation is complete to save disk space.
        """
        if self.path and os.path.exists(self.path):
            try:
                shutil.rmtree(self.path, ignore_errors=True)
            except Exception:
                pass

def build_model(X_train, y_train):
    """
    Factory function to return a fresh, unfitted model wrapper.
    """
    return AutoGluonRegressorWrapper()