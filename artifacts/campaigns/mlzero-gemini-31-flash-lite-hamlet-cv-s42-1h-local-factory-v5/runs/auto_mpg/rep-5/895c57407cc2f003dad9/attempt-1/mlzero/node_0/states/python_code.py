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
        
        # Initialize and fit the predictor
        # Using 'extreme' preset as requested for high-quality regression
        # We disable auto_stack to ensure we stay within time constraints 
        # while still benefiting from the powerful ensemble models.
        self.predictor = TabularPredictor(
            label=self.label,
            path=self.path,
            eval_metric='rmse'
        ).fit(
            train_data,
            presets='extreme',
            time_limit=240,  # Bound to ensure 10 folds fit in the remaining time
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
        Cleanup artifacts after the fold evaluation is complete.
        """
        if os.path.exists(self.path):
            try:
                shutil.rmtree(self.path)
            except Exception:
                pass

def build_model(X_train, y_train):
    """
    Factory function to return a fresh, unfitted model wrapper.
    """
    return AutoGluonRegressorWrapper()