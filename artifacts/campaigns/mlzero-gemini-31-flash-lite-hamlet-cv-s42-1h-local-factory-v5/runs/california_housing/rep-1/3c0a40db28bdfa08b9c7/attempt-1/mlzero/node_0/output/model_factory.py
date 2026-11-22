import os
import shutil
import tempfile
import uuid
import pandas as pd
from autogluon.tabular import TabularPredictor

class AutoGluonWrapper:
    """
    A wrapper for AutoGluon TabularPredictor that handles the lifecycle
    of a model within a cross-validation fold.
    """
    def __init__(self, label='median_house_value'):
        self.label = label
        self.predictor = None
        # Create a unique directory for this fold's artifacts
        self.path = os.path.join(tempfile.gettempdir(), f"ag_model_{uuid.uuid4().hex}")

    def fit(self, X, y):
        """
        Fits the AutoGluon predictor.
        X and y are provided as a training fold.
        """
        # Combine X and y for AutoGluon
        train_data = X.copy()
        train_data[self.label] = y
        
        # Initialize and fit
        # Using 'extreme' preset as requested. 
        # We disable auto_stack to ensure we stay within time limits if needed,
        # but 'extreme' is generally robust.
        self.predictor = TabularPredictor(
            label=self.label,
            path=self.path,
            eval_metric='rmse'
        ).fit(
            train_data,
            presets='extreme',
            time_limit=None,  # Rely on the harness to manage global time
            verbosity=1
        )

    def predict(self, X):
        """
        Predicts using the fitted AutoGluon model.
        """
        if self.predictor is None:
            raise ValueError("Model has not been fitted yet.")
        return self.predictor.predict(X)

    def __del__(self):
        """
        Cleanup temporary artifacts.
        """
        if self.path and os.path.exists(self.path):
            try:
                shutil.rmtree(self.path)
            except Exception:
                pass

def build_model(X_train, y_train):
    """
    Factory function to return a fresh, unfitted AutoGluonWrapper.
    """
    # The target column is fixed by the task description
    return AutoGluonWrapper(label='median_house_value')