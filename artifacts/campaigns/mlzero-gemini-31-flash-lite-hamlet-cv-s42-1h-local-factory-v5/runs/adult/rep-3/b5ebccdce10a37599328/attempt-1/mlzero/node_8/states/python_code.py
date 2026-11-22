import os
import shutil
import tempfile
import uuid
import pandas as pd
from autogluon.tabular import TabularPredictor

class AutoGluonWrapper:
    """
    A wrapper for AutoGluon TabularPredictor that handles the lifecycle
    of the predictor, including directory management and fitting.
    """
    def __init__(self, label):
        self.label = label
        self.predictor = None
        # Create a unique directory for this fold's artifacts
        self.path = os.path.join(tempfile.gettempdir(), f"ag_model_{uuid.uuid4().hex}")

    def fit(self, X, y):
        # Combine X and y into a single DataFrame for AutoGluon
        train_data = X.copy()
        train_data[self.label] = y
        
        # Initialize and fit the predictor.
        # We use 'extreme' preset as requested.
        # We set a time_limit to ensure all 10 folds finish within the shared deadline.
        # With 1119s total, we allocate 100s per fold to leave buffer for overhead.
        self.predictor = TabularPredictor(
            label=self.label,
            eval_metric='balanced_accuracy',
            path=self.path
        ).fit(
            train_data=train_data,
            presets='extreme',
            time_limit=100,
            verbosity=0
        )
        return self

    def predict(self, X):
        if self.predictor is None:
            raise ValueError("Model has not been fitted yet.")
        return self.predictor.predict(X)

    def __del__(self):
        # Clean up the artifact directory after the object is destroyed
        if self.path and os.path.exists(self.path):
            try:
                shutil.rmtree(self.path, ignore_errors=True)
            except Exception:
                pass

def build_model(X_train, y_train):
    """
    Factory function to build a fresh, unfitted AutoGluonWrapper.
    The target column is 'class' based on the task description.
    """
    target_col = 'class'
    return AutoGluonWrapper(label=target_col)