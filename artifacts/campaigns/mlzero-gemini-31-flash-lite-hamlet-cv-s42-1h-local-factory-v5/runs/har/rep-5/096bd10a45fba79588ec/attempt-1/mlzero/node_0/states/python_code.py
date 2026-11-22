import os
import shutil
import tempfile
import uuid
import pandas as pd
from autogluon.tabular import TabularPredictor

class AutoGluonWrapper:
    """
    A wrapper for AutoGluon TabularPredictor that handles the lifecycle of 
    a model within a cross-validation fold.
    """
    def __init__(self):
        # Create a unique directory for this fold's artifacts
        self.path = os.path.join(tempfile.gettempdir(), f"ag_model_{uuid.uuid4().hex}")
        self.predictor = None
        self.label = 'Class'

    def fit(self, X, y):
        """
        Fits the AutoGluon predictor on the provided training fold.
        """
        # Combine X and y into a single DataFrame for AutoGluon
        train_data = X.copy()
        train_data[self.label] = y
        
        # Initialize and fit the predictor
        # Using 'extreme' preset as requested, with a time limit to ensure 
        # all 10 folds complete within the shared deadline.
        # We allocate a portion of the remaining time per fold.
        self.predictor = TabularPredictor(
            label=self.label, 
            path=self.path,
            eval_metric='balanced_accuracy'
        ).fit(
            train_data,
            presets='extreme',
            time_limit=250,  # Conservative limit to ensure 10 folds finish
            verbosity=0
        )
        return self

    def predict(self, X):
        """
        Predicts class labels for the provided data.
        """
        if self.predictor is None:
            raise ValueError("Model has not been fitted yet.")
        return self.predictor.predict(X)

    def __del__(self):
        """
        Cleanup artifacts after the model is no longer needed.
        """
        if os.path.exists(self.path):
            try:
                shutil.rmtree(self.path)
            except Exception:
                pass

def build_model(X_train, y_train):
    """
    Factory function to return a fresh, unfitted AutoGluonWrapper instance.
    """
    return AutoGluonWrapper()