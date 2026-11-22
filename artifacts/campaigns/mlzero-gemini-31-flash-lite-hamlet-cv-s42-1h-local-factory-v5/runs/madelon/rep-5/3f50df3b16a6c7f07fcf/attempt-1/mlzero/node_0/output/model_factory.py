import os
import shutil
import uuid
import pandas as pd
from autogluon.tabular import TabularPredictor

class AutoGluonWrapper:
    """
    A wrapper for AutoGluon TabularPredictor that ensures a fresh environment
    for each fold and adheres to the required fit/predict interface.
    """
    def __init__(self, label='Class'):
        self.label = label
        self.predictor = None
        # Create a unique directory for this specific model instance
        self.path = os.path.join(os.getcwd(), f"ag_model_{uuid.uuid4().hex}")

    def fit(self, X, y):
        # Combine X and y into a single DataFrame for AutoGluon
        train_data = X.copy()
        train_data[self.label] = y
        
        # Initialize and fit the predictor
        # Using 'extreme' preset as requested. 
        # We rely on AutoGluon's internal validation splitting.
        self.predictor = TabularPredictor(
            label=self.label,
            path=self.path,
            eval_metric='balanced_accuracy'
        ).fit(
            train_data=train_data,
            presets='extreme',
            # We do not set a hard time_limit per fold here to allow the 
            # global harness to manage the total time budget across 10 folds.
            # AutoGluon will stop when it exhausts its internal logic or 
            # if the process is interrupted by the harness.
            verbosity=1
        )
        return self

    def predict(self, X):
        if self.predictor is None:
            raise ValueError("Model has not been fitted yet.")
        return self.predictor.predict(X)

    def __del__(self):
        # Cleanup the artifact directory to save disk space
        if self.path and os.path.exists(self.path):
            try:
                shutil.rmtree(self.path)
            except Exception:
                pass

def build_model(X_train, y_train):
    """
    Factory function to return a fresh, unfitted AutoGluonWrapper.
    """
    return AutoGluonWrapper(label='Class')