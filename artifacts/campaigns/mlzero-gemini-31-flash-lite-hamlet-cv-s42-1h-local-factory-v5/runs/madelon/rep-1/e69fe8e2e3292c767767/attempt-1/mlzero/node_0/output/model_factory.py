import os
import shutil
import uuid
import pandas as pd
from autogluon.tabular import TabularPredictor

class AutoGluonWrapper:
    """
    A wrapper for AutoGluon TabularPredictor that handles directory management
    and ensures a fresh model is trained for each fold.
    """
    def __init__(self, label='Class'):
        self.label = label
        self.predictor = None
        # Create a unique directory for this specific model instance
        self.path = os.path.join(os.getcwd(), f"ag_model_{uuid.uuid4().hex}")

    def fit(self, X, y):
        # Combine features and target for AutoGluon
        train_data = X.copy()
        train_data[self.label] = y
        
        # Clean up any existing directory from previous failed attempts
        if os.path.exists(self.path):
            shutil.rmtree(self.path)
            
        # Initialize and fit the predictor
        # Using 'extreme' preset as requested. 
        # We rely on AutoGluon's internal validation splitting.
        self.predictor = TabularPredictor(
            label=self.label, 
            path=self.path,
            eval_metric='balanced_accuracy'
        ).fit(
            train_data,
            presets='extreme',
            # We limit the time to ensure all 10 folds finish within the shared deadline.
            # Given 3570s total, we allocate ~300s per fold.
            time_limit=300,
            verbosity=1
        )
        return self

    def predict(self, X):
        if self.predictor is None:
            raise ValueError("Model has not been fitted yet.")
        return self.predictor.predict(X)

def build_model(X_train, y_train):
    """
    Factory function to return a fresh, unfitted AutoGluonWrapper.
    """
    return AutoGluonWrapper(label='Class')