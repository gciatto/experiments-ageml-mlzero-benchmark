import os
import shutil
import uuid
import pandas as pd
from autogluon.tabular import TabularPredictor

class AutoGluonWrapper:
    """
    A wrapper for AutoGluon TabularPredictor that handles the lifecycle
    of a model within a cross-validation fold.
    """
    def __init__(self):
        # Create a unique directory for this fold's artifacts
        self.path = os.path.join(os.getcwd(), f"ag_model_{uuid.uuid4().hex}")
        self.predictor = None
        self.label = 'Class'

    def fit(self, X, y):
        """
        Fits the AutoGluon predictor.
        X: pd.DataFrame of features
        y: pd.Series of targets
        """
        # Combine X and y for AutoGluon's fit method
        train_data = X.copy()
        train_data[self.label] = y
        
        # Initialize and fit
        # Using 'extreme' preset as requested. 
        # We limit time to ensure all 10 folds finish within the shared deadline.
        # Given 3571s total, we allocate ~300s per fold.
        self.predictor = TabularPredictor(
            label=self.label,
            eval_metric='balanced_accuracy',
            path=self.path
        ).fit(
            train_data=train_data,
            presets='extreme',
            time_limit=300,
            verbosity=0
        )
        return self

    def predict(self, X):
        """
        Predicts using the fitted AutoGluon model.
        """
        if self.predictor is None:
            raise ValueError("Model has not been fitted yet.")
        return self.predictor.predict(X)

    def __del__(self):
        """
        Cleanup artifacts to prevent disk exhaustion.
        """
        if os.path.exists(self.path):
            try:
                shutil.rmtree(self.path)
            except Exception:
                pass

def build_model(X_train, y_train):
    """
    Factory function to return a fresh, unfitted AutoGluonWrapper.
    """
    return AutoGluonWrapper()