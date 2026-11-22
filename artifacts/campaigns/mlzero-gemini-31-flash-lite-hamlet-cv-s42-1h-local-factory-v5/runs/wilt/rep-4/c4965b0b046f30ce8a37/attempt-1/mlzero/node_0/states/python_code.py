import os
import uuid
import shutil
import pandas as pd
from autogluon.tabular import TabularPredictor

class AutoGluonWrapper:
    """
    A wrapper for AutoGluon TabularPredictor that ensures a fresh environment
    for each fold and adheres to the fit/predict interface.
    """
    def __init__(self, label='class', time_limit=300):
        self.label = label
        self.time_limit = time_limit
        self.predictor = None
        # Create a unique directory for this specific model instance
        self.path = os.path.join(os.getcwd(), f"ag_model_{uuid.uuid4().hex}")

    def fit(self, X, y):
        # Combine X and y into a single DataFrame for AutoGluon
        train_data = X.copy()
        train_data[self.label] = y
        
        # Initialize and fit the predictor
        # Using 'best_quality' preset for high performance
        # Using 'balanced_accuracy' as the metric
        self.predictor = TabularPredictor(
            label=self.label,
            eval_metric='balanced_accuracy',
            path=self.path
        ).fit(
            train_data,
            time_limit=self.time_limit,
            presets='best_quality',
            verbosity=0
        )
        return self

    def predict(self, X):
        if self.predictor is None:
            raise ValueError("Model has not been fitted yet.")
        return self.predictor.predict(X)

    def __del__(self):
        # Cleanup the artifact directory after the object is destroyed
        if self.path and os.path.exists(self.path):
            try:
                shutil.rmtree(self.path)
            except Exception:
                pass

def build_model(X_train, y_train):
    """
    Factory function to return a fresh, unfitted AutoGluonWrapper.
    The time_limit is set to 300 seconds per fold to ensure all 10 folds
    fit within the remaining shared deadline.
    """
    return AutoGluonWrapper(label='class', time_limit=300)