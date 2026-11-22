import os
import shutil
import tempfile
import uuid
import pandas as pd
from autogluon.tabular import TabularPredictor

class AutoGluonRegressorWrapper:
    """
    A wrapper for AutoGluon TabularPredictor that handles the lifecycle
    of a model within a cross-validation fold.
    """
    def __init__(self, label='class'):
        self.label = label
        self.predictor = None
        self.temp_dir = tempfile.mkdtemp(prefix="ag_fold_")

    def fit(self, X, y):
        # Combine X and y for AutoGluon
        train_data = X.copy()
        train_data[self.label] = y
        
        # Initialize predictor in a fresh directory for this fold
        # Using 'extreme' preset as requested.
        # We limit time to ensure all 10 folds fit within the shared deadline.
        # Given 3576s total, we allocate ~300s per fold.
        self.predictor = TabularPredictor(
            label=self.label,
            path=self.temp_dir,
            eval_metric='rmse'
        ).fit(
            train_data,
            presets='extreme',
            time_limit=300,
            verbosity=0
        )

    def predict(self, X):
        if self.predictor is None:
            raise ValueError("Model has not been fitted yet.")
        return self.predictor.predict(X)

    def __del__(self):
        # Clean up the temporary directory after the fold is processed
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

def build_model(X_train, y_train):
    """
    Factory function to return a fresh, unfitted model wrapper.
    """
    return AutoGluonRegressorWrapper(label='class')