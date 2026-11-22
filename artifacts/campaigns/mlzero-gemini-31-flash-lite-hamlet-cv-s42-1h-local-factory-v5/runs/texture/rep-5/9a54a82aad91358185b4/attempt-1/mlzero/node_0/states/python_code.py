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
    def __init__(self):
        self.predictor = None
        self.temp_dir = tempfile.mkdtemp(prefix="ag_fold_")
        self.label = "Class"

    def fit(self, X, y):
        # Combine X and y into a single DataFrame for AutoGluon
        train_data = X.copy()
        train_data[self.label] = y
        
        # We use a time-constrained fit. Given 10 folds and ~3500s total,
        # we allocate a conservative budget per fold to ensure completion.
        # Using 'best_quality' preset for high performance.
        self.predictor = TabularPredictor(
            label=self.label,
            path=self.temp_dir,
            eval_metric='balanced_accuracy'
        ).fit(
            train_data,
            presets='best_quality',
            time_limit=280,  # Bound to ensure all 10 folds finish within ~3500s
            verbosity=0
        )
        return self

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
    Factory function to return a fresh, unfitted AutoGluonWrapper.
    """
    return AutoGluonWrapper()