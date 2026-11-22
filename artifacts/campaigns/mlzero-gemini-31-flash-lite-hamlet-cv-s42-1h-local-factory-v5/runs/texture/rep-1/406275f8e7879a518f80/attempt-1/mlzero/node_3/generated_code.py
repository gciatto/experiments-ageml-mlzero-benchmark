import os
import shutil
import tempfile
import uuid
import pandas as pd
from autogluon.tabular import TabularPredictor

class AutoGluonWrapper:
    """
    A wrapper for AutoGluon Tabular that handles the lifecycle of a predictor
    within a cross-validation fold.
    """
    def __init__(self, label='Class'):
        self.label = label
        self.predictor = None
        self.temp_dir = None

    def fit(self, X, y):
        # Create a unique directory for this fold's artifacts
        self.temp_dir = tempfile.mkdtemp(prefix="ag_fold_" + str(uuid.uuid4())[:8])
        
        # Combine X and y for AutoGluon
        train_data = X.copy()
        train_data[self.label] = y
        
        # Initialize and fit the predictor.
        # We use 'balanced_accuracy' as the metric.
        # 'presets' is set to 'medium' to ensure all 10 folds finish within 
        # the very tight remaining time budget (52 seconds total).
        # 'verbosity=0' keeps logs clean.
        self.predictor = TabularPredictor(
            label=self.label,
            eval_metric='balanced_accuracy',
            path=self.temp_dir
        ).fit(
            train_data=train_data,
            presets='medium',
            time_limit=4,  # Strict limit per fold to ensure completion
            verbosity=0
        )

    def predict(self, X):
        if self.predictor is None:
            raise ValueError("Model has not been fitted yet.")
        return self.predictor.predict(X)

    def __del__(self):
        # Clean up artifacts after the object is destroyed
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir, ignore_errors=True)
            except Exception:
                pass

def build_model(X_train, y_train):
    """
    Returns a fresh, unfitted AutoGluonWrapper instance.
    """
    return AutoGluonWrapper(label='Class')