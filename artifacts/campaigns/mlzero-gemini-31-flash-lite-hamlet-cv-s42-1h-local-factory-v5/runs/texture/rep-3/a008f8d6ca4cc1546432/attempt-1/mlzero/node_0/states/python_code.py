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
        
        # Initialize and fit the predictor
        # Using 'balanced_accuracy' as requested.
        # Presets 'medium_quality' is used to ensure we stay within the 
        # shared time budget across 10 folds.
        self.predictor = TabularPredictor(
            label=self.label,
            eval_metric='balanced_accuracy',
            path=self.temp_dir
        ).fit(
            train_data,
            presets='medium_quality',
            time_limit=250,  # Bound per fold to ensure 10 folds finish in time
            verbosity=0
        )
        return self

    def predict(self, X):
        if self.predictor is None:
            raise ValueError("Model has not been fitted yet.")
        return self.predictor.predict(X)

    def __del__(self):
        # Clean up artifacts after the fold evaluation is complete
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
            except Exception:
                pass

def build_model(X_train, y_train):
    """
    Factory function to return a fresh, unfitted model wrapper.
    """
    return AutoGluonWrapper(label='Class')