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
    def __init__(self):
        self.predictor = None
        self.temp_dir = None
        self.label = 'Class'

    def fit(self, X, y):
        # Create a unique directory for this fold's artifacts
        self.temp_dir = tempfile.mkdtemp(prefix="ag_fold_" + str(uuid.uuid4())[:8])
        
        # Combine X and y for AutoGluon
        train_data = X.copy()
        train_data[self.label] = y
        
        # Initialize and fit the predictor.
        # Given the strict 59s total time limit for 10 folds, we use a very 
        # aggressive time_limit and 'fast_inference' presets to ensure completion.
        # We disable heavy ensembling to save time.
        self.predictor = TabularPredictor(
            label=self.label,
            eval_metric='balanced_accuracy',
            path=self.temp_dir
        ).fit(
            train_data,
            presets='medium_quality',
            time_limit=4,
            num_bag_folds=0,
            num_stack_levels=0,
            verbosity=0
        )
        return self

    def predict(self, X):
        if self.predictor is None:
            raise ValueError("Model has not been fitted yet.")
        return self.predictor.predict(X)

    def __del__(self):
        # Clean up artifacts after use
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir, ignore_errors=True)
            except Exception:
                pass

def build_model(X_train, y_train):
    """
    Factory function to return a fresh, unfitted AutoGluonWrapper.
    """
    return AutoGluonWrapper()