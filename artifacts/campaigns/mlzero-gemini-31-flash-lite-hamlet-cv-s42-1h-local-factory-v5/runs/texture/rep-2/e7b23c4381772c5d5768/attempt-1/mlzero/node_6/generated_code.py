import os
import shutil
import uuid
import pandas as pd
from autogluon.tabular import TabularPredictor

class AutoGluonWrapper:
    """
    A wrapper for AutoGluon Tabular that handles the lifecycle of a predictor
    within a cross-validation fold.
    """
    def __init__(self, label='Class', time_limit=40):
        self.label = label
        self.time_limit = time_limit
        self.predictor = None
        # Create a unique directory for each fold to avoid collisions
        self.artifact_dir = os.path.join(os.getcwd(), f"ag_fold_{uuid.uuid4().hex}")

    def fit(self, X, y):
        # Combine X and y for AutoGluon
        train_data = X.copy()
        train_data[self.label] = y
        
        # Initialize and fit the predictor
        # num_cpus=1 and num_gpus=0 are set to prevent resource contention
        # in restricted container environments.
        # Using 'best_quality' preset for high performance.
        self.predictor = TabularPredictor(
            label=self.label,
            path=self.artifact_dir,
            eval_metric='balanced_accuracy'
        ).fit(
            train_data,
            time_limit=self.time_limit,
            presets='best_quality',
            num_cpus=1,
            num_gpus=0,
            verbosity=0
        )
        return self

    def predict(self, X):
        if self.predictor is None:
            raise ValueError("Model has not been fitted yet.")
        return self.predictor.predict(X)

    def __del__(self):
        # Clean up artifacts after the object is destroyed to save disk space
        if hasattr(self, 'artifact_dir') and os.path.exists(self.artifact_dir):
            try:
                shutil.rmtree(self.artifact_dir, ignore_errors=True)
            except Exception:
                pass

def build_model(X_train, y_train):
    """
    Factory function to return a fresh, unfitted AutoGluonWrapper.
    The time_limit is set to 40 seconds per fold to ensure all 10 folds 
    finish well within the remaining shared deadline of 484 seconds.
    """
    return AutoGluonWrapper(label='Class', time_limit=40)