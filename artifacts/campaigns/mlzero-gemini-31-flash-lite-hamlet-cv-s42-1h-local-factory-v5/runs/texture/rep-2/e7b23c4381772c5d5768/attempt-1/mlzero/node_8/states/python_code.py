import os
import shutil
import uuid
import pandas as pd
from autogluon.tabular import TabularPredictor

class AutoGluonWrapper:
    """
    A wrapper for AutoGluon Tabular designed for strict sequential execution
    within a cross-validation loop.
    """
    def __init__(self, label='Class'):
        self.label = label
        self.predictor = None
        # Use a unique directory per fold to avoid file system contention
        self.artifact_dir = os.path.join(os.getcwd(), f"ag_fold_{uuid.uuid4().hex}")

    def fit(self, X, y):
        # Prepare training data for AutoGluon
        train_data = X.copy()
        train_data[self.label] = y
        
        # We use a very constrained time_limit to ensure all 10 folds complete
        # within the remaining shared deadline.
        # num_bag_folds=0 and num_stack_levels=0 disable internal parallelization
        # to prevent segmentation faults in restricted environments.
        self.predictor = TabularPredictor(
            label=self.label,
            path=self.artifact_dir,
            eval_metric='balanced_accuracy'
        ).fit(
            train_data,
            time_limit=1.0,  # Aggressive limit to ensure total CV completion
            presets='medium_quality',
            num_bag_folds=0,
            num_stack_levels=0,
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
        # Clean up artifacts to prevent disk exhaustion
        if self.artifact_dir and os.path.exists(self.artifact_dir):
            try:
                shutil.rmtree(self.artifact_dir, ignore_errors=True)
            except Exception:
                pass

def build_model(X_train, y_train):
    """
    Factory function to return a fresh, unfitted AutoGluonWrapper.
    """
    return AutoGluonWrapper(label='Class')