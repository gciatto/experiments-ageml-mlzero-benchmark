import os
import uuid
import shutil
import pandas as pd
from autogluon.tabular import TabularPredictor

# Ensure environment stability for AutoGluon in constrained environments
os.environ["RAY_DISABLE_MEMORY_MONITOR"] = "1"

class AutoGluonWrapper:
    """
    A wrapper for AutoGluon TabularPredictor that ensures a fresh environment
    for each fold and adheres to the fit/predict interface.
    """
    def __init__(self, label='class'):
        self.label = label
        self.predictor = None
        # Create a unique directory for this specific fold's artifacts
        self.path = os.path.join(os.getcwd(), f"ag_model_{uuid.uuid4().hex}")

    def fit(self, X, y):
        # Combine X and y into a single DataFrame for AutoGluon
        train_data = X.copy()
        train_data[self.label] = y
        
        # Initialize and fit the predictor
        # We use a conservative time_limit to ensure all 10 folds finish within the 157s budget.
        # num_cpus=1 and num_gpus=0 are set to prevent resource contention/segfaults.
        self.predictor = TabularPredictor(
            label=self.label,
            path=self.path,
            eval_metric='balanced_accuracy'
        ).fit(
            train_data,
            presets='best_quality',
            time_limit=12,  # Budgeted time per fold to stay under 157s total
            num_cpus=1,
            num_gpus=0,
            verbosity=0,
            ag_args_fit={'num_gpus': 0}
        )
        return self

    def predict(self, X):
        if self.predictor is None:
            raise ValueError("Model has not been fitted yet.")
        return self.predictor.predict(X)

    def __del__(self):
        # Cleanup artifacts after use to save disk space
        if self.path and os.path.exists(self.path):
            try:
                shutil.rmtree(self.path, ignore_errors=True)
            except Exception:
                pass

def build_model(X_train, y_train):
    """
    Factory function to return a fresh, unfitted AutoGluonWrapper.
    The label is 'class' based on the provided dataset description.
    """
    return AutoGluonWrapper(label='class')