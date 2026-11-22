import os
import shutil
import tempfile
import uuid
import pandas as pd
from autogluon.tabular import TabularPredictor

class AutoGluonRegressorWrapper:
    """
    A wrapper for AutoGluon TabularPredictor designed for the auto_mpg regression task.
    Ensures resource constraints are met by disabling parallel processing and 
    limiting the time per fold.
    """
    def __init__(self, label='class'):
        self.label = label
        self.predictor = None
        # Create a unique directory for this fold's artifacts
        self.path = os.path.join(tempfile.gettempdir(), f"ag_model_{uuid.uuid4().hex}")

    def fit(self, X, y):
        """
        Fits the AutoGluon predictor on the provided training fold.
        """
        train_data = X.copy()
        train_data[self.label] = y
        
        # To ensure all 10 folds finish within the remaining 54s total,
        # we allocate a very conservative time limit per fold.
        # 54s total / 10 folds = 5.4s. We use 4s to account for overhead.
        self.predictor = TabularPredictor(
            label=self.label,
            problem_type='regression',
            eval_metric='rmse',
            path=self.path
        ).fit(
            train_data,
            presets='medium',
            time_limit=4,
            num_cpus=1,
            num_gpus=0,
            ag_args_fit={'num_gpus': 0},
            verbosity=0
        )
        return self

    def predict(self, X):
        """
        Predicts using the fitted AutoGluon predictor.
        """
        if self.predictor is None:
            raise ValueError("Model has not been fitted yet.")
        return self.predictor.predict(X)

    def __del__(self):
        """
        Cleanup temporary artifacts after the fold evaluation is complete.
        """
        if self.path and os.path.exists(self.path):
            try:
                shutil.rmtree(self.path, ignore_errors=True)
            except Exception:
                pass

def build_model(X_train, y_train):
    """
    Factory function to return a fresh, unfitted model wrapper.
    """
    return AutoGluonRegressorWrapper(label='class')