import os
import shutil
import tempfile
import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
import random

class AutoGluonWrapper:
    """
    A wrapper for AutoGluon TabularPredictor that handles the lifecycle
    of the model within the constraints of the cross-validation harness.
    """
    def __init__(self, seed=42):
        self.seed = seed
        self.model_path = tempfile.mkdtemp(prefix="ag_model_")
        self.predictor = None

    def _set_seeds(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        # Note: AutoGluon handles internal framework seeds via its own config,
        # but we set these to ensure consistent data handling.

    def fit(self, X, y):
        self._set_seeds()
        
        # Combine X and y for AutoGluon
        train_data = X.copy()
        train_data['class'] = y
        
        # Initialize predictor
        # We limit time_limit to ensure all 10 folds fit within the total budget.
        # Given 3549s total, we allocate ~300s per fold.
        self.predictor = TabularPredictor(
            label='class',
            path=self.model_path,
            eval_metric='balanced_accuracy'
        )
        
        self.predictor.fit(
            train_data,
            time_limit=300,
            presets='best_quality',
            ag_args_fit={'random_seed': self.seed},
            verbosity=0
        )
        return self

    def predict(self, X):
        if self.predictor is None:
            raise ValueError("Model has not been fitted yet.")
        return self.predictor.predict(X)

    def __del__(self):
        # Clean up the temporary directory used by AutoGluon
        if os.path.exists(self.model_path):
            try:
                shutil.rmtree(self.model_path)
            except Exception:
                pass

def build_model(X_train, y_train):
    """
    Factory function to return a fresh, unfitted AutoGluonWrapper.
    """
    return AutoGluonWrapper(seed=42)