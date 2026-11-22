import os
import shutil
import tempfile
import random
import numpy as np
import torch
from autogluon.tabular import TabularPredictor
from sklearn.base import BaseEstimator, RegressorMixin

class AutoGluonRegressorWrapper(BaseEstimator, RegressorMixin):
    """
    A wrapper for AutoGluon TabularPredictor that adheres to the 
    scikit-learn estimator interface for use in cross-validation.
    """
    def __init__(self, seed=42):
        self.seed = seed
        self.predictor = None
        self.temp_dir = None
        self.target_col = "class"

    def _set_seeds(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

    def fit(self, X, y):
        # Create a fresh directory for this fold's artifacts
        self.temp_dir = tempfile.mkdtemp(prefix="ag_fold_")
        
        # Prepare data
        df = X.copy()
        df[self.target_col] = y
        
        self._set_seeds()
        
        # Initialize and fit predictor
        # Using 'medium_quality' to balance performance and time constraints.
        # time_limit is set to 250s to ensure 10 folds fit within the ~3000s budget.
        self.predictor = TabularPredictor(
            label=self.target_col,
            path=self.temp_dir,
            problem_type='regression',
            verbosity=0
        )
        
        self.predictor.fit(
            df,
            time_limit=250,
            presets='medium_quality',
            fit_weighted_ensemble=True
        )
        return self

    def predict(self, X):
        if self.predictor is None:
            raise ValueError("Model must be fitted before calling predict.")
        return self.predictor.predict(X)

    def __del__(self):
        # Cleanup artifacts after use to prevent disk bloat
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir, ignore_errors=True)
            except:
                pass

def build_model(X_train, y_train):
    """
    Returns a fresh, unfitted AutoGluonRegressorWrapper instance.
    The harness will call fit() on this object using the provided training fold.
    """
    return AutoGluonRegressorWrapper(seed=42)