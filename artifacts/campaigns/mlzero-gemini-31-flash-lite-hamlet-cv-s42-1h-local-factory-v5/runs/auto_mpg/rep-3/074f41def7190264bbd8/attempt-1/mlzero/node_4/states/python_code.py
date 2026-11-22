import os
import shutil
import tempfile
import uuid
import pandas as pd
from autogluon.tabular import TabularPredictor

class AutoGluonRegressorWrapper:
    """
    A wrapper for AutoGluon TabularPredictor that handles the lifecycle
    of a model within a cross-validation fold.
    """
    def __init__(self, label='class'):
        self.label = label
        self.predictor = None
        # Create a unique directory for this fold's artifacts
        self.temp_dir = tempfile.mkdtemp(prefix=f"ag_fold_{uuid.uuid4().hex[:8]}_")

    def fit(self, X, y):
        """
        Fits the AutoGluon predictor. 
        We use 'best_quality' to balance performance and time constraints.
        """
        train_data = X.copy()
        train_data[self.label] = y
        
        # Initialize predictor
        # We use a time_limit to ensure the 10-fold process completes within the 466s budget.
        # 466s total / 10 folds = ~46s per fold. We set a safe limit of 35s.
        self.predictor = TabularPredictor(
            label=self.label,
            path=self.temp_dir,
            eval_metric='rmse'
        ).fit(
            train_data,
            presets='best_quality',
            time_limit=35,
            verbosity=0
        )

    def predict(self, X):
        """
        Predicts using the fitted AutoGluon predictor.
        """
        if self.predictor is None:
            raise ValueError("Model has not been fitted yet.")
        return self.predictor.predict(X)

    def __del__(self):
        """
        Clean up the temporary directory after the fold is processed.
        """
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir, ignore_errors=True)
            except Exception:
                pass

def build_model(X_train, y_train):
    """
    Factory function to return a fresh, unfitted model wrapper.
    """
    return AutoGluonRegressorWrapper(label='class')