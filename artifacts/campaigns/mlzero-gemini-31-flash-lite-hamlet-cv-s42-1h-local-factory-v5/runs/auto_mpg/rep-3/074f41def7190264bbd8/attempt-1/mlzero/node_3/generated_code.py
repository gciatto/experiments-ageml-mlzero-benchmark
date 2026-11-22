import os
import shutil
import tempfile
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
        # Create a unique temporary directory for this fold's artifacts
        self.temp_dir = tempfile.mkdtemp(prefix="ag_fold_")

    def fit(self, X, y):
        """
        Fits the AutoGluon predictor. 
        AutoGluon handles internal validation and preprocessing automatically.
        """
        # Combine X and y for AutoGluon's expected input format
        train_data = X.copy()
        train_data[self.label] = y
        
        # Initialize and fit the predictor.
        # 'best_quality' is used for robust performance.
        # We set a time_limit to ensure the 10-fold CV completes within the 
        # shared deadline (approx 300s per fold).
        self.predictor = TabularPredictor(
            label=self.label,
            path=self.temp_dir,
            eval_metric='rmse'
        ).fit(
            train_data,
            presets='best_quality',
            time_limit=280,
            verbosity=0
        )

    def predict(self, X):
        """
        Predicts using the fitted AutoGluon model.
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
    The harness will call fit() on this object using the fold-specific data.
    """
    return AutoGluonRegressorWrapper(label='class')