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
    def __init__(self, label='SalePrice'):
        self.label = label
        self.predictor = None
        # Create a unique directory for this specific fold instance
        self.temp_dir = tempfile.mkdtemp(prefix="ag_fold_")

    def fit(self, X, y):
        # Combine X and y for AutoGluon
        train_data = X.copy()
        train_data[self.label] = y
        
        # Define a unique path for this specific fit
        path = os.path.join(self.temp_dir, str(uuid.uuid4()))
        
        # Initialize and fit
        # Using 'best' preset for a balance of performance and time constraints.
        # We rely on AutoGluon's internal validation splitting.
        # verbosity=0 keeps the logs clean.
        self.predictor = TabularPredictor(
            label=self.label,
            path=path,
            eval_metric='rmse'
        ).fit(
            train_data,
            presets='best_quality',
            time_limit=15,  # Bound each fit to ensure 10 folds finish within the deadline
            verbosity=0
        )
        return self

    def predict(self, X):
        if self.predictor is None:
            raise ValueError("Model has not been fitted yet.")
        return self.predictor.predict(X)

    def __del__(self):
        # Cleanup temporary directory after the object is destroyed
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir, ignore_errors=True)
            except:
                pass

def build_model(X_train, y_train):
    """
    Factory function to return a fresh, unfitted model wrapper.
    The harness will call fit() on this object.
    """
    # The target column is SalePrice as per the task description
    return AutoGluonRegressorWrapper(label='SalePrice')