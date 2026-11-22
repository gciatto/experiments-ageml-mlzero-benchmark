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
        self.temp_dir = tempfile.mkdtemp(prefix="ag_fold_")

    def fit(self, X, y):
        # Combine X and y for AutoGluon
        train_data = X.copy()
        train_data[self.label] = y
        
        # Ensure a fresh directory for each fit
        path = os.path.join(self.temp_dir, str(uuid.uuid4()))
        
        # Initialize and fit
        # Using 'extreme' preset as requested. 
        # We rely on AutoGluon's internal validation splitting.
        self.predictor = TabularPredictor(
            label=self.label,
            path=path,
            eval_metric='rmse'
        ).fit(
            train_data,
            presets='extreme',
            verbosity=0
        )
        return self

    def predict(self, X):
        if self.predictor is None:
            raise ValueError("Model has not been fitted yet.")
        return self.predictor.predict(X)

    def __del__(self):
        # Cleanup temporary directory after the object is destroyed
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

def build_model(X_train, y_train):
    """
    Factory function to return a fresh, unfitted model wrapper.
    The harness will call fit() on this object.
    """
    # The target column is SalePrice as per the task description
    return AutoGluonRegressorWrapper(label='SalePrice')