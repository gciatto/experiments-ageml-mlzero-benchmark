import os
import shutil
import tempfile
import uuid
import pandas as pd
from autogluon.tabular import TabularPredictor

class AutoGluonWrapper:
    """
    A wrapper for AutoGluon TabularPredictor that handles the lifecycle
    of a model within a cross-validation fold.
    """
    def __init__(self, label='Class'):
        self.label = label
        self.predictor = None
        self.temp_dir = tempfile.mkdtemp(prefix="ag_fold_")

    def fit(self, X, y):
        # Combine X and y for AutoGluon
        train_data = X.copy()
        train_data[self.label] = y
        
        # Define a path for this specific fold's model
        fold_path = os.path.join(self.temp_dir, str(uuid.uuid4()))
        
        # Initialize and fit the predictor
        # Using 'extreme' preset as requested, with a focus on balanced_accuracy
        self.predictor = TabularPredictor(
            label=self.label,
            eval_metric='balanced_accuracy',
            path=fold_path
        )
        
        # Fit with a reasonable time limit to ensure all 10 folds finish
        # within the remaining shared deadline.
        self.predictor.fit(
            train_data,
            presets='extreme',
            time_limit=240,  # 4 minutes per fold
            verbosity=0
        )

    def predict(self, X):
        if self.predictor is None:
            raise ValueError("Model has not been fitted yet.")
        return self.predictor.predict(X)

    def __del__(self):
        # Clean up the temporary directory after the object is destroyed
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

def build_model(X_train, y_train):
    """
    Factory function to return a fresh, unfitted AutoGluonWrapper.
    """
    return AutoGluonWrapper(label='Class')