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
    def __init__(self, label):
        self.label = label
        self.predictor = None
        # Create a unique directory for this fold's artifacts
        self.path = os.path.join(tempfile.gettempdir(), f"ag_model_{uuid.uuid4().hex}")

    def fit(self, X, y):
        # Combine X and y into a single DataFrame for AutoGluon
        train_data = X.copy()
        train_data[self.label] = y
        
        # Initialize and fit the predictor
        # Using 'extreme' preset as requested, with a focus on balanced_accuracy
        self.predictor = TabularPredictor(
            label=self.label,
            eval_metric='balanced_accuracy',
            path=self.path
        ).fit(
            train_data,
            presets='extreme',
            # Ensure we don't exceed the total time budget by limiting individual fits
            # The total time is shared across 10 folds.
            time_limit=300, 
            verbosity=1
        )
        return self

    def predict(self, X):
        if self.predictor is None:
            raise ValueError("Model has not been fitted yet.")
        return self.predictor.predict(X)

    def __del__(self):
        # Clean up the artifact directory after the object is destroyed
        if self.path and os.path.exists(self.path):
            try:
                shutil.rmtree(self.path)
            except Exception:
                pass

def build_model(X_train, y_train):
    """
    Factory function to return a fresh, unfitted AutoGluonWrapper.
    The target column is identified from the training data provided.
    """
    # The target column is 'class' based on the task description
    label = 'class'
    return AutoGluonWrapper(label=label)