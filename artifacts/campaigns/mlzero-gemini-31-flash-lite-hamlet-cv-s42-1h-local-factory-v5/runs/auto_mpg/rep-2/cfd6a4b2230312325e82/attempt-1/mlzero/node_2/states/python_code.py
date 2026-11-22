import os
import uuid
import shutil
import pandas as pd
from autogluon.multimodal import MultiModalPredictor

class AutoGluonRegressionWrapper:
    """
    A wrapper for AutoGluon MultiModalPredictor to handle regression tasks
    within a cross-validation harness.
    """
    def __init__(self, label_column):
        self.label_column = label_column
        self.predictor = None
        # Create a unique directory for this fold's artifacts
        self.path = os.path.join(os.getcwd(), f"ag_model_{uuid.uuid4().hex}")

    def fit(self, X, y):
        """
        Fits the MultiModalPredictor on the provided training fold.
        """
        # Combine X and y into a single DataFrame for AutoGluon
        train_df = X.copy()
        train_df[self.label_column] = y
        
        # Initialize predictor
        # Using 'medium_quality' to ensure we stay within the total time budget
        # for 10 folds.
        self.predictor = MultiModalPredictor(
            label=self.label_column,
            problem_type="regression",
            path=self.path,
            eval_metric="rmse"
        )
        
        # Fit with a time limit that allows for 10 folds to complete
        # Total budget is ~3400s, so ~300s per fold is safe.
        self.predictor.fit(
            train_data=train_df,
            time_limit=280,
            presets="medium_quality"
        )

    def predict(self, X):
        """
        Predicts using the fitted MultiModalPredictor.
        """
        if self.predictor is None:
            raise ValueError("Model has not been fitted yet.")
        return self.predictor.predict(X)

    def __del__(self):
        """
        Cleanup artifacts to save disk space.
        """
        if self.path and os.path.exists(self.path):
            try:
                shutil.rmtree(self.path)
            except Exception:
                pass

def build_model(X_train, y_train):
    """
    Factory function to return a fresh, unfitted model wrapper.
    The target column is 'class' based on the task description.
    """
    return AutoGluonRegressionWrapper(label_column='class')