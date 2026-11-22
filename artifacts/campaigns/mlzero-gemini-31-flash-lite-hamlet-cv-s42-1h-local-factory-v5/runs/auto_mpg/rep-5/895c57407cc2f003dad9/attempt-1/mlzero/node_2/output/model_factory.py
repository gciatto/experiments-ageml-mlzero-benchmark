import os
import uuid
import shutil
import pandas as pd
from autogluon.multimodal import MultiModalPredictor

class AutoGluonRegressionWrapper:
    """
    A wrapper for AutoGluon MultiModalPredictor to handle regression tasks
    within the required factory interface.
    """
    def __init__(self, label_column="class"):
        self.label_column = label_column
        self.predictor = None
        # Create a unique directory for this specific fold instance
        self.path = os.path.join(os.getcwd(), f"ag_model_{uuid.uuid4().hex}")

    def fit(self, X, y):
        """
        Fits the MultiModalPredictor on the provided training fold.
        X and y are combined into a single DataFrame for AutoGluon.
        """
        # Combine features and target
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
        
        # Fit with a time limit that allows for 10 folds within the remaining budget.
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
        Cleanup the artifact directory to save disk space.
        """
        if self.path and os.path.exists(self.path):
            try:
                shutil.rmtree(self.path)
            except Exception:
                pass

def build_model(X_train, y_train):
    """
    Factory function to return a fresh, unfitted model wrapper.
    The X_train and y_train arguments are provided by the harness but 
    are not used here as the model must remain unfitted.
    """
    return AutoGluonRegressionWrapper(label_column="class")