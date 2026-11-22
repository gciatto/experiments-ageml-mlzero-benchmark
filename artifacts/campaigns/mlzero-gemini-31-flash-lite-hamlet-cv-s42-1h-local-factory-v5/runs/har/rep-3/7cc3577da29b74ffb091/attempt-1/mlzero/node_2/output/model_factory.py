import os
import shutil
import uuid
import pandas as pd
from autogluon.multimodal import MultiModalPredictor

class AutoGluonWrapper:
    """
    A wrapper for AutoGluon MultiModalPredictor that handles the lifecycle
    of a model within a cross-validation fold.
    """
    def __init__(self):
        self.predictor = None
        self.path = os.path.join(os.getcwd(), f"autogluon_model_{uuid.uuid4().hex}")
        self.label_col = "Class"

    def fit(self, X, y):
        """
        Fits the model on the provided training fold.
        X: DataFrame of features
        y: Series of labels
        """
        # Combine features and labels for AutoGluon
        train_data = X.copy()
        train_data[self.label_col] = y

        # Initialize predictor
        # Using 'medium_quality' to ensure we stay within the strict time budget
        # for 10 folds.
        self.predictor = MultiModalPredictor(
            label=self.label_col,
            path=self.path,
            eval_metric="balanced_accuracy"
        )

        # Fit with a time limit to ensure all 10 folds complete within the 929s budget.
        # 929s / 10 folds = ~92s per fold. We allocate 70s to be safe.
        self.predictor.fit(
            train_data=train_data,
            time_limit=70,
            presets="medium_quality"
        )

    def predict(self, X):
        """
        Predicts class labels for the provided features.
        """
        if self.predictor is None:
            raise ValueError("Model has not been fitted yet.")
        return self.predictor.predict(X)

    def __del__(self):
        """
        Cleanup the artifact directory after the model is no longer needed.
        """
        if self.path and os.path.exists(self.path):
            try:
                shutil.rmtree(self.path)
            except Exception:
                pass

def build_model(X_train, y_train):
    """
    Factory function to return a fresh, unfitted model wrapper.
    """
    return AutoGluonWrapper()