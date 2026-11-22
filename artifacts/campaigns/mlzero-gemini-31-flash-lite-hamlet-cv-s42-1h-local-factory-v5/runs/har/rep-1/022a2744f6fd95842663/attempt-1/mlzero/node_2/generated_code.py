import os
import uuid
import shutil
import pandas as pd
from autogluon.multimodal import MultiModalPredictor

class AutoGluonWrapper:
    """
    A wrapper for AutoGluon MultiModalPredictor that handles the lifecycle
    of the predictor, including directory management for cross-validation.
    """
    def __init__(self):
        self.predictor = None
        self.path = os.path.join(os.getcwd(), f"autogluon_model_{uuid.uuid4().hex}")

    def fit(self, X, y):
        """
        Fits the MultiModalPredictor on the provided training fold.
        X: DataFrame of features
        y: Series of labels
        """
        # Combine X and y for AutoGluon
        train_data = X.copy()
        train_data['Class'] = y
        
        # Initialize predictor
        # Using 'medium_quality' to ensure we stay within the total time budget
        # for 10 folds.
        self.predictor = MultiModalPredictor(
            label='Class',
            path=self.path,
            problem_type='multiclass'
        )
        
        # Fit with a time limit to ensure all 10 folds complete within the 
        # remaining shared deadline. 120 seconds per fold is a safe heuristic.
        self.predictor.fit(
            train_data=train_data,
            time_limit=120,
            presets='medium_quality'
        )
        return self

    def predict(self, X):
        """
        Predicts class labels for the provided data.
        """
        if self.predictor is None:
            raise ValueError("Model has not been fitted yet.")
        return self.predictor.predict(X)

    def __del__(self):
        """
        Cleanup the artifact directory after the model is no longer needed.
        """
        if os.path.exists(self.path):
            try:
                shutil.rmtree(self.path)
            except Exception:
                pass

def build_model(X_train, y_train):
    """
    Factory function to return a fresh, unfitted AutoGluonWrapper.
    """
    return AutoGluonWrapper()