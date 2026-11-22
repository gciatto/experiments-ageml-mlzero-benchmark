import os
import shutil
import uuid
import pandas as pd
from autogluon.tabular import TabularPredictor, TabularDataset

class AutoGluonWrapper:
    """
    A wrapper for AutoGluon TabularPredictor that handles the lifecycle
    of a model within a cross-validation fold.
    """
    def __init__(self, label='Class'):
        self.label = label
        self.predictor = None
        # Create a unique directory for this fold's artifacts
        self.path = os.path.join(os.getcwd(), f"ag_model_{uuid.uuid4().hex}")

    def fit(self, X, y):
        """
        Fits the AutoGluon predictor.
        X: pd.DataFrame of features
        y: pd.Series of target labels
        """
        # Combine X and y for AutoGluon's fit method
        train_data = X.copy()
        train_data[self.label] = y
        
        # Initialize and fit the predictor
        # Using 'extreme' preset as requested for high-quality performance
        # time_limit is set to 300s to ensure 10 folds fit within the remaining budget
        self.predictor = TabularPredictor(
            label=self.label,
            path=self.path,
            eval_metric='balanced_accuracy'
        ).fit(
            train_data=train_data,
            presets='extreme',
            time_limit=300,
            verbosity=0
        )
        return self

    def predict(self, X):
        """
        Predicts using the fitted AutoGluon predictor.
        """
        if self.predictor is None:
            raise ValueError("Model has not been fitted yet.")
        return self.predictor.predict(X)

def build_model(X_train, y_train):
    """
    Factory function to return a fresh, unfitted AutoGluonWrapper.
    """
    return AutoGluonWrapper(label='Class')