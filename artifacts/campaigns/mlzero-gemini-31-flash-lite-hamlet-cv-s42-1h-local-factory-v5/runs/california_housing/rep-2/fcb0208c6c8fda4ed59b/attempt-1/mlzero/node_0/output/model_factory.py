import os
import shutil
import tempfile
import uuid
import pandas as pd
from autogluon.tabular import TabularPredictor

class AutoGluonWrapper:
    """
    A wrapper for AutoGluon TabularPredictor that handles fitting and prediction
    within a self-contained, temporary directory for each fold.
    """
    def __init__(self, label='median_house_value'):
        self.label = label
        self.predictor = None
        # Create a unique path for this model instance to avoid collisions
        self.path = os.path.join(tempfile.gettempdir(), f"ag_model_{uuid.uuid4().hex}")

    def fit(self, X, y):
        """
        Fits the AutoGluon predictor.
        X: pd.DataFrame of features
        y: pd.Series of target values
        """
        # Combine X and y for AutoGluon's fit method
        train_data = X.copy()
        train_data[self.label] = y

        # Initialize and fit the predictor
        # Using 'extreme' preset as requested, with a focus on regression
        # We disable verbosity to keep logs clean
        self.predictor = TabularPredictor(
            label=self.label,
            path=self.path,
            eval_metric='rmse'
        ).fit(
            train_data,
            presets='extreme',
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

    def __del__(self):
        """
        Cleanup the temporary directory used by AutoGluon.
        """
        if self.path and os.path.exists(self.path):
            try:
                shutil.rmtree(self.path)
            except Exception:
                pass

def build_model(X_train, y_train):
    """
    Factory function to return a fresh, unfitted AutoGluonWrapper.
    The actual fitting happens later when the harness calls .fit().
    """
    # The target column is fixed as 'median_house_value' per the task description
    return AutoGluonWrapper(label='median_house_value')