import os
import shutil
import uuid
import pandas as pd
from autogluon.tabular import TabularPredictor

class AutoGluonWrapper:
    """
    A wrapper for AutoGluon TabularPredictor that handles the lifecycle
    of a model within a cross-validation fold.
    """
    def __init__(self, label='SalePrice'):
        self.label = label
        self.predictor = None
        # Create a unique directory for this fold's artifacts
        self.path = os.path.join(os.getcwd(), f"ag_model_{uuid.uuid4().hex}")

    def fit(self, X, y):
        """
        Fits the AutoGluon predictor.
        X and y are provided as a training fold.
        """
        # Combine X and y for AutoGluon's fit method
        train_data = X.copy()
        train_data[self.label] = y

        # Initialize and fit the predictor.
        # We use 'best_quality' as it provides a strong balance of performance 
        # and training time. Given the 1090s total budget for 10 folds, 
        # we allocate ~90s per fold to ensure completion.
        self.predictor = TabularPredictor(
            label=self.label,
            path=self.path,
            eval_metric='rmse'
        ).fit(
            train_data,
            presets='best_quality',
            time_limit=90,
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
        Cleanup artifacts after the model is no longer needed.
        """
        if self.path and os.path.exists(self.path):
            try:
                shutil.rmtree(self.path, ignore_errors=True)
            except Exception:
                pass

def build_model(X_train, y_train):
    """
    Factory function to return a fresh, unfitted AutoGluonWrapper.
    """
    # The label is fixed as 'SalePrice' per the task description.
    # We ensure the model is initialized with a clean state.
    return AutoGluonWrapper(label='SalePrice')