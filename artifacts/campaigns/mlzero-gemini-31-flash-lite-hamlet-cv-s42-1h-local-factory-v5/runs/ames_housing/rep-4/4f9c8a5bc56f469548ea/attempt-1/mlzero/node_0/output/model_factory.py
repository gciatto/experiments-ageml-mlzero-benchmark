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
        X and y are provided as a single training set for the current fold.
        """
        # Combine X and y into a single DataFrame for AutoGluon
        train_data = X.copy()
        train_data[self.label] = y

        # Initialize and fit the predictor
        # Using 'extreme' preset as requested for high-quality regression
        # We set a reasonable time_limit to ensure all 10 folds finish within the total budget
        # Total budget is ~3576s, so we allocate ~300s per fold to be safe.
        self.predictor = TabularPredictor(
            label=self.label,
            path=self.path,
            eval_metric='rmse'
        ).fit(
            train_data,
            presets='extreme',
            time_limit=300,
            verbosity=1
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
    # The wrapper handles the initialization of the predictor path
    # and the fitting process when .fit() is called.
    return AutoGluonWrapper(label='SalePrice')