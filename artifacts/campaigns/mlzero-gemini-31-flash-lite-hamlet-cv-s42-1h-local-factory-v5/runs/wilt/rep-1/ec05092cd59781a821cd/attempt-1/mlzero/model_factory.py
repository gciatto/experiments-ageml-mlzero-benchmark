import os
import uuid
import shutil
import pandas as pd
from autogluon.tabular import TabularPredictor

class AutoGluonWrapper:
    """
    A wrapper for AutoGluon TabularPredictor that ensures a fresh environment
    for each fold and adheres to the fit/predict interface.
    """
    def __init__(self, label='class'):
        self.label = label
        self.predictor = None
        # Create a unique directory for this specific model instance
        self.path = os.path.join(os.getcwd(), f"ag_model_{uuid.uuid4().hex}")

    def fit(self, X, y):
        """
        Fits the AutoGluon predictor. 
        X and y are combined into a single DataFrame for AutoGluon.
        """
        # Combine X and y for AutoGluon
        train_data = X.copy()
        train_data[self.label] = y
        
        # Initialize and fit
        # Using 'best_quality' preset for high performance.
        # We set a strict time_limit to ensure all 10 folds complete within the 
        # remaining shared deadline (233s total / 10 folds = ~20s per fold).
        # We use a conservative 18s to allow for overhead.
        self.predictor = TabularPredictor(
            label=self.label,
            path=self.path,
            eval_metric='balanced_accuracy'
        ).fit(
            train_data,
            presets='best_quality',
            time_limit=18,
            verbosity=0
        )
        return self

    def predict(self, X):
        """
        Predicts using the fitted AutoGluon model.
        """
        if self.predictor is None:
            raise ValueError("Model must be fitted before calling predict.")
        return self.predictor.predict(X)

    def __del__(self):
        """
        Cleanup the artifact directory after the model is no longer needed.
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
    # The label is 'class' based on the provided dataset description
    return AutoGluonWrapper(label='class')