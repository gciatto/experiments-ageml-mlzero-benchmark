import os
import uuid
import pandas as pd
from autogluon.tabular import TabularPredictor

class AutoGluonModel:
    """
    A wrapper for AutoGluon TabularPredictor designed for cross-validation.
    Each instance manages its own unique artifact directory to prevent 
    cross-fold contamination.
    """
    def __init__(self, label='Class'):
        self.label = label
        self.predictor = None
        # Unique path per instance to ensure isolation during parallel or sequential folds
        self.path = os.path.join(os.getcwd(), f"ag_model_{uuid.uuid4().hex}")

    def fit(self, X, y):
        """
        Fits the AutoGluon predictor on the provided training fold.
        Preprocessing is handled automatically by AutoGluon.
        """
        train_data = X.copy()
        train_data[self.label] = y
        
        # We use 'extreme' preset for high-quality performance.
        # time_limit is set conservatively to ensure all 10 folds complete 
        # within the shared 176s budget.
        self.predictor = TabularPredictor(
            label=self.label,
            path=self.path,
            eval_metric='balanced_accuracy'
        ).fit(
            train_data=train_data,
            presets='extreme',
            time_limit=15,  # 15s per fold * 10 folds = 150s < 176s budget
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
    Factory function required by the evaluation harness.
    Returns a fresh, unfitted AutoGluonModel instance.
    """
    return AutoGluonModel(label='Class')