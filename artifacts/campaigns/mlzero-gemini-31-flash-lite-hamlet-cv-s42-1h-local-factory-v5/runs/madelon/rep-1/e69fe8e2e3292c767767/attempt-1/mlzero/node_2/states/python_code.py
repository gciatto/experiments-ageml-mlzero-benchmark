import os
import uuid
import shutil
import pandas as pd
from autogluon.multimodal import MultiModalPredictor

class AutoMMWrapper:
    """
    A wrapper for AutoGluon MultiModalPredictor that handles the lifecycle
    of a model within a cross-validation fold.
    """
    def __init__(self, label_col='Class'):
        self.label_col = label_col
        self.predictor = None
        self.model_path = os.path.join(os.getcwd(), f"automm_model_{uuid.uuid4().hex}")

    def fit(self, X, y):
        """
        Fits the MultiModalPredictor on the provided training fold.
        """
        # Combine features and target for AutoMM
        train_df = X.copy()
        train_df[self.label_col] = y

        # Initialize predictor
        # We use a conservative time_limit to ensure all 10 folds finish within the 3260s budget.
        # 3260s / 10 folds = 326s per fold. We allocate 280s to allow for overhead.
        self.predictor = MultiModalPredictor(
            label=self.label_col,
            path=self.model_path,
            eval_metric='balanced_accuracy'
        )

        self.predictor.fit(
            train_data=train_df,
            time_limit=280,
            presets='medium_quality'
        )

    def predict(self, X):
        """
        Predicts class labels for the provided validation fold.
        """
        if self.predictor is None:
            raise ValueError("Model has not been fitted yet.")
        return self.predictor.predict(X)

    def __del__(self):
        """
        Cleanup the artifact directory after the object is destroyed.
        """
        if self.model_path and os.path.exists(self.model_path):
            try:
                shutil.rmtree(self.model_path)
            except Exception:
                pass

def build_model(X_train, y_train):
    """
    Factory function to return a fresh, unfitted AutoMMWrapper instance.
    """
    return AutoMMWrapper(label_col='Class')