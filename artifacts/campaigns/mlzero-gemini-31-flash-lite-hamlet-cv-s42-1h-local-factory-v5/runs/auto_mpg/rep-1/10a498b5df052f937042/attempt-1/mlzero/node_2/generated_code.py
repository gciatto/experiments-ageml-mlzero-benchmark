import os
import uuid
import shutil
import pandas as pd
from autogluon.multimodal import MultiModalPredictor

class AutoGluonWrapper:
    """
    A wrapper for AutoGluon MultiModalPredictor that adheres to the 
    fit/predict interface required by the evaluation harness.
    """
    def __init__(self, label_column="class"):
        self.label_column = label_column
        self.predictor = None
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
        # We use a conservative time_limit to ensure all 10 folds finish 
        # within the shared deadline.
        self.predictor = MultiModalPredictor(
            label=self.label_column,
            problem_type="regression",
            path=self.path,
            eval_metric="rmse"
        )
        
        # Fit with a time limit that allows for 10 folds within ~3000s
        # 240s per fold is safe given the total budget.
        self.predictor.fit(
            train_data=train_df,
            time_limit=240,
            presets="medium_quality"
        )
        return self

    def predict(self, X):
        """
        Predicts using the fitted MultiModalPredictor.
        """
        if self.predictor is None:
            raise ValueError("Model has not been fitted yet.")
        return self.predictor.predict(X)

    def __del__(self):
        """
        Cleanup the artifact directory after use to save disk space.
        """
        if self.path and os.path.exists(self.path):
            try:
                shutil.rmtree(self.path)
            except Exception:
                pass

def build_model(X_train, y_train):
    """
    Factory function to return a fresh, unfitted AutoGluonWrapper.
    """
    # The target column is 'class' as per the task description.
    return AutoGluonWrapper(label_column="class")