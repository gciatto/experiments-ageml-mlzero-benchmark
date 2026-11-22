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
    def __init__(self):
        self.label_col = "class"
        self.model_path = os.path.join(os.getcwd(), f"ag_model_{uuid.uuid4().hex}")
        self.predictor = None

    def fit(self, X, y):
        # Combine X and y into a single DataFrame for AutoGluon
        train_df = X.copy()
        train_df[self.label_col] = y
        
        # Initialize predictor
        # We use a conservative time limit and epoch count to ensure 
        # all 10 folds complete within the shared deadline.
        self.predictor = MultiModalPredictor(
            label=self.label_col,
            path=self.model_path,
            problem_type="binary" if len(y.unique()) == 2 else "multiclass"
        )
        
        # Fit with constrained resources
        self.predictor.fit(
            train_data=train_df,
            hyperparameters={
                "optim.max_epochs": 5,
                "env.num_gpus": 0,  # Ensure compatibility in restricted environments
                "env.num_workers": 0,
            },
            time_limit=40,  # Strict per-fold limit to stay under total deadline
        )
        return self

    def predict(self, X):
        if self.predictor is None:
            raise ValueError("Model has not been fitted yet.")
        return self.predictor.predict(X)

    def __del__(self):
        # Cleanup artifact directory to save disk space
        if os.path.exists(self.model_path):
            try:
                shutil.rmtree(self.model_path)
            except:
                pass

def build_model(X_train, y_train):
    """
    Factory function to return a fresh, unfitted AutoGluonWrapper.
    """
    return AutoGluonWrapper()