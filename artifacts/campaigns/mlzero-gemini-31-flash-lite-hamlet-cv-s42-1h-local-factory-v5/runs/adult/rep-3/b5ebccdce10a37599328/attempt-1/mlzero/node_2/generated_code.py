import os
import uuid
import shutil
import pandas as pd
from autogluon.multimodal import MultiModalPredictor

class AutoGluonWrapper:
    """
    A wrapper for AutoGluon MultiModalPredictor that adheres to the 
    fit/predict interface required by the benchmark.
    """
    def __init__(self):
        self.model_path = os.path.join(os.getcwd(), f"autogluon_model_{uuid.uuid4().hex}")
        self.predictor = None
        self.label_column = "class"

    def fit(self, X, y):
        """
        Fits the model using the provided training data.
        X and y are combined into a single DataFrame for AutoMM.
        """
        # Combine features and target
        train_df = X.copy()
        train_df[self.label_column] = y

        # Initialize predictor
        # We use a conservative time_limit to ensure all 10 folds finish within the 3530s budget.
        # 3530s / 10 folds = 353s per fold. We allocate 280s to allow for overhead.
        self.predictor = MultiModalPredictor(
            label=self.label_column,
            path=self.model_path,
            problem_type="binary"
        )

        self.predictor.fit(
            train_data=train_df,
            time_limit=280,
            hyperparameters={
                "env.num_gpus": 0,  # Ensure CPU usage if no GPU is available
                "optim.max_epochs": 5,
                "model.names": ["tab_transformer"] # Use tabular-specific model for efficiency
            }
        )
        return self

    def predict(self, X):
        """
        Predicts the target variable for the provided input data.
        """
        if self.predictor is None:
            raise ValueError("Model has not been fitted yet.")
        
        predictions = self.predictor.predict(X)
        return predictions

    def __del__(self):
        """
        Cleanup the artifact directory after the object is destroyed.
        """
        if os.path.exists(self.model_path):
            try:
                shutil.rmtree(self.model_path)
            except Exception:
                pass

def build_model(X_train, y_train):
    """
    Factory function to return a fresh, unfitted AutoGluonWrapper.
    """
    return AutoGluonWrapper()