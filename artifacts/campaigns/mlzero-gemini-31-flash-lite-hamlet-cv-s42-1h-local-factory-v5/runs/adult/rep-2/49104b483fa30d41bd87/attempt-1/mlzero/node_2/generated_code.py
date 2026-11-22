import os
import uuid
import shutil
import pandas as pd
from autogluon.multimodal import MultiModalPredictor

class AutoGluonWrapper:
    """
    A wrapper for AutoGluon MultiModalPredictor that handles the lifecycle
    of a model within a cross-validation fold.
    """
    def __init__(self, label_column='class'):
        self.label_column = label_column
        self.predictor = None
        # Create a unique path for this specific model instance to avoid collisions
        self.path = os.path.join(os.getcwd(), f"autogluon_model_{uuid.uuid4().hex}")

    def fit(self, X, y):
        """
        Fits the model using the provided training data.
        X and y are combined into a single DataFrame for AutoGluon.
        """
        train_data = X.copy()
        train_data[self.label_column] = y
        
        # Initialize predictor
        # We use 'medium_quality' to ensure we stay within the time constraints
        # for 10 folds.
        self.predictor = MultiModalPredictor(
            label=self.label_column,
            path=self.path,
            eval_metric='balanced_accuracy'
        )
        
        # Fit with constraints to ensure completion within the shared deadline
        # We limit max_epochs and use a preset that balances speed and performance.
        self.predictor.fit(
            train_data=train_data,
            presets='medium_quality',
            hyperparameters={
                "optim.max_epochs": 5,
                "env.num_gpus": 0  # Ensure compatibility if GPU is not available
            }
        )
        return self

    def predict(self, X):
        """
        Predicts the target variable for the provided input data.
        """
        if self.predictor is None:
            raise ValueError("Model has not been fitted yet.")
        return self.predictor.predict(X)

    def __del__(self):
        """
        Cleanup the artifact directory after the model is no longer needed.
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
    # The target column is 'class' as per the dataset description.
    return AutoGluonWrapper(label_column='class')