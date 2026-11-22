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
    def __init__(self, label_column="SalePrice"):
        self.label_column = label_column
        self.predictor = None
        # Create a unique directory for this specific model instance
        self.path = os.path.join(os.getcwd(), f"autogluon_model_{uuid.uuid4().hex}")

    def fit(self, X, y):
        """
        Fits the model on the provided training data.
        """
        # Combine X and y into a single DataFrame for AutoGluon
        train_data = X.copy()
        train_data[self.label_column] = y
        
        # Drop 'Id' if present as per requirements
        if 'Id' in train_data.columns:
            train_data = train_data.drop(columns=['Id'])

        # Initialize predictor
        # We use a conservative time_limit to ensure all 10 folds finish within the total budget.
        # Given 2724s total, we allocate ~240s per fold.
        self.predictor = MultiModalPredictor(
            label=self.label_column,
            problem_type="regression",
            path=self.path,
            eval_metric="rmse"
        )
        
        self.predictor.fit(
            train_data=train_data,
            time_limit=240,
            presets="medium_quality"
        )

    def predict(self, X):
        """
        Predicts using the fitted model.
        """
        if self.predictor is None:
            raise ValueError("Model has not been fitted yet.")
        
        test_data = X.copy()
        if 'Id' in test_data.columns:
            test_data = test_data.drop(columns=['Id'])
            
        return self.predictor.predict(test_data)

    def __del__(self):
        """
        Cleanup the artifact directory after the model is no longer needed.
        """
        if os.path.exists(self.path):
            try:
                shutil.rmtree(self.path)
            except Exception:
                pass

def build_model(X_train, y_train):
    """
    Factory function to return a fresh, unfitted model wrapper.
    """
    # The target column is SalePrice as per the task description.
    return AutoGluonWrapper(label_column="SalePrice")