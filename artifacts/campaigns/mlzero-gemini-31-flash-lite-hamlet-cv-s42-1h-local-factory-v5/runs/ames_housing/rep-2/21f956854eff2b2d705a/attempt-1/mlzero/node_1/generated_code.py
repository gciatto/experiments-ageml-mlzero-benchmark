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
        self.path = os.path.join(os.getcwd(), f"ag_model_{uuid.uuid4().hex}")

    def fit(self, X, y):
        # Combine X and y into a single DataFrame for AutoGluon
        train_data = X.copy()
        train_data[self.label_column] = y
        
        # Drop 'Id' if present as per instructions
        if 'Id' in train_data.columns:
            train_data = train_data.drop(columns=['Id'])

        # Initialize predictor
        # Using 'medium_quality' to ensure we stay within the shared time budget
        # for 10 folds.
        self.predictor = MultiModalPredictor(
            label=self.label_column,
            path=self.path,
            problem_type="regression",
            eval_metric="rmse"
        )
        
        # Fit with a time limit that allows for 10 folds to complete
        # Total budget is ~2700s, so ~240s per fold is safe.
        self.predictor.fit(
            train_data=train_data,
            time_limit=220,
            presets="medium_quality"
        )

    def predict(self, X):
        # Prepare input data
        test_data = X.copy()
        if 'Id' in test_data.columns:
            test_data = test_data.drop(columns=['Id'])
            
        return self.predictor.predict(test_data)

    def __del__(self):
        # Cleanup artifact directory to save space
        if os.path.exists(self.path):
            try:
                shutil.rmtree(self.path)
            except:
                pass

def build_model(X_train, y_train):
    """
    Factory function to return a fresh, unfitted model wrapper.
    """
    # The target column is SalePrice
    return AutoGluonWrapper(label_column="SalePrice")