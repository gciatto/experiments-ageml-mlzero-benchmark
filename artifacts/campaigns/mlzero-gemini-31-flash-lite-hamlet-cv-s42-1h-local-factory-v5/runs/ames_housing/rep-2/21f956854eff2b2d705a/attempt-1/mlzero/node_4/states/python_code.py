import os
import uuid
import shutil
import random
import numpy as np
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
        # Create a unique directory for this fold's artifacts
        self.path = os.path.join(os.getcwd(), f"ag_model_{uuid.uuid4().hex}")

    def _set_seeds(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)

    def fit(self, X, y):
        # Ensure reproducibility
        self._set_seeds(42)
        
        # Combine X and y into a single DataFrame for AutoGluon
        train_data = X.copy()
        train_data[self.label_column] = y
        
        # Drop 'Id' if present as per instructions
        if 'Id' in train_data.columns:
            train_data = train_data.drop(columns=['Id'])

        # Initialize predictor
        # Using 'medium_quality' to balance performance and time constraints.
        # We limit the time per fold to ensure all 10 folds finish within the 1064s budget.
        # 1064s total / 10 folds = ~106s per fold. We set a conservative 80s limit.
        self.predictor = MultiModalPredictor(
            label=self.label_column,
            path=self.path,
            problem_type="regression",
            eval_metric="rmse"
        )
        
        self.predictor.fit(
            train_data=train_data,
            time_limit=80,
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
                shutil.rmtree(self.path, ignore_errors=True)
            except:
                pass

def build_model(X_train, y_train):
    """
    Factory function to return a fresh, unfitted model wrapper.
    """
    # The target column is SalePrice
    return AutoGluonWrapper(label_column="SalePrice")