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
        X and y are combined into a single DataFrame for AutoGluon.
        """
        # Ensure we don't use 'Id' as a feature
        train_df = X.copy()
        if 'Id' in train_df.columns:
            train_df = train_df.drop(columns=['Id'])
        
        train_df[self.label_column] = y

        # Initialize predictor
        # Using 'medium_quality' to ensure we stay within the shared time budget
        # across 10 folds.
        self.predictor = MultiModalPredictor(
            label=self.label_column,
            problem_type="regression",
            eval_metric="rmse",
            path=self.path
        )

        # Fit with a time limit that respects the shared budget.
        # Total budget is ~2725s for 10 folds, so ~250s per fold is safe.
        self.predictor.fit(
            train_data=train_df,
            time_limit=240,
            presets="medium_quality"
        )

    def predict(self, X):
        """
        Predicts using the fitted AutoGluon model.
        """
        if self.predictor is None:
            raise ValueError("Model has not been fitted yet.")
        
        test_df = X.copy()
        if 'Id' in test_df.columns:
            test_df = test_df.drop(columns=['Id'])
            
        return self.predictor.predict(test_df)

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
    # The target column is SalePrice as per the task description.
    return AutoGluonWrapper(label_column="SalePrice")