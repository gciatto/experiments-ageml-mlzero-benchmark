import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.compose import ColumnTransformer

class AutoMPGRegressor(BaseEstimator, RegressorMixin):
    """
    A robust regression wrapper for the auto_mpg dataset.
    Uses a pipeline with feature interaction, scaling, and gradient boosting.
    """
    def __init__(self, random_state=42):
        self.random_state = random_state
        # Define the pipeline:
        # 1. Polynomial features to capture non-linear interactions (e.g., displacement * weight)
        # 2. Standard scaling for numerical stability
        # 3. HistGradientBoostingRegressor for efficient, high-performance regression
        self.model = Pipeline([
            ('poly', PolynomialFeatures(degree=2, include_bias=False)),
            ('scaler', StandardScaler()),
            ('regressor', HistGradientBoostingRegressor(
                random_state=self.random_state,
                max_iter=200,
                learning_rate=0.05,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10
            ))
        ])

    def fit(self, X, y):
        """
        Fits the pipeline on the provided training data.
        """
        self.model.fit(X, y)
        return self

    def predict(self, X):
        """
        Predicts using the fitted pipeline.
        """
        return self.model.predict(X)

def build_model(X_train, y_train):
    """
    Factory function to return a fresh, unfitted model instance.
    
    Parameters:
    X_train: Training features (pandas DataFrame or numpy array)
    y_train: Training targets (pandas Series or numpy array)
    
    Returns:
    An unfitted instance of AutoMPGRegressor.
    """
    # The model is initialized with a fixed seed for reproducibility.
    # All preprocessing is encapsulated within the pipeline.
    return AutoMPGRegressor(random_state=42)