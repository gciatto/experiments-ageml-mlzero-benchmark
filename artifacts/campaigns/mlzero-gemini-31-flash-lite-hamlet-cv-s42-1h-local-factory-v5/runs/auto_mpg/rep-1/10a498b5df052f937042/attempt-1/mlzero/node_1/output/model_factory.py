import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.base import BaseEstimator, RegressorMixin

class AutoMPGRegressor(BaseEstimator, RegressorMixin):
    """
    A wrapper for a regression pipeline using HistGradientBoostingRegressor.
    This model is robust to missing values and handles numerical features efficiently.
    """
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', HistGradientBoostingRegressor(
                random_state=self.random_state,
                max_iter=100,
                learning_rate=0.1,
                early_stopping=True
            ))
        ])

    def fit(self, X, y):
        """
        Fits the pipeline on the provided training data.
        """
        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        """
        Predicts using the fitted pipeline.
        """
        return self.pipeline.predict(X)

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
    # Preprocessing (StandardScaler) is encapsulated within the pipeline.
    return AutoMPGRegressor(random_state=42)