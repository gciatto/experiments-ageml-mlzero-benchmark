import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.compose import ColumnTransformer

class AutoMPGRegressor(BaseEstimator, RegressorMixin):
    """
    A robust regression wrapper using HistGradientBoostingRegressor.
    Includes polynomial feature expansion and scaling to capture non-linear 
    relationships in the auto_mpg dataset.
    """
    def __init__(self, random_state=42):
        self.random_state = random_state
        # Define a pipeline that handles feature scaling and non-linear interactions
        # HistGradientBoostingRegressor is fast and handles large datasets well.
        self.pipeline = Pipeline([
            ('preprocessor', ColumnTransformer([
                ('poly', PolynomialFeatures(degree=2, include_bias=False), 
                 ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model', 'origin'])
            ], remainder='passthrough')),
            ('scaler', StandardScaler()),
            ('regressor', HistGradientBoostingRegressor(
                random_state=self.random_state,
                max_iter=100,
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
        # Ensure input is treated as a DataFrame for column selection
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=['cylinders', 'displacement', 'horsepower', 
                                         'weight', 'acceleration', 'model', 'origin'])
        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        """
        Predicts using the fitted pipeline.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=['cylinders', 'displacement', 'horsepower', 
                                         'weight', 'acceleration', 'model', 'origin'])
        return self.pipeline.predict(X)

def build_model(X_train, y_train):
    """
    Factory function to return a fresh, unfitted model instance.
    
    Parameters:
    X_train: Training features (pandas DataFrame)
    y_train: Training targets (pandas Series)
    
    Returns:
    An unfitted instance of AutoMPGRegressor.
    """
    return AutoMPGRegressor(random_state=42)