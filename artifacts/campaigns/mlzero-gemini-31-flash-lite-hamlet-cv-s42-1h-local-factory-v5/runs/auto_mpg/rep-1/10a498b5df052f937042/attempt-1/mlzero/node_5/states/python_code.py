import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import HistGradientBoostingRegressor

class AutoMPGRegressor(BaseEstimator, RegressorMixin):
    """
    A robust regression wrapper using HistGradientBoostingRegressor.
    Includes automated feature preprocessing to handle numerical and categorical
    data types effectively for the auto_mpg dataset.
    """
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.pipeline = None

    def _build_pipeline(self, X):
        # Identify numerical and categorical columns
        # 'origin' is categorical, others are numerical
        numeric_features = [col for col in X.columns if col != 'origin']
        categorical_features = ['origin'] if 'origin' in X.columns else []

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ]
        )

        # HistGradientBoostingRegressor is highly efficient for tabular data
        # and handles missing values natively.
        regressor = HistGradientBoostingRegressor(
            random_state=self.random_state,
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=31,
            early_stopping=True,
            validation_fraction=0.1
        )

        return Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', regressor)
        ])

    def fit(self, X, y):
        """
        Fits the pipeline on the provided training data.
        """
        if self.pipeline is None:
            self.pipeline = self._build_pipeline(X)
        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        """
        Predicts using the fitted pipeline.
        """
        if self.pipeline is None:
            raise ValueError("Model must be fitted before calling predict.")
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
    # The model is initialized with a fixed seed for reproducibility.
    # Preprocessing is encapsulated within the pipeline and fitted on fold data.
    return AutoMPGRegressor(random_state=42)