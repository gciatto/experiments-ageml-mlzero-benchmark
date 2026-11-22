import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.compose import ColumnTransformer

class AutoMPGRegressor(BaseEstimator, RegressorMixin):
    """
    A robust regression wrapper for the auto_mpg dataset.
    Uses a pipeline with power transformation for skewed features,
    imputation, scaling, and a gradient boosting regressor.
    """
    def __init__(self, random_state=42):
        self.random_state = random_state
        # Using PowerTransformer to handle non-normal distributions common in auto data
        # HistGradientBoostingRegressor is efficient and handles missing values internally,
        # but we include an imputer for safety.
        self.pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('transformer', PowerTransformer(method='yeo-johnson')),
            ('scaler', StandardScaler()),
            ('regressor', HistGradientBoostingRegressor(
                random_state=self.random_state,
                max_iter=100,
                learning_rate=0.1,
                max_depth=6,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=5,
                scoring='neg_root_mean_squared_error'
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
    # All preprocessing is encapsulated within the pipeline.
    return AutoMPGRegressor(random_state=42)