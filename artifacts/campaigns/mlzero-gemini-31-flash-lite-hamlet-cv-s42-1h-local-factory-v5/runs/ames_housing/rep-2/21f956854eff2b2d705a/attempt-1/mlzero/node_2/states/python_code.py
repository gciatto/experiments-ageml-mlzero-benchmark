import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import HistGradientBoostingRegressor

class ModelWrapper:
    """
    A wrapper for a regression pipeline that handles preprocessing and 
    HistGradientBoostingRegressor for efficient, robust performance.
    """
    def __init__(self):
        self.model = None
        self._build_pipeline()

    def _build_pipeline(self):
        # Identify feature types
        # We drop 'Id' as per instructions
        self.numeric_features = []
        self.categorical_features = []
        self.model = None

    def _setup_pipeline(self, X):
        # Dynamically detect types
        numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Remove Id if present
        if 'Id' in numeric_cols:
            numeric_cols.remove('Id')

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_cols),
                ('cat', categorical_transformer, categorical_cols)
            ])

        # HistGradientBoostingRegressor is fast and handles missing values/large data well
        self.model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', HistGradientBoostingRegressor(
                random_state=42,
                max_iter=100,
                early_stopping=True
            ))
        ])

    def fit(self, X, y):
        if self.model is None:
            self._setup_pipeline(X)
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

def build_model(X_train, y_train):
    """
    Returns a fresh, unfitted model wrapper.
    """
    return ModelWrapper()