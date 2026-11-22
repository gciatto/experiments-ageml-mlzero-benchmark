import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import HistGradientBoostingRegressor

class ModelWrapper:
    """
    A wrapper class that encapsulates preprocessing and the model.
    Uses HistGradientBoostingRegressor for efficient regression on tabular data.
    """
    def __init__(self):
        self.model = None
        self.pipeline = None

    def _build_pipeline(self, X):
        # Identify numeric and categorical columns
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

        # Preprocessing for numeric data: Impute missing with median, then scale
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        # Preprocessing for categorical data: Impute missing with constant, then OHE
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        # HistGradientBoostingRegressor is robust and handles missing values/large datasets well
        # We limit max_iter to ensure we stay within the time budget for 10 folds
        regressor = HistGradientBoostingRegressor(
            max_iter=100,
            learning_rate=0.1,
            max_depth=10,
            random_state=42
        )

        return Pipeline(steps=[('preprocessor', preprocessor),
                              ('regressor', regressor)])

    def fit(self, X, y):
        # Drop 'Id' if present as per instructions
        if 'Id' in X.columns:
            X = X.drop(columns=['Id'])
            
        self.pipeline = self._build_pipeline(X)
        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        if 'Id' in X.columns:
            X = X.drop(columns=['Id'])
        return self.pipeline.predict(X)

def build_model(X_train, y_train):
    """
    Returns a fresh, unfitted model wrapper.
    """
    return ModelWrapper()