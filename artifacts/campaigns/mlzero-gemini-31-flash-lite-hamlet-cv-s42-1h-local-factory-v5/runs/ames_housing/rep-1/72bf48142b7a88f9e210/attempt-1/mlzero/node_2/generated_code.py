import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.base import BaseEstimator, RegressorMixin

class AmesHousingModel(BaseEstimator, RegressorMixin):
    """
    A wrapper for a regression pipeline tailored for the Ames Housing dataset.
    Uses HistGradientBoostingRegressor for efficiency and performance.
    """
    def __init__(self):
        self.model = None
        self.pipeline = None

    def _build_pipeline(self, X):
        # Identify feature types
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

        # Drop 'Id' if present
        if 'Id' in numeric_features:
            numeric_features.remove('Id')

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        # HistGradientBoostingRegressor is fast and handles missing values/categorical data well
        # We limit iterations to ensure we stay within the total time budget for 10 folds
        regressor = HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.1,
            max_depth=10,
            random_state=42,
            early_stopping=True
        )

        return Pipeline(steps=[('preprocessor', preprocessor),
                              ('regressor', regressor)])

    def fit(self, X, y):
        # Drop Id column if it exists to prevent leakage/noise
        X_clean = X.drop(columns=['Id'], errors='ignore')
        self.pipeline = self._build_pipeline(X_clean)
        self.pipeline.fit(X_clean, y)
        return self

    def predict(self, X):
        X_clean = X.drop(columns=['Id'], errors='ignore')
        return self.pipeline.predict(X_clean)

def build_model(X_train, y_train):
    """
    Returns a fresh, unfitted AmesHousingModel instance.
    """
    return AmesHousingModel()