import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor

class CaliforniaHousingModel:
    """
    A wrapper class for the California Housing regression task.
    Uses HistGradientBoostingRegressor for efficiency and performance.
    """
    def __init__(self):
        self.model = None
        self._build_pipeline()

    def _build_pipeline(self):
        # Identify feature types
        numeric_features = [
            'longitude', 'latitude', 'housing_median_age', 'total_rooms', 
            'total_bedrooms', 'population', 'households', 'median_income'
        ]
        categorical_features = ['ocean_proximity']

        # Preprocessing pipeline
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )

        # Regressor with constraints to ensure it fits within the time budget
        # HistGradientBoostingRegressor is fast and handles large datasets well
        regressor = HistGradientBoostingRegressor(
            random_state=42,
            max_iter=200,
            early_stopping=True,
            learning_rate=0.1
        )

        self.model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', regressor)
        ])

    def fit(self, X, y):
        """Fits the pipeline to the training data."""
        self.model.fit(X, y)
        return self

    def predict(self, X):
        """Predicts target values."""
        return self.model.predict(X)

def build_model(X_train, y_train):
    """
    Factory function to return a fresh, unfitted model instance.
    The X_train and y_train arguments are provided by the harness 
    but the model is initialized fresh here.
    """
    return CaliforniaHousingModel()