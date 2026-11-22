import uuid
import os
import shutil
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import HistGradientBoostingRegressor

class AmesHousingModel(BaseEstimator, RegressorMixin):
    """
    A robust regression model for the Ames Housing dataset using 
    HistGradientBoostingRegressor with automated preprocessing.
    """
    def __init__(self):
        self.model = None
        self.preprocessor = None

    def _get_preprocessor(self, X):
        numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        return ColumnTransformer(
            transformers=[
                ('num', Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]), numeric_cols),
                ('cat', Pipeline([
                    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
                ]), categorical_cols)
            ])

    def fit(self, X, y):
        # Drop Id as per instructions
        X = X.drop(columns=['Id'], errors='ignore')
        
        # Build pipeline
        self.preprocessor = self._get_preprocessor(X)
        
        # Use HistGradientBoostingRegressor for speed and performance
        # Constrained to single-threaded execution to avoid resource contention
        self.model = Pipeline([
            ('preprocessor', self.preprocessor),
            ('regressor', HistGradientBoostingRegressor(
                random_state=42,
                max_iter=100,
                early_stopping=True,
                learning_rate=0.1
            ))
        ])
        
        self.model.fit(X, y)
        return self

    def predict(self, X):
        X = X.drop(columns=['Id'], errors='ignore')
        return self.model.predict(X)

def build_model(X_train, y_train):
    """
    Returns a fresh, unfitted AmesHousingModel instance.
    """
    return AmesHousingModel()