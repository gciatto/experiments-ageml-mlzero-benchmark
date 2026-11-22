import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier

class ModelWrapper:
    """
    A wrapper class that encapsulates preprocessing and the model.
    Uses HistGradientBoostingClassifier for efficient training on the adult dataset.
    """
    def __init__(self):
        self.model = None
        self.pipeline = None

    def _build_pipeline(self, X):
        # Identify feature types
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

        # Preprocessing for numeric data: Impute missing with median, scale
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        # Preprocessing for categorical data: Impute missing with constant, one-hot encode
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        # Classifier: HistGradientBoosting is fast and handles missing values/categorical data well
        # We limit the complexity to ensure all 10 folds finish within the time budget
        clf = HistGradientBoostingClassifier(
            random_state=42,
            max_iter=100,
            learning_rate=0.1,
            early_stopping=True
        )

        return Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', clf)])

    def fit(self, X, y):
        # Ensure y is encoded as binary if necessary
        if y.dtype == 'object':
            y = (y == y.unique()[0]).astype(int)
        
        self.pipeline = self._build_pipeline(X)
        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        return self.pipeline.predict(X)

def build_model(X_train, y_train):
    """
    Factory function to return a fresh, unfitted model wrapper.
    """
    return ModelWrapper()