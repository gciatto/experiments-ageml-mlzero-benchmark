import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin

class WiltClassifierWrapper(BaseEstimator, ClassifierMixin):
    """
    A wrapper for a Random Forest classifier with integrated scaling.
    Designed to be lightweight and fast to fit within the 10-fold CV time constraints.
    """
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('rf', RandomForestClassifier(
                n_estimators=50,
                max_depth=10,
                min_samples_split=5,
                n_jobs=1,
                random_state=self.random_state,
                class_weight='balanced'
            ))
        ])

    def fit(self, X, y):
        """
        Fits the pipeline on the provided training data.
        """
        self.model.fit(X, y)
        return self

    def predict(self, X):
        """
        Predicts class labels for the provided data.
        """
        return self.model.predict(X)

    def get_params(self, deep=True):
        return {"random_state": self.random_state}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

def build_model(X_train, y_train):
    """
    Factory function to return a fresh, unfitted WiltClassifierWrapper.
    
    Parameters:
    X_train: Training features (not used for initialization, but required by contract)
    y_train: Training labels (not used for initialization, but required by contract)
    
    Returns:
    An unfitted instance of WiltClassifierWrapper.
    """
    return WiltClassifierWrapper(random_state=42)