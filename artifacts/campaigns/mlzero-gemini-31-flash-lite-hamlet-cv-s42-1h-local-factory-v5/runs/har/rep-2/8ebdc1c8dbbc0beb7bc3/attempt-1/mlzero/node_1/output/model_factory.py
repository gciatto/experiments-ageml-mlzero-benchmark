import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, ClassifierMixin

class HARModelWrapper(BaseEstimator, ClassifierMixin):
    """
    A wrapper for a classification pipeline designed for the HAR dataset.
    Uses a Random Forest classifier with standard scaling.
    """
    def __init__(self):
        # We use a Random Forest as it is robust for high-dimensional sensor data
        # and performs well without extensive hyperparameter tuning.
        # n_estimators is set to 100 to balance performance and time constraints.
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                random_state=42,
                n_jobs=1
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

def build_model(X_train, y_train):
    """
    Factory function to return a fresh, unfitted HARModelWrapper instance.
    
    Parameters:
    X_train: Training features (not used for initialization, but required by contract)
    y_train: Training labels (not used for initialization, but required by contract)
    
    Returns:
    An unfitted instance of HARModelWrapper.
    """
    return HARModelWrapper()