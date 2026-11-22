import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, ClassifierMixin

class WiltClassifier(BaseEstimator, ClassifierMixin):
    """
    A wrapper class for a classification pipeline.
    Uses a Random Forest with balanced class weights to handle potential
    class imbalance in the wilt dataset.
    """
    def __init__(self):
        # Using a pipeline to ensure scaling is fitted only on training data
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                class_weight='balanced',
                random_state=42,
                n_jobs=1  # Constrained to 1 to avoid resource contention
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
    Factory function to return a fresh, unfitted estimator.
    
    Parameters:
    X_train: Training features (provided for context, though not used for initialization)
    y_train: Training labels (provided for context)
    
    Returns:
    An unfitted instance of WiltClassifier.
    """
    return WiltClassifier()