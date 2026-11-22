import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin

class WiltClassifierWrapper(BaseEstimator, ClassifierMixin):
    """
    A wrapper for a Random Forest classifier with integrated scaling.
    Designed to be efficient for 10-fold CV within the time constraints.
    """
    def __init__(self, random_state=42):
        self.random_state = random_state
        # Using a pipeline to ensure preprocessing is fitted only on training data
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('rf', RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                n_jobs=1,  # Keep n_jobs=1 to avoid overhead in parallel environments
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

def build_model(X_train, y_train):
    """
    Factory function to return a fresh, unfitted estimator.
    The X_train and y_train arguments are provided by the harness 
    to allow for potential dynamic model configuration.
    """
    return WiltClassifierWrapper(random_state=42)