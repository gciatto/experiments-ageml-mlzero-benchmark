import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

class ModelWrapper:
    """
    A wrapper for a Scikit-Learn pipeline that handles classification.
    Uses SGDClassifier with 'log_loss' to approximate high-performance 
    linear classification suitable for MNIST-like datasets.
    """
    def __init__(self, seed=42):
        self.seed = seed
        # Pipeline: Scale features to mean 0, var 1, then apply linear classifier.
        # SGDClassifier is efficient for 784 features and large datasets.
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SGDClassifier(
                loss='log_loss',
                penalty='l2',
                alpha=1e-4,
                max_iter=1000,
                tol=1e-3,
                random_state=self.seed,
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
        Predicts class labels for the provided input features.
        """
        return self.model.predict(X)

def build_model(X_train, y_train):
    """
    Factory function to return a fresh, unfitted model wrapper.
    The X_train and y_train arguments are provided by the harness 
    but are not used for initialization, as the model must be 
    unfitted upon return.
    """
    return ModelWrapper(seed=42)