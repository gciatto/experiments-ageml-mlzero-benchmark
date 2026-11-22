import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, ClassifierMixin

class MadelonModelWrapper(BaseEstimator, ClassifierMixin):
    """
    A wrapper for a classification pipeline optimized for the Madelon dataset.
    Uses feature selection to handle the high-dimensional, sparse-feature nature
    of the dataset while maintaining computational efficiency for 10-fold CV.
    """
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('variance_filter', VarianceThreshold(threshold=0.1)),
            ('feature_selector', SelectKBest(score_func=f_classif, k=100)),
            ('classifier', RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                n_jobs=1,
                random_state=self.random_state
            ))
        ])

    def fit(self, X, y):
        # Ensure input is handled as expected by sklearn
        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        return self.pipeline.predict(X)

def build_model(X_train, y_train):
    """
    Factory function to return a fresh, unfitted model instance.
    The model encapsulates all preprocessing and feature selection.
    """
    return MadelonModelWrapper(random_state=42)