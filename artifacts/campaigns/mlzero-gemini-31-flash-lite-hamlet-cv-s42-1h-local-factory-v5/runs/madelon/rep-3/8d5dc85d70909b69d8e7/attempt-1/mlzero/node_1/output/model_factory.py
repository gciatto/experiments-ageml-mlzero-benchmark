import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, ClassifierMixin

class MadelonModelWrapper(BaseEstimator, ClassifierMixin):
    """
    A wrapper for a classification pipeline designed for the Madelon dataset.
    Uses feature selection to handle the high-dimensional (500 features) 
    nature of the dataset and a robust Random Forest classifier.
    """
    def __init__(self, random_state=42):
        self.random_state = random_state
        # Madelon is known to have many redundant features. 
        # SelectKBest with f_classif helps reduce noise.
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('selector', SelectKBest(score_func=f_classif, k=100)),
            ('clf', RandomForestClassifier(
                n_estimators=150,
                max_depth=10,
                min_samples_split=5,
                n_jobs=1,
                random_state=self.random_state
            ))
        ])

    def fit(self, X, y):
        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        return self.pipeline.predict(X)

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)

def build_model(X_train, y_train):
    """
    Factory function to return a fresh, unfitted MadelonModelWrapper.
    The harness will call fit() on this object using the provided training folds.
    """
    return MadelonModelWrapper(random_state=42)