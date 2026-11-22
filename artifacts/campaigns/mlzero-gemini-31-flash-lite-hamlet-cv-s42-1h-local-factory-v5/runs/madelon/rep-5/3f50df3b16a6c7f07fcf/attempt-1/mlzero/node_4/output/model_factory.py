import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import HistGradientBoostingClassifier

class MadelonModelWrapper(BaseEstimator, ClassifierMixin):
    """
    A robust wrapper for the Madelon dataset using HistGradientBoostingClassifier.
    
    Madelon is a high-dimensional dataset with many redundant features.
    We use:
    1. StandardScaler: Normalizes features for better convergence.
    2. SelectKBest: Reduces dimensionality to focus on the most informative features.
    3. HistGradientBoostingClassifier: Highly efficient for medium-to-large datasets,
       natively handles missing values and is significantly faster than standard 
       RandomForest, ensuring we stay within the strict time budget.
    """
    def __init__(self, random_state=42):
        self.random_state = random_state
        # We use a pipeline to ensure all preprocessing is fitted on training data only.
        # k=150 is chosen as a balance between capturing signal and reducing noise.
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('selector', SelectKBest(score_func=f_classif, k=150)),
            ('clf', HistGradientBoostingClassifier(
                max_iter=100,
                learning_rate=0.1,
                max_depth=8,
                random_state=self.random_state,
                early_stopping=True
            ))
        ])

    def fit(self, X, y):
        """Fits the pipeline on the provided training data."""
        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        """Predicts class labels."""
        return self.pipeline.predict(X)

    def predict_proba(self, X):
        """Predicts class probabilities."""
        return self.pipeline.predict_proba(X)

def build_model(X_train, y_train):
    """
    Returns a fresh, unfitted instance of the MadelonModelWrapper.
    The harness will call fit() on this object using the provided training data.
    """
    return MadelonModelWrapper(random_state=42)