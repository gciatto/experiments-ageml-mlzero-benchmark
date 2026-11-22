import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.ensemble import HistGradientBoostingClassifier

class MadelonModelWrapper(BaseEstimator, ClassifierMixin):
    """
    A robust classifier for the Madelon dataset.
    Uses HistGradientBoostingClassifier for efficiency and speed, 
    combined with feature selection to handle the 500-feature space.
    """
    def __init__(self, random_state=42):
        self.random_state = random_state
        # Pipeline optimized for speed and performance within tight time constraints.
        # HistGradientBoosting is significantly faster than standard RF for large feature sets.
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('variance_filter', VarianceThreshold(threshold=0.1)),
            ('selector', SelectKBest(score_func=f_classif, k=150)),
            ('clf', HistGradientBoostingClassifier(
                max_iter=100,
                learning_rate=0.1,
                max_depth=7,
                random_state=self.random_state,
                early_stopping=True
            ))
        ])

    def fit(self, X, y):
        # Ensure X is treated as a DataFrame or array consistently
        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        return self.pipeline.predict(X)

    def get_params(self, deep=True):
        return {"random_state": self.random_state}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

def build_model(X_train, y_train):
    """
    Returns a fresh, unfitted instance of the MadelonModelWrapper.
    The harness will call fit() on this object using the provided training data.
    """
    return MadelonModelWrapper(random_state=42)