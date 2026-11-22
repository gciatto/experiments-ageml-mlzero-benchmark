import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import HistGradientBoostingClassifier

class MadelonModelWrapper(BaseEstimator, ClassifierMixin):
    """
    Optimized wrapper for the Madelon dataset.
    Uses HistGradientBoostingClassifier for speed and performance,
    combined with feature selection to handle the 500-feature space.
    """
    def __init__(self, random_state=42):
        self.random_state = random_state
        # HistGradientBoosting is significantly faster than RandomForest
        # and handles high-dimensional data well.
        # We use a pipeline to ensure all steps are fitted on training data only.
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
        """
        Fits the pipeline. X and y are provided by the CV harness.
        """
        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        """
        Predicts class labels.
        """
        return self.pipeline.predict(X)

    def predict_proba(self, X):
        """
        Predicts class probabilities.
        """
        return self.pipeline.predict_proba(X)

def build_model(X_train, y_train):
    """
    Factory function returning a fresh, unfitted MadelonModelWrapper.
    The harness handles the cross-validation loop and calls fit() 
    on the returned object.
    """
    return MadelonModelWrapper(random_state=42)