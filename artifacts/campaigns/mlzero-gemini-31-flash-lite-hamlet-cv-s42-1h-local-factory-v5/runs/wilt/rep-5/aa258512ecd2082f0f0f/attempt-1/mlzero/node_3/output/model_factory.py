import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.base import BaseEstimator, ClassifierMixin

class WiltClassifier(BaseEstimator, ClassifierMixin):
    """
    A robust classification wrapper using HistGradientBoostingClassifier.
    
    HistGradientBoostingClassifier is chosen for its efficiency, native support 
    for missing values, and excellent performance on tabular data. 
    PowerTransformer is used to handle potential skewness in the features.
    """
    def __init__(self):
        # Pipeline:
        # 1. PowerTransformer: Maps data to a normal distribution (helps with skewed features).
        # 2. StandardScaler: Ensures features are on a similar scale.
        # 3. HistGradientBoostingClassifier: Fast, robust, and handles class imbalance 
        #    via class_weight='balanced'.
        self.model = Pipeline([
            ('yeo-johnson', PowerTransformer(method='yeo-johnson', standardize=True)),
            ('clf', HistGradientBoostingClassifier(
                loss='log_loss',
                learning_rate=0.05,
                max_iter=200,
                max_leaf_nodes=31,
                class_weight='balanced',
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10
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
    X_train: Training features (contextual)
    y_train: Training labels (contextual)
    
    Returns:
    An unfitted instance of WiltClassifier.
    """
    return WiltClassifier()