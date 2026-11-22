import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.base import BaseEstimator, ClassifierMixin

class WiltClassifier(BaseEstimator, ClassifierMixin):
    """
    A robust classifier for the wilt dataset using HistGradientBoostingClassifier.
    
    HistGradientBoostingClassifier is chosen for its speed and native support 
    for missing values and large datasets, making it highly efficient for 
    ten-fold cross-validation within strict time constraints.
    """
    def __init__(self):
        # Pipeline ensures scaling is fitted only on training data.
        # HistGradientBoostingClassifier is generally faster and more memory-efficient
        # than standard RandomForest for this type of tabular data.
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', HistGradientBoostingClassifier(
                loss='log_loss',
                learning_rate=0.1,
                max_iter=100,
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