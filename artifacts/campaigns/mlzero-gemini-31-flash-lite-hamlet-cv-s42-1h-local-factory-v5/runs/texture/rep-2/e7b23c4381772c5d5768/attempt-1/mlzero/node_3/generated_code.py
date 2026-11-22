import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin

class ModelWrapper(BaseEstimator, ClassifierMixin):
    """
    A wrapper class that encapsulates preprocessing and a classifier.
    Uses HistGradientBoostingClassifier which is highly efficient and 
    robust for tabular data. We disable internal parallelism to prevent 
    resource contention in the restricted CV environment.
    """
    def __init__(self, random_state=42):
        self.random_state = random_state
        # HistGradientBoostingClassifier is faster and often more accurate 
        # than RandomForest for numerical features.
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', HistGradientBoostingClassifier(
                max_iter=200,
                learning_rate=0.1,
                max_depth=10,
                random_state=self.random_state,
                early_stopping=True,
                scoring='balanced_accuracy'
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
        Predicts class labels for the provided input.
        """
        return self.model.predict(X)

def build_model(X_train, y_train):
    """
    Factory function to return a fresh, unfitted model instance.
    
    Args:
        X_train: Training features.
        y_train: Training labels.
        
    Returns:
        An unfitted instance of ModelWrapper.
    """
    # We return a fresh instance as required by the contract.
    # The random_state is fixed to 42 for reproducibility.
    return ModelWrapper(random_state=42)