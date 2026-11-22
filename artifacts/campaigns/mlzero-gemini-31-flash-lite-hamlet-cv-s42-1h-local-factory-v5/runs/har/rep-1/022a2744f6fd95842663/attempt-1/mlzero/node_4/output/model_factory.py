import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

class HARModelWrapper:
    """
    A wrapper class for the HAR classification task.
    Uses a pipeline with feature selection, scaling, and a Random Forest classifier.
    Optimized for speed to fit within the 14-second total budget for 10 folds.
    """
    def __init__(self):
        # Using a smaller subset of features and limited trees to ensure 
        # all 10 folds complete within the very tight time budget.
        self.model = Pipeline([
            ('selector', SelectKBest(score_func=f_classif, k=150)),
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(
                n_estimators=50,
                max_depth=15,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=1  # n_jobs=1 to avoid overhead and resource contention
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
    Factory function to return a fresh, unfitted model wrapper.
    The X_train and y_train arguments are provided by the harness but 
    the model is returned unfitted as per the contract.
    """
    return HARModelWrapper()