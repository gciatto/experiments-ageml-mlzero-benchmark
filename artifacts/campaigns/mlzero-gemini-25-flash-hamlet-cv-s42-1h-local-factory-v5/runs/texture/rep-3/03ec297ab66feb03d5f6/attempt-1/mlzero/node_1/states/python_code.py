import pandas as pd
import numpy as np
import os
import shutil
import tempfile
from autogluon.tabular import TabularPredictor
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

# Module-level constants for reproducibility and resource management.
RANDOM_SEED = 42
# The total remaining search time is 3400 seconds for 10 folds.
# This allows approximately 340 seconds per fold.
# Setting a time_limit of 300 seconds per fit to leave a buffer for
# setup, teardown, and other overhead within each fold.
TIME_LIMIT_PER_FOLD = 300 

class AutoGluonModelWrapper(BaseEstimator, ClassifierMixin):
    """
    A wrapper for AutoGluon's TabularPredictor to conform to scikit-learn's
    fit/predict interface. This class handles the creation of unique artifact
    directories for each model instance and ensures proper cleanup.
    """
    def __init__(self):
        self.predictor_ = None
        self.path_ = None
        # The target column name is 'Class' as specified in the data contract.
        self.label_column_ = 'Class' 
        self.is_fitted_ = False # Internal flag to track if the model has been fitted

    def fit(self, X, y):
        """
        Fits the AutoGluon TabularPredictor model.
        All necessary preprocessing is handled internally by AutoGluon and
        is fitted exclusively on the provided training data (X, y).

        Args:
            X (pd.DataFrame or array-like): Training features.
            y (pd.Series or array-like): Training labels.

        Returns:
            self: The fitted model wrapper instance.
        """
        # Ensure X and y are pandas DataFrames/Series for AutoGluon compatibility.
        # This also handles cases where X, y might be numpy arrays.
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            # Ensure y has a name for the label column and aligns with X's index.
            y = pd.Series(y, name=self.label_column_, index=X.index)

        # Create a unique temporary directory for AutoGluon artifacts for this fold.
        # This addresses the requirement for "fresh artifact directories per fold"
        # and prevents conflicts between different model instances or folds.
        self.path_ = tempfile.mkdtemp(prefix="autogluon_model_")
        
        # Combine X and y into a single DataFrame as required by AutoGluon's fit method.
        train_data = X.copy()
        train_data[self.label_column_] = y

        # Initialize TabularPredictor.
        # The 'random_seed' argument is not for the constructor, as per previous error.
        # 'eval_metric' is set to 'balanced_accuracy' as specified in the README.md.
        self.predictor_ = TabularPredictor(
            label=self.label_column_,
            path=self.path_,
            eval_metric='balanced_accuracy',
            verbosity=0 # Suppress excessive output during training for cleaner logs.
        )

        # Fit the predictor with a time limit and the specified random seed.
        # The 'seed' parameter is passed directly to the fit method for reproducibility.
        self.predictor_.fit(
            train_data,
            time_limit=TIME_LIMIT_PER_FOLD,
            presets='medium_quality_faster_inference', # Balances model quality and training speed.
            seed=RANDOM_SEED 
        )
        
        self.is_fitted_ = True
        return self

    def predict(self, X):
        """
        Predicts class labels for the input data.

        Args:
            X (pd.DataFrame or array-like): Input features for prediction.

        Returns:
            np.ndarray: A 1-dimensional array of predicted class labels (integers).
        """
        # Ensure the model has been fitted before making predictions.
        check_is_fitted(self, 'is_fitted_')
        
        # Ensure X is a pandas DataFrame for AutoGluon compatibility.
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # AutoGluon's predict method expects a DataFrame.
        predictions = self.predictor_.predict(X)
        
        # Ensure the output is a 1-dimensional array of integers, as required by the contract.
        return predictions.values.astype(int)

    def __del__(self):
        """
        Cleans up the temporary directory created by AutoGluon when the object
        is garbage collected. This is crucial for managing disk space, especially
        in a cross-validation setup with multiple model instances.
        """
        if self.path_ and os.path.exists(self.path_):
            try:
                shutil.rmtree(self.path_)
            except OSError as e:
                # Print a warning if cleanup fails, but do not block execution.
                print(f"Warning: Could not remove AutoGluon temporary directory {self.path_}: {e}")

def build_model(X_train, y_train):
    """
    Factory function to create a new, unfitted AutoGluon model wrapper.
    This function adheres to the contract by returning a fresh model instance
    that has not yet been fitted.

    Args:
        X_train (pd.DataFrame): Training features (provided for context, but not
                                used for fitting within this function call).
        y_train (pd.Series): Training labels (provided for context, but not
                              used for fitting within this function call).

    Returns:
        AutoGluonModelWrapper: A new, unfitted model instance with `fit(X,y)`
                               and `predict(X)` methods.
    """
    return AutoGluonModelWrapper()