import pandas as pd
import numpy as np
import random
import os
import tempfile
import shutil
import sys
from autogluon.tabular import TabularPredictor
from sklearn.base import BaseEstimator, RegressorMixin

# Set global seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# AutoGluon can be sensitive to environment variables for reproducibility
# and resource management. While often set at the shell level,
# setting them here can sometimes help ensure consistent behavior.
# For this context, we primarily rely on global Python seeds.
# os.environ['AG_OPENBLAS_NUM_THREADS'] = '1'
# os.environ['AG_NUM_CORES'] = '1'


class AutoGluonTabularWrapper(BaseEstimator, RegressorMixin):
    """
    A wrapper for AutoGluon's TabularPredictor to conform to scikit-learn's
    fit/predict interface for regression tasks.
    """
    def __init__(self, time_limit=60):
        """
        Initializes the AutoGluonTabularWrapper.

        Args:
            time_limit (int, optional): Maximum time in seconds for AutoGluon to train.
                                        This helps ensure all ten fits finish within
                                        the shared deadline. Default is 60 seconds.
        """
        self.time_limit = time_limit
        self.predictor = None
        self._temp_path = None  # To store the path of the temporary directory for AutoGluon artifacts
        self.label_column_name = 'class' # Default name for the target column, based on problem description
        self._feature_columns = None # To store feature column names from training data

    def fit(self, X, y):
        """
        Fits the AutoGluon TabularPredictor model.

        All preprocessing and model fitting are performed within this method,
        using only the provided training data (X, y).

        Args:
            X (pd.DataFrame or np.ndarray): Training features.
            y (pd.Series or np.ndarray): Training target values.
        """
        # The contract states X_train is pd.DataFrame, so X should be a DataFrame here.
        # If it's not (e.g., numpy array), convert it.
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # Store feature column names for consistent prediction later.
        # This is crucial if X in predict is a numpy array and needs column names.
        self._feature_columns = X.columns.tolist()

        # Ensure y is a Series with a name. If y is already a Series, use its name.
        # Otherwise, use the default 'class' as per the problem description.
        # Ensure index alignment with X for correct data merging.
        if isinstance(y, pd.Series):
            self.label_column_name = y.name if y.name else self.label_column_name
            y = y.reindex(X.index) # Align index with X
        else:
            y = pd.Series(y, name=self.label_column_name, index=X.index)

        # AutoGluon expects X and y to be combined into a single DataFrame
        train_data = X.copy()
        train_data[self.label_column_name] = y

        # Create a unique temporary directory for AutoGluon artifacts for this fold.
        # This ensures fresh artifact directories per fold as required by the contract.
        self._temp_path = tempfile.mkdtemp(prefix="autogluon_model_")
        
        # Initialize TabularPredictor
        # For regression, problem_type is automatically inferred if not specified.
        # eval_metric is also automatically inferred (RMSE for regression).
        self.predictor = TabularPredictor(
            label=self.label_column_name,
            path=self._temp_path,
            verbosity=0 # Suppress verbose output for cleaner logs
        )

        # Fit the predictor
        # 'best_quality' preset aims for the best performance, potentially trying
        # multiple models and ensembles.
        self.predictor.fit(
            train_data=train_data,
            time_limit=self.time_limit,
            presets='best_quality',
            # AutoGluon's TabularPredictor.fit does not have a direct 'seed' argument.
            # Global numpy and random seeds are set at the module level.
        )
        return self

    def predict(self, X):
        """
        Predicts target values for new data.

        Args:
            X (pd.DataFrame or np.ndarray): Features to make predictions on.

        Returns:
            np.ndarray: Predicted target values.
        """
        if self.predictor is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # Ensure X is a DataFrame with the same column names as training data.
        # This is important for AutoGluon to correctly map features.
        if not isinstance(X, pd.DataFrame):
            if self._feature_columns:
                X = pd.DataFrame(X, columns=self._feature_columns)
            else:
                # Fallback if _feature_columns was not set (e.g., if fit was not called,
                # or if X_train was an empty DataFrame, which is unlikely).
                X = pd.DataFrame(X)

        predictions = self.predictor.predict(X)
        return predictions.values # Return as numpy array

    def __del__(self):
        """
        Clean up the temporary directory created for AutoGluon artifacts.
        This method is called when the object is garbage collected.
        """
        if self._temp_path and os.path.exists(self._temp_path):
            try:
                shutil.rmtree(self._temp_path)
            except OSError as e:
                # Log the error or print a warning if cleanup fails, but don't fail the program.
                # In a production environment, a more robust logging mechanism would be used.
                print(f"Warning: Could not remove temporary directory {self._temp_path}: {e}", file=sys.stderr)


def build_model(X_train, y_train):
    """
    Factory function to create a new, unfitted model wrapper.

    This function is called for each cross-validation fold. It returns a fresh
    model instance that will be fitted on the specific fold's training data.

    Args:
        X_train (pd.DataFrame): Training features (provided for context, e.g., column names).
        y_train (pd.Series): Training target (provided for context, e.g., target name).

    Returns:
        AutoGluonTabularWrapper: An unfitted model wrapper instance ready for fitting.
    """
    # The time_limit is set to 60 seconds per fold. For 10 folds, this totals
    # 600 seconds, which is well within the remaining shared deadline of 729 seconds,
    # allowing for overhead.
    return AutoGluonTabularWrapper(time_limit=60)