import pandas as pd
import numpy as np
import os
import uuid
import shutil
from autogluon.tabular import TabularPredictor

# Global counter for unique paths, to ensure distinct directories for each fold.
# This helps in creating fresh artifact directories as required by the contract.
_fold_counter = 0

class AutoGluonTabularWrapper:
    """
    A wrapper class for autogluon.tabular.TabularPredictor to conform to the
    fit/predict interface required by the evaluation harness.
    """
    def __init__(self, target_column_name: str, problem_type: str, eval_metric: str, random_seed: int = 42):
        """
        Initializes the wrapper. The AutoGluon predictor itself is not fitted here.

        Args:
            target_column_name (str): The name of the target column (e.g., 'SalePrice').
            problem_type (str): The type of the machine learning problem (e.g., 'regression').
            eval_metric (str): The evaluation metric to optimize (e.g., 'rmse').
            random_seed (int): Seed for reproducibility.
        """
        self.predictor = None
        self.target_column_name = target_column_name
        self.problem_type = problem_type
        self.eval_metric = eval_metric
        self.random_seed = random_seed
        self.model_path = None # To store the path for the current fold's artifacts

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Fits the AutoGluon TabularPredictor model.
        All preprocessing and model training happens within this method,
        using only the provided training data for the current fold.

        Args:
            X_train (pd.DataFrame): Training features for the current fold.
            y_train (pd.Series): Training target for the current fold.
        """
        global _fold_counter
        _fold_counter += 1
        
        # Create a unique path for each fold's model artifacts.
        # This ensures "fresh artifact directories per fold" as required.
        # The base directory 'autogluon_models_artifacts' will be created in the current
        # working directory of the execution environment.
        base_output_dir = "autogluon_models_artifacts"
        os.makedirs(base_output_dir, exist_ok=True)
        
        # Use a combination of fold counter and UUID for robust uniqueness
        self.model_path = os.path.join(base_output_dir, f"fold_{_fold_counter}_{uuid.uuid4().hex}")
        
        # AutoGluon's TabularPredictor expects a single DataFrame containing both features and target.
        # Ensure we are working on copies to avoid modifying the original DataFrames passed in.
        train_data = X_train.copy()
        train_data[self.target_column_name] = y_train

        # Drop 'Id' column if it exists, as it's typically not a predictive feature
        # in tabular datasets like Ames Housing. 'errors='ignore'' prevents an error
        # if the column is already absent.
        if 'Id' in train_data.columns:
            train_data = train_data.drop(columns=['Id'])

        # Initialize TabularPredictor for the specified problem type and metric.
        # 'path' parameter ensures models are saved to a unique directory for each fold.
        # FIX: As per the latest error message, 'random_seed' is passed to the constructor.
        self.predictor = TabularPredictor(
            label=self.target_column_name,
            problem_type=self.problem_type,
            eval_metric=self.eval_metric,
            path=self.model_path,
            random_seed=self.random_seed # FIX: Added random_seed to the constructor.
        )

        # Fit the predictor.
        # 'time_limit' is crucial for bounding each fit within the overall deadline.
        # Adjusted time_limit to 250 seconds per fold (2500s total for 10 folds)
        # to be safely within the remaining shared deadline (2726 seconds).
        # 'presets'='extreme' is used as suggested by the prompt.
        # FIX: Removed 'seed' from the fit method as per the latest error message.
        self.predictor.fit(
            train_data,
            time_limit=250, # Adjusted to 250 seconds per fold for safety.
            presets='extreme', # As suggested by the prompt.
            # seed=self.random_seed # FIX: Removed seed from fit method.
        )

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Makes predictions using the fitted AutoGluon TabularPredictor.

        Args:
            X_test (pd.DataFrame): Features for which to make predictions.

        Returns:
            np.ndarray: A 1D NumPy array of predicted values.
        """
        if self.predictor is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        
        # Drop 'Id' column from test data if it exists, consistent with training.
        X_test_processed = X_test.copy()
        if 'Id' in X_test_processed.columns:
            X_test_processed = X_test_processed.drop(columns=['Id'])

        # AutoGluon's predict method returns a pandas Series.
        # The contract requires a 1D NumPy array.
        return self.predictor.predict(X_test_processed).values

    def __del__(self):
        """
        Attempts to clean up the model artifacts directory when the wrapper object
        is garbage collected. This helps manage disk space.
        Note: __del__ is not guaranteed to be called in all circumstances.
        """
        if self.model_path and os.path.exists(self.model_path):
            try:
                shutil.rmtree(self.model_path)
            except OSError:
                # Ignore errors during cleanup, e.g., if directory is already removed
                # or if there are permission issues.
                pass


def build_model(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Factory function to create a new, unfitted model wrapper for the Ames Housing dataset.
    
    Args:
        X_train (pd.DataFrame): Example training features. This data is used to infer
                                 the target column name and problem type, but not for fitting.
        y_train (pd.Series): Example training target. Used to get the target column name.
        
    Returns:
        AutoGluonTabularWrapper: An unfitted model wrapper instance configured for
                                 the Ames Housing regression task.
    """
    # The problem is regression, target column is 'SalePrice', and the metric is RMSE.
    # AutoGluon can often infer the problem_type, but being explicit is safer.
    return AutoGluonTabularWrapper(
        target_column_name=y_train.name, # Dynamically get the target column name (e.g., 'SalePrice')
        problem_type='regression',
        eval_metric='rmse', # Root Mean Squared Error, lower is better
        random_seed=42
    )