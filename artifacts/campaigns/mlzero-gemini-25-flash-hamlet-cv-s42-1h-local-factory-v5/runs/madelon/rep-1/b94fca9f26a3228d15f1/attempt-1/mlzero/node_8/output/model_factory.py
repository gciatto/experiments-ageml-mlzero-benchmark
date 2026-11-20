import pandas as pd
import numpy as np
import os
import logging
import random
import shutil

# AutoGluon imports
from autogluon.tabular import TabularPredictor

# Configure basic logging for the module.
# This helps in debugging if AutoGluon or other parts of the code emit logs.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set global seeds for reproducibility as requested by the task.
# AutoGluon's internal models often rely on numpy and standard random module seeds.
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
logging.info(f"Global random seed set to {SEED} for random and numpy.")

# Module-level constants.
# The remaining TOTAL search time is 242 seconds for 10 fits.
# This means each fold gets approximately 24.2 seconds on average.
# Setting a time limit of 20 seconds per fit ensures that
# 10 fits will take at most 200 seconds, leaving a buffer for overhead (e.g., Python startup,
# data loading, AutoGluon initialization outside of the .fit() call).
# This is a very aggressive time limit, chosen to ensure all folds complete within the deadline.
DEFAULT_TIME_LIMIT_SECONDS = 20

class AutoGluonTabularWrapper:
    """
    A wrapper class for AutoGluon's TabularPredictor to conform to the
    fit/predict interface required by the evaluation harness.
    """
    def __init__(self, target_column_name: str, problem_type: str):
        """
        Initializes the wrapper with target column name and problem type.
        The actual AutoGluon predictor is initialized and fitted in the .fit() method.

        Args:
            target_column_name (str): The name of the target column.
            problem_type (str): The type of the machine learning problem (e.g., 'binary', 'multiclass').
        """
        self.target_column_name = target_column_name
        self.problem_type = problem_type
        self.predictor = None
        # AutoGluon will save its models to this path.
        # The harness guarantees "fresh artifact directories per fold",
        # so a simple relative path like "AutogluonModels" will be unique
        # for each fold's execution if the working directory changes per fold.
        self.model_path = "AutogluonModels"
        logging.info(f"AutoGluonTabularWrapper initialized. Model artifacts path: {self.model_path}")

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fits the AutoGluon TabularPredictor model.
        All preprocessing and model fitting happens here, using only the provided X and y.

        Args:
            X (pd.DataFrame): Training features for the current fold.
            y (pd.Series): Training target for the current fold.
        """
        logging.info(f"Starting AutoGluon fit for target '{self.target_column_name}' with problem type '{self.problem_type}'.")
        logging.info(f"AutoGluon model artifacts will be saved to: {os.path.abspath(self.model_path)}")

        # Defensive measure: Ensure the model_path directory is clean before starting a new fit.
        # This helps prevent issues if the harness doesn't fully clean the working directory
        # between folds or if AutoGluon tries to load existing models.
        if os.path.exists(self.model_path):
            logging.warning(f"Existing AutoGluon model directory '{self.model_path}' found. Removing it.")
            try:
                shutil.rmtree(self.model_path)
            except OSError as e:
                logging.error(f"Error removing existing AutoGluon model directory: {e}")
                # Attempt to proceed, AutoGluon might handle it, or it might fail later.

        # Combine features and target into a single DataFrame for AutoGluon.
        # AutoGluon expects the target column to be present in the training data.
        train_data = pd.concat([X, y], axis=1)

        # Initialize TabularPredictor.
        # The 'path' argument is crucial for AutoGluon to store its artifacts.
        # 'seed' and 'random_state' are not valid for the constructor or fit method directly
        # in the installed AutoGluon version, as per previous errors.
        # Global seeds (random.seed, np.random.seed) are set at the module level.
        self.predictor = TabularPredictor(
            label=self.target_column_name,
            problem_type=self.problem_type,
            path=self.model_path,
            verbosity=2 # Set verbosity to 2 for INFO level output from AutoGluon
        )

        # Fit the predictor.
        # All learned preprocessing (e.g., feature engineering, scaling) is handled
        # internally by AutoGluon during this fit call, using only `train_data`.
        # Using 'good_quality' preset for a balance between performance and speed,
        # given the extremely tight time limit. 'best_quality' is likely too slow.
        self.predictor.fit(
            train_data,
            time_limit=DEFAULT_TIME_LIMIT_SECONDS, # Bound the fitting time
            presets='good_quality', # Use a faster preset for the tight deadline
        )
        logging.info("AutoGluon fit completed.")

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Makes predictions using the fitted AutoGluon model.

        Args:
            X (pd.DataFrame): Features for which to make predictions.

        Returns:
            pd.Series: Predicted class labels.
        """
        if self.predictor is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        logging.info("Making predictions with AutoGluon model.")
        return self.predictor.predict(X)

def build_model(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Factory function to create a new, unfitted AutoGluon model wrapper.
    This function is called by the evaluation harness for each fold.

    Args:
        X_train (pd.DataFrame): Training features. Used to infer target column name and problem type.
                                This data is NOT used for fitting here, only for metadata.
        y_train (pd.Series): Training target. Used to infer target column name and problem type.
                             This data is NOT used for fitting here, only for metadata.

    Returns:
        AutoGluonTabularWrapper: An unfitted model wrapper with fit() and predict() methods.
    """
    logging.info("build_model called: Initializing AutoGluonTabularWrapper.")

    # Infer the target column name from y_train Series name.
    target_column_name = y_train.name
    if target_column_name is None:
        # Fallback if y_train has no name. This might happen if y_train was created
        # without explicitly setting a name.
        target_column_name = "Class" # Default to 'Class' as per problem description
        logging.warning(f"y_train Series has no name. Assuming target column name is '{target_column_name}'.")

    # Infer problem type (binary or multiclass classification) based on unique values in y_train.
    num_unique_classes = y_train.nunique()
    if num_unique_classes == 2:
        problem_type = 'binary'
        logging.info(f"Inferred problem type: {problem_type} (2 unique classes).")
    elif num_unique_classes > 2:
        problem_type = 'multiclass'
        logging.info(f"Inferred problem type: {problem_type} ({num_unique_classes} unique classes).")
    else:
        raise ValueError(f"Invalid number of unique classes in y_train: {num_unique_classes}. "
                         "Must be >= 2 for classification tasks.")

    # Return a fresh, unfitted instance of our wrapper.
    # The actual AutoGluon predictor will be initialized and fitted when .fit() is called.
    return AutoGluonTabularWrapper(target_column_name=target_column_name, problem_type=problem_type)