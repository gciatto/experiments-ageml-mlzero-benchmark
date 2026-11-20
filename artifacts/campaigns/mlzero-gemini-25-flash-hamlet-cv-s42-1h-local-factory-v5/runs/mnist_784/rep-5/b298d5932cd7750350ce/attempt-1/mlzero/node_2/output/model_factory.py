import pandas as pd
import numpy as np
import random
import os
import logging
import uuid
from autogluon.tabular import TabularPredictor
from autogluon.common.utils.utils import set_seed

# --- Module-level initialization ---
# Set global seeds for reproducibility across different libraries and operations.
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
set_seed(SEED)  # Set AutoGluon's internal seed for its components

# Configure logging to suppress excessive AutoGluon output.
# This helps keep the console clean during evaluation and focuses on critical messages.
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('autogluon').setLevel(logging.WARNING)
logging.getLogger('autogluon.core').setLevel(logging.WARNING)
logging.getLogger('autogluon.tabular').setLevel(logging.WARNING)
logging.getLogger('autogluon.features').setLevel(logging.WARNING)
logging.getLogger('autogluon.mxnet').setLevel(logging.WARNING) # Relevant if MXNet backend is used
logging.getLogger('autogluon.extra').setLevel(logging.WARNING)

# Define a base directory for AutoGluon models' artifacts.
# This path is relative to the current working directory where the harness executes.
# Using a unique subdirectory for each fold ensures isolation and prevents conflicts.
AUTOGLUON_BASE_DIR = "autogluon_models_artifacts"

class AutoGluonModelWrapper:
    """
    A wrapper class for AutoGluon's TabularPredictor, designed to provide a
    scikit-learn-like fit/predict interface. This class encapsulates the
    AutoGluon model and handles its lifecycle, including artifact management.
    """
    def __init__(self, problem_type: str, eval_metric: str):
        """
        Initializes the wrapper. The AutoGluon predictor itself is not
        instantiated or fitted until the .fit() method is called.

        Args:
            problem_type (str): The type of machine learning problem (e.g., 'multiclass').
            eval_metric (str): The evaluation metric to optimize (e.g., 'balanced_accuracy').
        """
        self.predictor = None
        self.problem_type = problem_type
        self.eval_metric = eval_metric
        self.model_path = None  # Stores the unique path for this model instance's artifacts

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fits the AutoGluon TabularPredictor model using the provided training data.
        All preprocessing, feature engineering, and model training are performed
        internally by AutoGluon within this method, using only the supplied (X, y).

        Args:
            X (pd.DataFrame): Training features.
            y (pd.Series): Training target labels.
        """
        # Create a unique path for this fold's AutoGluon artifacts.
        # This is crucial for cross-validation to ensure each fold's model
        # is trained and saved independently without interfering with others.
        self.model_path = os.path.join(AUTOGLUON_BASE_DIR, f"fold_{uuid.uuid4()}")
        os.makedirs(self.model_path, exist_ok=True)

        # AutoGluon's TabularPredictor expects the target column to be part of the DataFrame.
        train_data = X.copy()
        train_data[y.name] = y

        # Initialize TabularPredictor.
        # 'label' specifies the name of the target column.
        # 'path' specifies where AutoGluon will save its models and intermediate files.
        # 'problem_type' and 'eval_metric' are set based on the task requirements.
        self.predictor = TabularPredictor(
            label=y.name,
            path=self.model_path,
            problem_type=self.problem_type,
            eval_metric=self.eval_metric,
        )

        # Set a time limit for each fit operation.
        # The total time for ten folds is 3220 seconds, so approximately 322 seconds per fold.
        # Setting it to 300 seconds allows for some overhead and ensures all fits can complete.
        time_limit_per_fold = 300

        # Fit the predictor. 'presets' like 'best_quality' instruct AutoGluon
        # to try various models and ensemble them for optimal performance.
        # Based on previous errors, 'env.seed' and 'random_state' are not
        # valid direct arguments for `predictor.fit()`, so they are omitted.
        self.predictor.fit(
            train_data=train_data,
            time_limit=time_limit_per_fold,
            presets='best_quality',  # Aims for the highest quality model within the time limit
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generates predictions for the input features X using the fitted model.

        Args:
            X (pd.DataFrame): Features for which to generate predictions.

        Returns:
            np.ndarray: A 1-dimensional NumPy array of predicted class labels.
        """
        if self.predictor is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit(X, y) first.")

        # AutoGluon's predict method returns a pandas Series or DataFrame.
        # The contract requires a 1-dimensional NumPy array of predicted class labels.
        predictions = self.predictor.predict(X)
        return predictions.to_numpy()

def build_model(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Factory function to create a new, unfitted AutoGluon model wrapper.
    This function is called once for each cross-validation fold by the harness.

    Args:
        X_train (pd.DataFrame): Training features (used here for metadata inference, not fitting).
        y_train (pd.Series): Training target labels (used here for metadata inference, not fitting).

    Returns:
        AutoGluonModelWrapper: An unfitted instance of the model wrapper.
    """
    # The data science task is classification, specifically multi-class (MNIST digits 0-9).
    # The target column is named 'class'.
    # The evaluation metric specified in the README is 'balanced_accuracy'.
    return AutoGluonModelWrapper(problem_type='multiclass', eval_metric='balanced_accuracy')