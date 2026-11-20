import pandas as pd
import numpy as np
import os
import tempfile
import shutil
from autogluon.multimodal import MultiModalPredictor
import logging

# Configure logging to suppress excessive output from AutoGluon if needed.
# For this task, default logging might be verbose, so setting a higher level
# can reduce console spam.
logging.basicConfig(level=logging.WARNING)

# Seed for reproducibility across stochastic operations.
RANDOM_SEED = 42

class AutoGluonMultiModalWrapper:
    """
    A wrapper class for AutoGluon's MultiModalPredictor to conform to the
    MLZero benchmark's fit/predict model contract.
    """
    def __init__(self, target_column: str, problem_type: str, random_state: int = RANDOM_SEED):
        """
        Initializes the wrapper for AutoGluon MultiModalPredictor.

        Args:
            target_column (str): The name of the target column for prediction.
            problem_type (str): The type of the machine learning task (e.g., "regression").
            random_state (int): Seed for reproducibility.
        """
        self.target_column = target_column
        self.problem_type = problem_type
        self.random_state = random_state
        self.predictor = None
        self.temp_dir = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fits the AutoGluon MultiModalPredictor model using the provided training data.

        All preprocessing and model training are performed within this method,
        using only the supplied X and y.

        Args:
            X (pd.DataFrame): Training features.
            y (pd.Series): Training target values.
        """
        # Combine X and y into a single DataFrame, which is the expected input
        # format for AutoGluon's MultiModalPredictor.
        train_data = X.copy()
        train_data[self.target_column] = y

        # Create a temporary directory for AutoGluon artifacts for this specific fold.
        # This ensures that each fold's training is isolated and uses fresh artifacts,
        # preventing data leakage or conflicts between folds.
        # The directory will be automatically cleaned up when the wrapper object is deleted.
        self.temp_dir = tempfile.mkdtemp()
        
        # Initialize MultiModalPredictor.
        # The 'path' argument directs AutoGluon to store its models and intermediate
        # artifacts in the temporary directory.
        # The 'seed' argument ensures reproducibility for any stochastic processes
        # within AutoGluon's training pipeline.
        self.predictor = MultiModalPredictor(
            label=self.target_column,
            problem_type=self.problem_type,
            path=self.temp_dir,
            seed=self.random_state
        )

        # Fit the predictor.
        # A 'time_limit' is crucial to ensure that each fold's training completes
        # within a reasonable portion of the total shared deadline.
        # For a total of 3292 seconds and 10 folds, 200 seconds per fold allows
        # for 2000 seconds of training, leaving approximately 1292 seconds for
        # data loading, model initialization, and other overheads across all folds.
        self.predictor.fit(
            train_data=train_data,
            time_limit=200, # seconds per fold
            # AutoGluon's default presets are generally robust. The problem statement
            # indicates that hyperparameter tuning is unnecessary as presets are configured.
        )

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Makes predictions using the fitted AutoGluon MultiModalPredictor model.

        Args:
            X (pd.DataFrame): Features for which to make predictions.

        Returns:
            pd.Series: Predicted target values.
        """
        if self.predictor is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        
        # AutoGluon MultiModalPredictor expects a DataFrame for prediction input.
        predictions = self.predictor.predict(X)
        return predictions

    def __del__(self):
        """
        Cleans up the temporary directory created for AutoGluon artifacts.
        This method is called automatically when the wrapper object is garbage collected.
        """
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
            except OSError as e:
                # Log any errors during directory removal but do not prevent program termination.
                logging.error(f"Error removing temporary directory {self.temp_dir}: {e}")

def build_model(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Factory function to build a new, unfitted AutoGluon MultiModal regression model.

    This function adheres to the MLZero benchmark contract, returning an object
    that exposes `fit(X,y)` and `predict(X)` methods. The model returned is
    unfitted and ready for training on specific fold data.

    Args:
        X_train (pd.DataFrame): Training features. Used primarily for schema inference
                                (e.g., column names) if needed, but not for fitting at this stage.
        y_train (pd.Series): Training target. Used to infer the target column name.
                             Not used for fitting at this stage.

    Returns:
        AutoGluonMultiModalWrapper: An unfitted model wrapper instance ready for training.
    """
    # Infer the target column name from the y_train Series.
    # Provide a fallback name ("target") if y_train does not have a name attribute.
    target_column = y_train.name if y_train.name is not None else "target"
    
    # The task is regression, as specified in the problem description.
    problem_type = "regression"

    return AutoGluonMultiModalWrapper(target_column=target_column, problem_type=problem_type, random_state=RANDOM_SEED)