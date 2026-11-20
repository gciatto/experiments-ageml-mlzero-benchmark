import pandas as pd
import numpy as np
import random
import os
import uuid
from pathlib import Path

# AutoGluon imports
from autogluon.multimodal import MultiModalPredictor
# Fix: Update the import path for set_seed as per AutoGluon API changes.
# For AutoGluon version 1.6.4b, set_seed is typically found in autogluon.core.utils.utils.
from autogluon.core.utils.utils import set_seed 

# Set global seed for reproducibility across all folds.
# This ensures that any stochastic processes within AutoGluon, numpy, and Python's random module
# are consistent across runs and folds, as much as the library allows.
SEED = 42
set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


class AutoGluonMultiModalWrapper:
    """
    A wrapper class for AutoGluon's MultiModalPredictor to conform to the
    fit/predict interface required by the benchmark harness.
    This class encapsulates the model training and prediction logic,
    ensuring that all preprocessing and fitting occur within the `fit` method
    using only the provided training data for each fold.
    """

    def __init__(self):
        self.predictor = None
        self.model_path = None
        self.target_column = 'SalePrice'  # As per problem description
        self.problem_type = 'regression'  # As per problem description

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fits the AutoGluon MultiModalPredictor on the provided training data.

        Args:
            X (pd.DataFrame): Training features for the current fold.
            y (pd.Series): Training labels for the current fold.
        """
        # Combine X and y into a single DataFrame, which MultiModalPredictor expects.
        train_data = X.copy()
        train_data[self.target_column] = y

        # Drop the 'Id' column if it exists, as per the problem statement.
        # This ensures 'Id' is not used as a feature.
        if 'Id' in train_data.columns:
            train_data = train_data.drop(columns=['Id'])

        # Create a unique path for model artifacts for this specific fold.
        # This satisfies the requirement for "fresh artifact directories per fold"
        # and prevents conflicts between different folds or runs.
        # The path is relative to the current working directory where the script is executed.
        self.model_path = Path(f"./automm_models/{uuid.uuid4().hex}")
        self.model_path.mkdir(parents=True, exist_ok=True)

        # Initialize MultiModalPredictor with problem-specific settings.
        # A conservative `time_limit` is set to ensure all ten fits finish
        # within the shared deadline, accounting for overhead.
        # `env.num_gpus=0` forces CPU training, enhancing robustness in environments
        # where GPU availability or setup might be inconsistent.
        # `env.num_workers=0` avoids multiprocessing issues that can occur in
        # containerized or constrained environments.
        self.predictor = MultiModalPredictor(
            label=self.target_column,
            problem_type=self.problem_type,
            eval_metric='rmse',  # Explicitly set RMSE as the evaluation metric
            path=str(self.model_path),
            verbosity=0  # Reduce log output to keep the console clean
        )

        # Define hyperparameters to control training behavior and resource usage.
        hyperparameters = {
            "env.num_gpus": 0,          # Force CPU training
            "env.num_workers": 0,       # Avoid multiprocessing issues
            "optim.max_epochs": 10,     # Set an upper bound for epochs to prevent excessively long runs
        }

        # Fit the predictor. The `time_limit` is the primary control for training duration.
        # 'medium_quality' preset offers a good balance between model performance and training time.
        # Adjusted time_limit to 150 seconds per fold to ensure all 10 folds finish within the total deadline.
        self.predictor.fit(
            train_data=train_data,
            time_limit=150,             # Allocate 150 seconds for fitting this fold (conservative for 1790s total)
            presets='medium_quality',   # Use a balanced preset for model configuration
            hyperparameters=hyperparameters
        )

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Generates predictions for the given input features using the fitted model.

        Args:
            X (pd.DataFrame): Features for which to generate predictions.

        Returns:
            pd.Series: Predicted continuous numerical values for 'SalePrice'.
        """
        test_data = X.copy()
        # Drop the 'Id' column if it exists, consistent with training.
        if 'Id' in test_data.columns:
            test_data = test_data.drop(columns=['Id'])

        # Ensure the predictor has been fitted before attempting to make predictions.
        if self.predictor is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit(X, y) first.")

        predictions = self.predictor.predict(test_data)
        return predictions


def build_model(X_train: pd.DataFrame, y_train: pd.Series) -> AutoGluonMultiModalWrapper:
    """
    Factory function to build and return a fresh, unfitted model wrapper.
    This function is called by the evaluation harness for each fold.

    Args:
        X_train (pd.DataFrame): Training features (provided for signature compatibility,
                                not used for fitting at this stage).
        y_train (pd.Series): Training labels (provided for signature compatibility,
                              not used for fitting at this stage).

    Returns:
        AutoGluonMultiModalWrapper: An unfitted model wrapper instance with `fit` and `predict` methods.
    """
    return AutoGluonMultiModalWrapper()