import pandas as pd
import numpy as np
import os
import shutil
import datetime
import uuid

from autogluon.tabular import TabularPredictor

# Module-level constants or helper objects can be defined here.
# For this task, the default AutoGluon RMSE metric is sufficient.

class AutoGluonTabularWrapper:
    """
    A wrapper class for AutoGluon's TabularPredictor to conform to the
    fit/predict interface required by the evaluation harness.
    """
    def __init__(self, random_seed: int = 42, time_limit_per_fold: int = 200):
        """
        Initializes the wrapper with configuration for AutoGluon.

        Args:
            random_seed: Seed for reproducibility of stochastic elements.
            time_limit_per_fold: Maximum time in seconds allowed for fitting
                                 the model in each fold.
        """
        self.predictor: TabularPredictor = None
        self.random_seed = random_seed
        self.time_limit_per_fold = time_limit_per_fold
        self.model_path: str = None # Path where AutoGluon stores its models for the current fold

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fits the AutoGluon TabularPredictor model.

        Args:
            X: Training features.
            y: Training target variable.
        """
        # Ensure X and y are pandas DataFrames/Series and handle target column name
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            # Ensure the target column is named 'SalePrice' as per problem description
            # and that its index aligns with X.
            y = pd.Series(y, name='SalePrice', index=X.index)

        # Drop 'Id' column if present, as it should not be used as a feature.
        X_processed = X.drop(columns=['Id'], errors='ignore')

        # Combine X and y into a single DataFrame for AutoGluon
        train_data = X_processed.copy()
        train_data[y.name] = y

        # Generate a unique path for this fold's model artifacts.
        # This ensures "fresh artifact directories per fold" even if multiple
        # folds are processed within the same Python process.
        # AutoGluon creates a directory at this path.
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8] # Short unique ID
        self.model_path = f"autogluon_models_{timestamp}_{unique_id}"

        # Initialize TabularPredictor
        # problem_type='regression' and eval_metric='root_mean_squared_error' are explicit.
        # The 'random_state' argument is not accepted in the constructor,
        # it should be passed as 'seed' to the .fit() method.
        self.predictor = TabularPredictor(
            label=y.name,
            problem_type='regression',
            eval_metric='root_mean_squared_error', # Metric for the task (RMSE)
            path=self.model_path, # Store models in a unique path
        )

        # Fit the predictor with specified time limit and presets.
        # 'best' preset is chosen for a good balance of accuracy and training time
        # on CPU environments, considering the shared deadline.
        # Pass the random_seed to the fit method as 'seed' for reproducibility.
        self.predictor.fit(
            train_data,
            time_limit=self.time_limit_per_fold,
            presets='best', # Using 'best' preset for high quality
            seed=self.random_seed, # Set random seed for reproducibility in fit method
        )

        # AutoGluon stores model artifacts in self.model_path.
        # These files are necessary for `predict` to function.
        # The overall experiment harness is assumed to manage the cleanup
        # of the output directory for each fold.

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Makes predictions using the fitted AutoGluon TabularPredictor model.

        Args:
            X: Features for which to make predictions.

        Returns:
            An array of predicted values.
        """
        if self.predictor is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit(X, y) first.")

        # Drop 'Id' column if present, consistent with fitting.
        X_processed = X.drop(columns=['Id'], errors='ignore')

        # Make predictions. AutoGluon returns a pandas Series by default for regression.
        predictions = self.predictor.predict(X_processed)
        return predictions.values # Return as numpy array as typically expected by evaluation harnesses

def build_model(X_train: pd.DataFrame, y_train: pd.Series) -> AutoGluonTabularWrapper:
    """
    Factory function to create a new, unfitted AutoGluonTabularWrapper model.

    Args:
        X_train: Training features (not used for initialization, but part of contract).
        y_train: Training target (not used for initialization, but part of contract).

    Returns:
        A fresh, unfitted instance of AutoGluonTabularWrapper.
    """
    # The total shared deadline is 2292 seconds for ten-fold cross-validation.
    # This allocates approximately 229.2 seconds per fold.
    # We set the time_limit_per_fold for AutoGluon's fit method to 200 seconds,
    # leaving a buffer for data loading, Python overhead, and other operations.
    return AutoGluonTabularWrapper(random_seed=42, time_limit_per_fold=200)