import pandas as pd
import numpy as np
import os
import uuid
import shutil
import random
# The autogluon.multimodal library internally uses deep learning frameworks
# like PyTorch. While MultiModalPredictor aims to manage reproducibility,
# explicitly setting global seeds for Python's random module and NumPy
# at the start of the fit method helps ensure consistency across environments
# and addresses potential issues with specific AutoGluon versions not
# recognizing 'env.seed' in hyperparameters.
# import torch # Uncomment if direct torch seeding is found necessary

from autogluon.multimodal import MultiModalPredictor

# Global seed for reproducibility. This seed will be used to initialize
# Python's random module and NumPy's random number generator within each
# fold's fit method.
GLOBAL_SEED = 42

class AutoGluonMultiModalWrapper:
    """
    A wrapper class for AutoGluon's MultiModalPredictor to conform to the
    fit/predict interface required by the evaluation harness.
    """
    def __init__(self):
        self.predictor = None
        self.model_path = None
        self.label_column_name = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fits the AutoGluon MultiModalPredictor model using the provided training data.

        All preprocessing and model fitting are performed within this method,
        using only the supplied training rows.

        Args:
            X (pd.DataFrame): Training features.
            y (pd.Series): Training target.
        """
        # Set seeds for reproducibility within this fold's training session.
        # This ensures that any random operations, including those potentially
        # initiated by AutoGluon's internal components, are seeded.
        random.seed(GLOBAL_SEED)
        np.random.seed(GLOBAL_SEED)
        # If AutoGluon's MultiModalPredictor does not fully propagate the seed
        # to PyTorch, uncommenting the following line might be necessary:
        # import torch
        # torch.manual_seed(GLOBAL_SEED)

        self.label_column_name = y.name
        if self.label_column_name is None:
            # Fallback if y.name is not set, though the contract implies it will be 'SalePrice'.
            self.label_column_name = "target" 

        # Combine X and y into a single DataFrame, which is the expected input format for AutoGluon.
        train_df = X.copy()
        train_df[self.label_column_name] = y

        # Drop the 'Id' column if it exists, as it's an identifier and not a predictive feature.
        if 'Id' in train_df.columns:
            train_df = train_df.drop(columns=['Id'])
        
        # Create a unique path for this model's artifacts for the current fold.
        # os.getcwd() typically points to a temporary directory unique to the current fold's execution.
        # A subdirectory 'autogluon_mm_models' is created within it, and then a UUID-named folder.
        base_output_dir = os.path.join(os.getcwd(), "autogluon_mm_models")
        os.makedirs(base_output_dir, exist_ok=True)
        self.model_path = os.path.join(base_output_dir, str(uuid.uuid4()))

        # Initialize MultiModalPredictor for a regression task.
        # The 'seed' argument is not passed to the constructor, as per previous error fixes.
        self.predictor = MultiModalPredictor(
            label=self.label_column_name,
            problem_type='regression',
            path=self.model_path,
        )

        # Fit the predictor.
        # The global seed is handled by manual seeding at the start of this method.
        # A time_limit is set to ensure each fold's training completes within the overall deadline.
        # With a total remaining time of 2538 seconds for 10 folds, an average of 253.8 seconds
        # per fold is available. Setting time_limit to 200 seconds provides a safe buffer
        # for overhead and other operations.
        self.predictor.fit(
            train_data=train_df,
            time_limit=200, # seconds per fit call, adjusted for safety
            # The 'hyperparameters={"env.seed": GLOBAL_SEED}' argument is removed
            # from the fit method call, as per the ERROR_SUMMARY indicating
            # 'env.seed' is not a valid hyperparameter for MultiModalPredictor.fit.
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Makes predictions using the fitted AutoGluon MultiModalPredictor model.

        Args:
            X (pd.DataFrame): Features for making predictions.

        Returns:
            np.ndarray: A 1D NumPy array of predicted numerical values.
        """
        if self.predictor is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit(X, y) first.")
        
        # Prepare prediction data: drop the 'Id' column if it's present.
        X_pred = X.copy()
        if 'Id' in X_pred.columns:
            X_pred = X_pred.drop(columns=['Id'])
        
        # AutoGluon MultiModalPredictor is generally robust to column order and
        # handles missing/extra columns by using only the features it was trained on.
        
        predictions = self.predictor.predict(X_pred)
        # Convert the pandas Series output from AutoGluon to a 1D NumPy array
        # as required by the prediction output format.
        return predictions.to_numpy()

    def __del__(self):
        """
        Attempts to clean up the model artifact directory when the wrapper object is deleted.
        This helps manage disk space, especially in environments with many folds.
        Note: Python's __del__ method is not guaranteed to be called reliably.
        """
        if self.model_path and os.path.exists(self.model_path):
            try:
                shutil.rmtree(self.model_path)
            except OSError as e:
                # Print a warning if cleanup fails, but do not block execution.
                print(f"Warning: Could not remove model directory {self.model_path}: {e}")


def build_model(X_train: pd.DataFrame, y_train: pd.Series) -> AutoGluonMultiModalWrapper:
    """
    Factory function to create and return a new, unfitted model wrapper.
    This function is called by the evaluation harness for each fold.

    Args:
        X_train (pd.DataFrame): Example training features. This argument is provided
                                for context but is not used for fitting within this function.
        y_train (pd.Series): Example training target. This argument is provided
                             for context but is not used for fitting within this function.

    Returns:
        AutoGluonMultiModalWrapper: An unfitted model wrapper instance, ready to be
                                    fitted via its .fit(X, y) method.
    """
    return AutoGluonMultiModalWrapper()