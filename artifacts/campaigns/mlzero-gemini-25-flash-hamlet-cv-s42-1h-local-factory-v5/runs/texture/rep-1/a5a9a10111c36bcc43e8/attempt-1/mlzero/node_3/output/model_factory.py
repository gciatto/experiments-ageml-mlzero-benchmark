import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor
import tempfile
import shutil
import os
import logging
import uuid
from autogluon.common.utils.utils import set_seed # Corrected import for set_seed based on latest error

# Configure logging to suppress excessive AutoGluon output and other noisy libraries.
# This helps keep the console output clean during evaluation.
logging.basicConfig(level=logging.WARNING)
logging.getLogger('autogluon').setLevel(logging.WARNING)
logging.getLogger('lightning').setLevel(logging.ERROR)  # Suppress PyTorch Lightning logs
logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)  # Suppress PyTorch Lightning logs
logging.getLogger('numba').setLevel(logging.WARNING) # Suppress Numba warnings

class ModelWrapper:
    """
    A wrapper class for autogluon.tabular.TabularPredictor that conforms to the
    fit/predict interface required by the evaluation harness.
    """
    def __init__(self, label_column: str, random_seed: int = 42):
        self.label_column = label_column
        self.random_seed = random_seed
        self.predictor: TabularPredictor = None
        self.model_path: str = None  # Path where the AutoGluon model artifacts are stored

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Fits the AutoGluon TabularPredictor model.
        All preprocessing and model training are handled internally by AutoGluon,
        using only the provided X_train and y_train for this specific fold.
        """
        # Combine X_train and y_train into a single DataFrame for AutoGluon.
        train_data = X_train.copy()
        train_data[self.label_column] = y_train

        # Create a unique temporary directory for AutoGluon artifacts for this fold.
        # This ensures isolation and prevents conflicts between different folds,
        # fulfilling the "fresh artifact directories per fold" requirement.
        self.model_path = tempfile.mkdtemp(prefix=f"autogluon_model_{uuid.uuid4().hex}_")

        # Set global seed for reproducibility before initializing and fitting AutoGluon.
        # This addresses the `random_state` error by using the recommended `set_seed` utility.
        set_seed(self.random_seed)

        # Initialize TabularPredictor.
        # - label: The name of the target column.
        # - eval_metric: 'balanced_accuracy' as specified in the problem description.
        # - path: The temporary directory for model artifacts.
        # - verbosity: 0 to suppress most AutoGluon output during fitting.
        # The `random_state` argument is removed from the constructor as it's invalid.
        self.predictor = TabularPredictor(
            label=self.label_column,
            eval_metric='balanced_accuracy',
            path=self.model_path,
            verbosity=0,
            # random_state=self.random_seed # Removed as per error fix: invalid argument
        )

        # Fit the predictor.
        # - time_limit: Set to 300 seconds per fold. With 10 folds, this allocates
        #               3000 seconds for fitting, leaving ample time for overhead
        #               within the total shared deadline of 3205 seconds.
        # - presets: 'extreme' as requested by the library guidance.
        # AutoGluon automatically handles internal validation splits and feature engineering.
        self.predictor.fit(
            train_data,
            time_limit=300,  # Allocate 300 seconds per fold for fitting
            presets='extreme',  # Using 'extreme' preset as per guidance
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Makes predictions using the fitted AutoGluon TabularPredictor.
        """
        if self.predictor is None:
            raise RuntimeError("Model has not been fitted yet. Call fit(X, y) first.")
        
        # AutoGluon's predict method returns class labels directly for classification.
        # Convert the predictions to a NumPy array as required by the output format.
        predictions = self.predictor.predict(X)
        return predictions.to_numpy()

    def __del__(self):
        """
        Cleans up the temporary directory created for AutoGluon model artifacts.
        This method is called when the ModelWrapper object is garbage collected.
        While __del__ is not guaranteed to be called in all circumstances,
        it's a reasonable attempt at cleanup for temporary resources.
        """
        if self.model_path and os.path.exists(self.model_path):
            try:
                shutil.rmtree(self.model_path)
            except OSError as e:
                # Log a warning if cleanup fails, but don't block execution.
                logging.warning(f"Error removing temporary directory {self.model_path}: {e}")

def build_model(X_train: pd.DataFrame, y_train: pd.Series) -> ModelWrapper:
    """
    Factory function to create a new, unfitted ModelWrapper instance.
    This function is called by the evaluation harness for each fold.
    
    Args:
        X_train (pd.DataFrame): Training features (used to infer label column name).
        y_train (pd.Series): Training target (used to infer label column name).
        
    Returns:
        ModelWrapper: An unfitted model wrapper ready for training.
    """
    # Infer the label column name from the y_train Series.
    # This is more robust than hardcoding 'Class' in case the harness renames it.
    label_column = y_train.name
    if label_column is None:
        # Fallback if y_train is an unnamed Series.
        # The problem description states 'Class' is the target column.
        logging.warning("y_train Series has no name. Assuming label column is 'Class'.")
        label_column = 'Class'
    
    return ModelWrapper(label_column=label_column, random_seed=42)