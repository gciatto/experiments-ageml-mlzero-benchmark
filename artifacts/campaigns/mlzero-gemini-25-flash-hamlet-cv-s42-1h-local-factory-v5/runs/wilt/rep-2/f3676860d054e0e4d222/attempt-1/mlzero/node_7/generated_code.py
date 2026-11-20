import pandas as pd
from autogluon.tabular import TabularPredictor
import os
import tempfile
import shutil
import logging

# Configure logging to suppress excessive output from AutoGluon and related libraries.
# This is a module-level initialization, allowed by the prompt.
logging.getLogger('autogluon').setLevel(logging.WARNING)
logging.getLogger('lightning').setLevel(logging.WARNING)  # Often used by AutoGluon for NN models
logging.getLogger('numba').setLevel(logging.WARNING)      # Often used by AutoGluon for some operations
logging.getLogger('matplotlib').setLevel(logging.WARNING) # Sometimes used by AutoGluon for plotting
logging.getLogger('gluoncv').setLevel(logging.WARNING)    # If any vision models are used
logging.getLogger('mxnet').setLevel(logging.WARNING)      # If MXNet backend is used
logging.getLogger('torch').setLevel(logging.WARNING)      # If PyTorch backend is used (e.g., NN_TORCH models)


class AutoGluonTabularWrapper:
    """
    A wrapper class for autogluon.tabular.TabularPredictor to conform to the
    fit/predict interface required by the benchmark harness.
    """
    def __init__(self, label_column: str, random_seed: int, time_limit_per_fit: int):
        """
        Initializes the wrapper with configuration for TabularPredictor.

        Args:
            label_column: The name of the target column.
            random_seed: Seed for reproducibility.
            time_limit_per_fit: Maximum time in seconds allowed for each fit call.
        """
        self.label_column = label_column
        self.random_seed = random_seed
        self.time_limit_per_fit = time_limit_per_fit
        self.predictor = None
        self.model_path = None  # Path to store AutoGluon artifacts for the current model

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fits the AutoGluon TabularPredictor model.

        All preprocessing and model training are handled internally by AutoGluon
        and are fitted only on the provided training data (X, y).

        Args:
            X: Training features.
            y: Training target labels.
        """
        # AutoGluon's TabularPredictor expects features and labels in a single DataFrame.
        train_data = X.copy()
        train_data[self.label_column] = y

        # Create a temporary directory for AutoGluon artifacts for this specific fold.
        # This ensures "fresh artifact directories per fold" and isolates models.
        # The prefix helps identify these directories.
        self.model_path = tempfile.mkdtemp(prefix="autogluon_wilt_model_")

        # Initialize TabularPredictor.
        # - label: The target column name.
        # - eval_metric: 'balanced_accuracy' is specified in the README.md.
        # - path: The temporary directory for model artifacts.
        # - verbosity: Set to 0 to minimize console output during training.
        # The `random_seed` argument is NOT passed to the constructor, as it's not supported.
        self.predictor = TabularPredictor(
            label=self.label_column,
            eval_metric='balanced_accuracy',
            path=self.model_path,
            verbosity=0,
        )

        # Fit the predictor.
        # - presets='medium': A faster preset compared to 'best_quality',
        #   chosen to attempt to meet the extremely tight time limits.
        #   The prompt suggests 'extreme', but it requires a GPU and is likely too slow.
        #   'medium' is a reasonable compromise for speed on CPU.
        # - time_limit: Bounds the training time for each fold, crucial for
        #   meeting the shared deadline across all ten folds.
        # - ag_args_fit: This dictionary is the correct way to pass `random_seed`
        #   and other AutoGluon-specific arguments to the fit process.
        self.predictor.fit(
            train_data,
            presets='medium', # Changed from 'best_quality' for significantly faster training
            time_limit=self.time_limit_per_fit,
            ag_args_fit={'random_seed': self.random_seed}, # Corrected way to pass random_seed
        )

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Makes predictions using the fitted AutoGluon TabularPredictor model.

        Args:
            X: Features for which to make predictions.

        Returns:
            A Pandas Series containing the predicted class labels.
        """
        if self.predictor is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit(X, y) first.")

        # AutoGluon's predict method expects a DataFrame of features.
        predictions = self.predictor.predict(X)
        return predictions

    def __del__(self):
        """
        Cleans up the temporary directory created for AutoGluon model artifacts.
        This helps manage disk space, especially in environments with many folds.
        """
        if self.model_path and os.path.exists(self.model_path):
            try:
                shutil.rmtree(self.model_path)
            except OSError:
                # Ignore errors during cleanup, as the directory might already be gone
                # or permissions issues might occur in some environments.
                pass


def build_model(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Factory function to create a new, unfitted model wrapper for AutoGluon Tabular.

    Args:
        X_train: Training features (provided for context, not used for initialization).
        y_train: Training target labels (provided for context, not used for initialization).

    Returns:
        An unfitted instance of AutoGluonTabularWrapper with fit/predict methods.
    """
    # The target column name is 'class' as specified in the task description.
    label_column = 'class'
    # Seed for reproducibility across AutoGluon's internal processes.
    random_seed = 42
    # The remaining total search time is 54 seconds for 10 folds.
    # To ensure all ten fits finish inside this shared deadline, we must
    # drastically limit the time for each individual fit.
    # Allocating 4 seconds per fold for AutoGluon's fit method.
    # This leaves 1.4 seconds per fold for overhead (data loading, wrapper instantiation,
    # predict calls, cleanup, and harness overhead). This is extremely tight.
    time_limit_per_fit = 4

    return AutoGluonTabularWrapper(label_column, random_seed, time_limit_per_fit)