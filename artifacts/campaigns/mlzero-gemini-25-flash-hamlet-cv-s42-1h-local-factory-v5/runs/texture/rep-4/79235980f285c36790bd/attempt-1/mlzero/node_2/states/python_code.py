import pandas as pd
import numpy as np
import os
import tempfile
import shutil
from autogluon.tabular import TabularPredictor
import logging

# Configure logging to suppress excessive AutoGluon output during benchmark runs.
# Set to WARNING or ERROR to reduce console spam.
# For debugging, you might set it to INFO or DEBUG.
logging.getLogger('autogluon').setLevel(logging.WARNING)
logging.getLogger('autogluon.core').setLevel(logging.WARNING)
logging.getLogger('autogluon.tabular').setLevel(logging.WARNING)

class AutoGluonModelWrapper:
    """
    A wrapper class for AutoGluon's TabularPredictor to conform to the
    fit/predict interface required by the benchmark harness.
    """
    def __init__(self):
        self.predictor = None
        self.model_dir = None  # Path to the temporary directory for AutoGluon artifacts

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Fits the AutoGluon TabularPredictor model.

        All learned preprocessing is fitted within this method, using only
        the provided training rows (X_train, y_train).

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target labels.
        """
        # Create a unique temporary directory for AutoGluon artifacts for this specific fit call.
        # This ensures "fresh artifact directories per fold" as required.
        # Using a prefix helps identify these directories if they persist.
        self.model_dir = tempfile.mkdtemp(prefix="autogluon_model_")
        # print(f"AutoGluon model artifacts for this fold will be stored in: {self.model_dir}") # Optional debug print

        # Combine features and target into a single DataFrame required by AutoGluon.
        # Ensure a copy is made to avoid modifying the original DataFrame if it's a view.
        train_data = X_train.copy()
        train_data[y_train.name] = y_train  # Add the target column to the DataFrame

        # Initialize TabularPredictor.
        # label: The name of the target column, inferred from y_train.name.
        # eval_metric: 'balanced_accuracy' as specified in the task description.
        # path: The temporary directory for storing model artifacts for this fold.
        self.predictor = TabularPredictor(
            label=y_train.name,
            eval_metric='balanced_accuracy',
            path=self.model_dir,
            # verbosity=0 suppresses most AutoGluon output, 1 for warnings, 2 for info, 3 for debug.
            # Keeping it at default or 0 for benchmark runs is usually preferred.
            verbosity=0
        )

        # Fit the predictor.
        # seed=42 for reproducibility, as specified.
        # time_limit: Crucial for ensuring all ten fits finish inside the remaining shared deadline.
        #             The total time is 1087 seconds for 10 folds. Setting ~90 seconds per fold
        #             (900 seconds total) leaves a buffer for overhead. This value can be tuned.
        # presets: 'medium_quality_fast_inference' offers a good balance of performance
        #          and speed for benchmarks. 'best_quality' might be too slow for the time limit.
        self.predictor.fit(
            train_data,
            seed=42,
            time_limit=90,  # Example: 90 seconds per fold. Adjust as needed for total deadline.
            presets='medium_quality_fast_inference',
            # num_cpus and num_gpus can be set if specific resource limits are known,
            # otherwise AutoGluon defaults to using available resources.
        )

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Makes predictions using the fitted AutoGluon model.

        Args:
            X_test (pd.DataFrame): Test features.

        Returns:
            np.ndarray: A 1-dimensional NumPy array of predicted class labels,
                        matching the number of input samples.
        """
        if self.predictor is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        # AutoGluon's predict method for classification returns a pandas Series.
        # Convert it to a NumPy array as required by the prediction output format.
        predictions = self.predictor.predict(X_test).to_numpy()
        return predictions

    def __del__(self):
        """
        Destructor to clean up the temporary directory created for AutoGluon artifacts.
        This helps prevent accumulation of temporary files on the system.
        """
        if self.model_dir and os.path.exists(self.model_dir):
            try:
                shutil.rmtree(self.model_dir)
                # print(f"Cleaned up AutoGluon temporary directory: {self.model_dir}") # Optional debug print
            except OSError as e:
                # print(f"Error removing AutoGluon temporary directory {self.model_dir}: {e}") # Optional debug print
                pass


def build_model(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Factory function to generate a new, unfitted model wrapper for the classification task.

    This function adheres to the contract of returning a NEW, unfitted model
    with `fit(X,y)` and `predict(X)` methods.

    Args:
        X_train (pd.DataFrame): Training features (provided by harness, used for type hinting
                                and context, but not for model initialization here).
        y_train (pd.Series): Training target (provided by harness, used for type hinting
                             and context, but not for model initialization here).

    Returns:
        AutoGluonModelWrapper: An unfitted instance of the model wrapper, ready for fitting.
    """
    # The X_train and y_train arguments are provided by the harness to build_model,
    # but the contract specifies returning an *unfitted* model.
    # Therefore, we only instantiate the wrapper here.
    # The actual fitting and use of X_train/y_train data happens inside the wrapper's .fit() method.
    return AutoGluonModelWrapper()