import pandas as pd
import numpy as np
import tempfile
import shutil
import os
import logging

from autogluon.tabular import TabularPredictor
from autogluon.core.metrics import make_scorer
from sklearn.metrics import balanced_accuracy_score

# --- Module-level constants and initialization ---

# Global seed for reproducibility. AutoGluon's internal components,
# especially with 'presets', often manage their own randomness.
# This seed can influence other parts of the code or environment.
GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED) # Seed numpy for any potential internal use

# Target column name as specified in the problem description.
TARGET_COLUMN_NAME = 'class'

# Define a custom scorer for balanced accuracy, as specified in the task description.
# AutoGluon's TabularPredictor can optimize for this metric during fitting.
ag_balanced_accuracy_scorer = make_scorer(
    name='balanced_accuracy',
    score_func=balanced_accuracy_score,
    optimum=1,  # The best possible score for balanced accuracy is 1
    greater_is_better=True,  # Higher balanced accuracy is better
    needs_class=True  # This metric requires predicted class labels, not probabilities
)

# Configure logging for the module, especially for cleanup operations.
# This helps in debugging issues with temporary directory removal.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Model Wrapper Class ---

class AutoGluonTabularModel:
    """
    A wrapper class for AutoGluon's TabularPredictor, conforming to the
    fit/predict interface required by the evaluation harness.
    """
    def __init__(self, target_column_name: str, time_limit_per_fit: int, seed: int):
        """
        Initializes the wrapper.

        Args:
            target_column_name (str): The name of the target column.
            time_limit_per_fit (int): Maximum time in seconds for each AutoGluon fit call.
            seed (int): Random seed for reproducibility.
        """
        self.target_column_name = target_column_name
        self.time_limit_per_fit = time_limit_per_fit
        self.seed = seed # Store seed, though AG's direct use is limited by API
        self.predictor: TabularPredictor = None
        self.temp_dir: str = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fits the AutoGluon TabularPredictor model.

        All preprocessing and model training are performed within this method,
        using only the provided training data (X, y). A unique temporary
        directory is created for AutoGluon's artifacts for each fit call.

        Args:
            X (pd.DataFrame): Training features.
            y (pd.Series): Training target labels.
        """
        # Combine features and target into a single DataFrame as required by AutoGluon
        train_data = X.copy()
        train_data[self.target_column_name] = y

        # Create a unique temporary directory for AutoGluon artifacts for this fold.
        # This ensures "fresh artifact directories per fold" and prevents conflicts.
        # Using a prefix helps identify these directories.
        self.temp_dir = tempfile.mkdtemp(prefix='autogluon_model_')
        logger.info(f"Created temporary directory for AutoGluon artifacts: {self.temp_dir}")

        # Initialize TabularPredictor.
        # The 'random_seed' argument is not supported in the constructor.
        self.predictor = TabularPredictor(
            label=self.target_column_name,
            path=self.temp_dir,
            eval_metric=ag_balanced_accuracy_scorer, # Optimize for balanced accuracy
            # AutoGluon automatically infers the problem type (classification in this case)
            # and handles feature engineering.
        )

        # Fit the predictor with a time limit and 'best_quality' preset.
        # 'best_quality' is generally recommended for high accuracy and is a good balance
        # for the given time constraints, as 'extreme' can be very slow or require GPU.
        # The 'random_state' argument is not supported in the fit method.
        # For reproducibility with 'best_quality' preset, AutoGluon typically
        # manages internal randomness.
        self.predictor.fit(
            train_data,
            time_limit=self.time_limit_per_fit,
            presets='best_quality',
            # ag_args_fit={'seed': self.seed} could be used for some models,
            # but 'best_quality' preset might override or have its own seeding.
            # Sticking to the simpler approach as per previous error fixes.
        )
        logger.info(f"AutoGluon predictor fitted in {self.temp_dir}")


    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Makes predictions using the fitted AutoGluon TabularPredictor.

        Args:
            X (pd.DataFrame): Features for which to make predictions.

        Returns:
            np.ndarray: A 1D NumPy array of predicted class labels.
        """
        if self.predictor is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit(X, y) first.")

        # AutoGluon's predict method returns a Series, convert to numpy array as required.
        predictions = self.predictor.predict(X)
        return predictions.to_numpy()

    def __del__(self):
        """
        Cleans up the temporary directory created for AutoGluon artifacts
        when the model object is garbage collected.
        """
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
                logger.info(f"Successfully removed temporary directory: {self.temp_dir}")
            except OSError as e:
                # Log the error if the directory cannot be removed.
                # This can happen if files are still in use or permissions issues.
                logger.error(f"Error removing temporary directory {self.temp_dir}: {e}")

# --- Factory Function ---

def build_model(X_train: pd.DataFrame, y_train: pd.Series) -> AutoGluonTabularModel:
    """
    Factory function to create a new, unfitted AutoGluonTabularModel wrapper.

    Args:
        X_train (pd.DataFrame): Training features (not used for building, but part of signature).
        y_train (pd.Series): Training target labels (not used for building, but part of signature).

    Returns:
        AutoGluonTabularModel: An unfitted model wrapper instance.
    """
    # The total remaining shared deadline is 476 seconds for 10 folds.
    # Allocating 40 seconds per fold (40 * 10 = 400 seconds) leaves a buffer
    # for overhead and other operations, ensuring all fits finish within the deadline.
    # This is a tight budget, so 'best_quality' preset is chosen over 'extreme'.
    time_limit_per_fit = 40

    return AutoGluonTabularModel(TARGET_COLUMN_NAME, time_limit_per_fit, GLOBAL_SEED)