import pandas as pd
from autogluon.multimodal import MultiModalPredictor
import uuid
import logging
import os
from pathlib import Path
import numpy as np
import random
import warnings

# Filter warnings to reduce noise in output, especially from libraries like AutoGluon
warnings.filterwarnings('ignore')

# Set up logging for the module. This helps in debugging during execution.
logger = logging.getLogger(__name__)
# Configure basic logging to stdout, which is typically captured by the evaluation harness.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Module-level constants and initialization ---
# The name of the target column as specified in the problem description.
TARGET_COLUMN = 'class'
# The random seed to ensure reproducibility across runs and folds.
SEED = 42

# Set global random seeds for numpy and the standard 'random' module.
# As per previous error feedback, AutoGluon's MultiModalPredictor in this environment
# does not accept a 'random_seed' argument in its constructor or fit method directly.
# Relying on global seeds is the recommended approach for reproducibility in this context.
np.random.seed(SEED)
random.seed(SEED)

# Define a base directory for AutoGluon's model artifacts.
# Each model instance (corresponding to a single cross-validation fold) will
# store its data in a unique subdirectory under this base path.
# Using '/tmp' is a standard practice for temporary files in Linux environments,
# and the evaluation harness is expected to handle cleanup or isolation.
BASE_ARTIFACT_DIR = Path("/tmp/autogluon_mm_models")
# Ensure the base directory exists. `exist_ok=True` prevents an error if the
# directory already exists from a previous run or another process.
BASE_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

class AutoGluonMMWrapper:
    """
    A wrapper class for AutoGluon MultiModalPredictor.
    This wrapper conforms to the `fit(X, y)` and `predict(X)` interface required
    by the evaluation harness, encapsulating the MultiModalPredictor's lifecycle
    for a single cross-validation fold.
    """
    def __init__(self, model_path: Path):
        """
        Initializes the wrapper for a new model instance.

        Args:
            model_path: The unique file system path where AutoGluon will store
                        all its temporary and permanent artifacts for this
                        specific model instance (i.e., for one CV fold).
        """
        self.model_path = model_path
        self.predictor = None  # The MultiModalPredictor instance, initialized in .fit()
        self.problem_type = 'classification' # Task type as specified in the problem
        logger.info(f"AutoGluonMMWrapper initialized with model_path: {self.model_path}")

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fits the AutoGluon MultiModalPredictor using the provided training data.
        All necessary preprocessing, model selection, and training are performed
        within this method, using only the data from the current fold.

        Args:
            X: A pandas DataFrame containing the training features for the current fold.
            y: A pandas Series containing the training target labels for the current fold.
        """
        logger.info(f"Starting fit for model at path: {self.model_path}")

        # AutoGluon's MultiModalPredictor.fit() method expects a single DataFrame
        # that includes both features and the target column.
        train_data = X.copy()
        train_data[TARGET_COLUMN] = y

        # Instantiate the MultiModalPredictor.
        # - `label`: Specifies the name of the target column.
        # - `path`: Crucially, this ensures that each model instance (per fold)
        #   uses a fresh, isolated directory for its artifacts, preventing data leakage
        #   or interference between folds. AutoGluon expects a string path.
        # - `problem_type`: Explicitly set to 'classification' for this task.
        # - `random_seed`: Removed from constructor as per previous error feedback.
        #   Global `np.random.seed` and `random.seed` are used instead.
        # - `verbosity`: Controls the amount of logging output from AutoGluon itself.
        #   1 shows some progress, 0 is silent, 4 is debug-level verbose.
        self.predictor = MultiModalPredictor(
            label=TARGET_COLUMN,
            path=str(self.model_path),
            problem_type=self.problem_type,
            verbosity=1,
        )

        # Fit the predictor.
        # The `time_limit` parameter is critical for managing the total execution
        # time across all ten cross-validation folds. Given a total deadline of
        # ~1209 seconds for 10 folds, 100 seconds per fold is a conservative estimate.
        # This allows for overhead and potential variance in training times, aiming
        # to ensure all fits complete within the deadline (10 folds * 100s = 1000s,
        # leaving ~200s for other operations).
        fit_time_limit = 100 # seconds
        logger.info(f"Fitting AutoGluon MultiModalPredictor with time_limit={fit_time_limit}s "
                    f"for problem_type='{self.problem_type}'")
        self.predictor.fit(
            train_data=train_data,
            time_limit=fit_time_limit,
            # No `random_seed` parameter here, as per previous error feedback for TabularPredictor
            # and the general guidance to rely on global seeds for this environment.
            # For purely tabular data, MultiModalPredictor automatically infers
            # column types and selects appropriate internal models. No specific
            # `presets` or `hyperparameters` are explicitly set here, relying
            # on AutoGluon's robust defaults for efficiency and performance.
        )
        logger.info(f"Fit complete for model at path: {self.model_path}")

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Generates predictions using the fitted AutoGluon MultiModalPredictor.

        Args:
            X: A pandas DataFrame containing the features for which to make predictions.

        Returns:
            A pandas Series containing the predicted class labels for each row in X.
        """
        # Ensure the model has been fitted before attempting to make predictions.
        if self.predictor is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        logger.info(f"Making predictions for model at path: {self.model_path}")
        predictions = self.predictor.predict(X)
        return predictions

def build_model(X_train: pd.DataFrame, y_train: pd.Series) -> AutoGluonMMWrapper:
    """
    Factory function required by the evaluation harness.
    This function is called for each cross-validation fold to provide a new,
    unfitted model instance.

    Args:
        X_train: Training features for the current fold. Provided for context
                 (e.g., to infer schema or column names), but not used for
                 fitting at this stage.
        y_train: Training target labels for the current fold. Provided for context,
                 but not used for fitting at this stage.

    Returns:
        An unfitted instance of `AutoGluonMMWrapper`, ready for its `.fit()`
        method to be called with the actual training data for the fold.
    """
    # Generate a unique identifier for this specific model instance.
    # This UUID is used to create a distinct directory for AutoGluon's artifacts,
    # ensuring complete isolation between models trained for different folds.
    unique_id = uuid.uuid4().hex
    model_instance_path = BASE_ARTIFACT_DIR / f"autogluon_mm_model_{unique_id}"
    logger.info(f"build_model called, creating new AutoGluonMMWrapper with unique path: {model_instance_path}")
    return AutoGluonMMWrapper(model_instance_path)