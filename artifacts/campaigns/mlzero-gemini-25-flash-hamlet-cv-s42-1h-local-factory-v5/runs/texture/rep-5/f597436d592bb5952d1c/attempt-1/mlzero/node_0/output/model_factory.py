import pandas as pd
from autogluon.tabular import TabularPredictor
import os
import uuid
import shutil
import logging

# Configure logging for the module to provide insights during execution
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Module-level Constants and Configuration ---
# Target column name as specified in the problem description
TARGET_COLUMN = 'Class'

# Evaluation metric as specified in the problem description
EVAL_METRIC = 'balanced_accuracy'

# Random seed for reproducibility of stochastic processes within AutoGluon
RANDOM_SEED = 42

# Time limit for each individual model fit operation in seconds.
# The total shared deadline is 3462 seconds for 10 folds.
# Allocating 300 seconds per fold (300 * 10 = 3000 seconds) leaves a buffer
# for overhead and other operations. This value can be adjusted if fits
# consistently time out or finish too quickly.
TIME_LIMIT_PER_FIT = 300


class AutoGluonTabularWrapper:
    """
    A wrapper class for autogluon.tabular.TabularPredictor to conform to the
    fit/predict interface required by the benchmark harness.
    """
    def __init__(self, label, eval_metric, random_state, time_limit_per_fit):
        """
        Initializes the wrapper. The AutoGluon TabularPredictor is not fitted here.

        Args:
            label (str): The name of the target column.
            eval_metric (str): The evaluation metric to optimize for.
            random_state (int): Seed for reproducibility.
            time_limit_per_fit (int): Maximum time in seconds allowed for each fit call.
        """
        self.label = label
        self.eval_metric = eval_metric
        self.random_state = random_state
        self.time_limit_per_fit = time_limit_per_fit
        self.predictor = None  # Will hold the fitted TabularPredictor instance
        self.model_path = None  # Unique path for AutoGluon artifacts for this model instance

    def fit(self, X, y):
        """
        Fits the AutoGluon TabularPredictor model using the provided training data.

        Args:
            X (pd.DataFrame): Training features.
            y (pd.Series): Training target labels.
        """
        # Combine features and target into a single DataFrame as expected by AutoGluon
        train_data = X.copy()
        train_data[self.label] = y

        # Generate a unique directory for AutoGluon's artifacts for this specific fold.
        # This ensures "fresh artifact directories per fold" as required.
        self.model_path = f"autogluon_models_{uuid.uuid4()}"
        logger.info(f"Fitting AutoGluon model. Artifacts will be stored at: {self.model_path}")

        # Initialize TabularPredictor.
        # verbosity=0 suppresses most AutoGluon output during benchmark runs.
        # random_state ensures reproducibility.
        self.predictor = TabularPredictor(
            label=self.label,
            eval_metric=self.eval_metric,
            path=self.model_path,
            verbosity=0,
            random_state=self.random_state
        )

        # Fit the predictor.
        # 'best_quality' preset is chosen for high performance, as suggested by AutoGluon docs.
        # time_limit bounds the training time for each fold.
        # calibrate_decision_threshold=True is beneficial for metrics like balanced_accuracy
        # in classification tasks.
        self.predictor.fit(
            train_data,
            presets='best_quality',
            time_limit=self.time_limit_per_fit,
            calibrate_decision_threshold=True
        )
        logger.info(f"AutoGluon model fitted successfully. Leaderboard summary:\n{self.predictor.leaderboard(silent=True)}")

    def predict(self, X):
        """
        Makes predictions on new data using the fitted AutoGluon model.

        Args:
            X (pd.DataFrame): Features for which to make predictions.

        Returns:
            pd.Series: Predicted class labels.
        """
        if self.predictor is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit(X, y) first.")
        
        # AutoGluon's predict method expects a DataFrame
        return self.predictor.predict(X)

    def __del__(self):
        """
        Attempts to clean up the temporary directory created by AutoGluon.
        This is a best-effort cleanup and helps manage disk space in long-running benchmarks.
        """
        if self.model_path and os.path.exists(self.model_path):
            try:
                shutil.rmtree(self.model_path)
                logger.info(f"Cleaned up AutoGluon model directory: {self.model_path}")
            except OSError as e:
                logger.error(f"Error removing AutoGluon model directory {self.model_path}: {e}")


def build_model(X_train, y_train):
    """
    Factory function to generate a new, unfitted model wrapper for AutoGluon Tabular.

    This function is called by the benchmark harness for each fold. It must return
    a fresh, unfitted model instance. The actual fitting will occur when the
    harness calls the .fit(X, y) method on the returned object.

    Args:
        X_train (pd.DataFrame): Training features (provided by harness, but not
                                used for fitting at this stage as the model is unfitted).
        y_train (pd.Series): Training target (provided by harness, but not
                             used for fitting at this stage).

    Returns:
        AutoGluonTabularWrapper: An unfitted model wrapper instance with
                                 .fit(X, y) and .predict(X) methods.
    """
    logger.info("Building a new unfitted AutoGluonTabularWrapper instance.")
    return AutoGluonTabularWrapper(
        label=TARGET_COLUMN,
        eval_metric=EVAL_METRIC,
        random_state=RANDOM_SEED,
        time_limit_per_fit=TIME_LIMIT_PER_FIT
    )