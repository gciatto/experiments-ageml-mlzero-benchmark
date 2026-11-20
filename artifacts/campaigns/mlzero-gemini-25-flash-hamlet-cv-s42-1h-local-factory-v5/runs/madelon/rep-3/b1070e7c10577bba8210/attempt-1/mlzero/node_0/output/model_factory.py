import os
import uuid
import random
import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor

# Set global seeds for reproducibility.
# AutoGluon also has its own internal seed parameter.
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Base directory for AutoGluon model artifacts.
# The harness runs model_factory.py in the node output folder,
# so creating a subdirectory here is appropriate.
AUTOGLUON_ARTIFACTS_BASE_DIR = "autogluon_models_artifacts"
os.makedirs(AUTOGLUON_ARTIFACTS_BASE_DIR, exist_ok=True)

# Total time limit for all 10 folds is 3431 seconds.
# We need to allocate this time across 10 fits, plus overhead.
# Let's aim for approximately 80% of the average time per fold for AutoGluon's fit,
# leaving 20% for data loading, wrapper instantiation, prediction, evaluation, etc.
TOTAL_BENCHMARK_TIME_SECONDS = 3431
NUM_CV_FOLDS = 10
TIME_LIMIT_PER_FIT_SECONDS = max(60, int(TOTAL_BENCHMARK_TIME_SECONDS / NUM_CV_FOLDS * 0.8))


class AutoGluonTabularWrapper:
    """
    A wrapper class for AutoGluon's TabularPredictor to conform to the
    fit/predict interface required by the benchmark harness.
    """
    def __init__(self, label_column: str, time_limit: int, seed: int, output_dir_base: str):
        self.label_column = label_column
        self.time_limit = time_limit
        self.seed = seed
        self.output_dir_base = output_dir_base
        self.predictor: TabularPredictor = None
        self.model_path: str = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fits the AutoGluon TabularPredictor model.
        All preprocessing and model training are handled internally by AutoGluon.
        """
        # AutoGluon's TabularPredictor expects features and target in a single DataFrame.
        train_data = X.copy()
        train_data[self.label_column] = y

        # Create a unique path for each fold's model artifacts.
        # This ensures "fresh artifact directories per fold" as required.
        unique_id = str(uuid.uuid4())
        self.model_path = os.path.join(self.output_dir_base, f"autogluon_model_{unique_id}")

        # Initialize TabularPredictor for binary classification.
        # 'balanced_accuracy' is the specified evaluation metric.
        # 'presets="best"' is chosen for high quality within the time limit,
        # as 'extreme' often requires GPU and significantly more time, which might not
        # be available in a general benchmark environment.
        self.predictor = TabularPredictor(
            label=self.label_column,
            problem_type='binary',  # Madelon is a binary classification task
            eval_metric='balanced_accuracy',
            path=self.model_path,
            verbosity=0,  # Suppress AutoGluon output for cleaner logs
            seed=self.seed, # Set seed for internal AutoGluon operations
        )

        # Fit the predictor with the allocated time limit.
        self.predictor.fit(
            train_data,
            time_limit=self.time_limit,
            presets='best', # Aim for high quality, robust for CPU environments
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Makes predictions using the fitted AutoGluon TabularPredictor.
        """
        if self.predictor is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit(X, y) first.")
        
        # AutoGluon's predict method returns a pandas Series.
        # Convert to numpy array as evaluation harnesses often expect this format.
        predictions = self.predictor.predict(X)
        return predictions.values


def build_model(X_train: pd.DataFrame, y_train: pd.Series) -> AutoGluonTabularWrapper:
    """
    Factory function to create a new, unfitted AutoGluonTabularWrapper model.

    Args:
        X_train (pd.DataFrame): Training features (not used for fitting here,
                                 but provides context for target column name).
        y_train (pd.Series): Training target (not used for fitting here,
                               but provides context for target column name).

    Returns:
        AutoGluonTabularWrapper: An unfitted model wrapper with fit/predict methods.
    """
    # The target column name is 'Class' as per the problem description.
    label_column = 'Class'

    return AutoGluonTabularWrapper(
        label_column=label_column,
        time_limit=TIME_LIMIT_PER_FIT_SECONDS,
        seed=RANDOM_SEED,
        output_dir_base=AUTOGLUON_ARTIFACTS_BASE_DIR
    )