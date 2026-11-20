import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor
import uuid
from pathlib import Path
import logging
import random

# Configure logging to suppress excessive AutoGluon output.
# This helps keep the console output clean during evaluation.
logging.getLogger('autogluon').setLevel(logging.WARNING)
logging.getLogger('autogluon.core').setLevel(logging.WARNING)
logging.getLogger('autogluon.tabular').setLevel(logging.WARNING)

# Attempt to import AutoGluon's set_seed utility for reproducibility.
# The path for set_seed has been unstable across AutoGluon versions,
# so a try-except block is used to handle potential ImportErrors.
_autogluon_set_seed = None
try:
    from autogluon.common.utils.utils import set_seed as _autogluon_set_seed_func
    _autogluon_set_seed = _autogluon_set_seed_func
except ImportError:
    try:
        from autogluon.core.utils.utils import set_seed as _autogluon_set_seed_func
        _autogluon_set_seed = _autogluon_set_seed_func
    except ImportError:
        logging.warning("AutoGluon's set_seed utility not found. Reproducibility might be affected.")

# Module-level constants
GLOBAL_SEED = 42
TARGET_COLUMN = 'SalePrice'
# Time limit for each fit call, in seconds.
# The total shared deadline is 2816 seconds for 10 folds.
# Setting 250 seconds per fold allows for 2500 seconds of fitting time
# across all folds, leaving 316 seconds for data loading, model initialization,
# and other overheads. This is a reasonable and conservative allocation.
FIT_TIME_LIMIT = 250

class AutoGluonModelWrapper:
    """
    A wrapper class for AutoGluon's TabularPredictor to conform to the
    fit/predict interface required by the evaluation harness.
    """
    def __init__(self):
        # Generate a unique path for AutoGluon artifacts for each model instance.
        # This ensures that each fold's training process uses a fresh, isolated
        # directory for storing models and intermediate files, preventing conflicts
        # and adhering to the "fresh artifact directories per fold" requirement.
        self.model_path = Path(f"./AutogluonModels/{uuid.uuid4()}")
        
        # Initialize TabularPredictor.
        # problem_type is 'regression' and eval_metric is 'rmse' as per task description.
        # The 'path' argument directs AutoGluon to store its artifacts in the unique directory.
        # Based on previous error guidance, 'random_seed' is NOT passed to the constructor.
        self.predictor = TabularPredictor(
            path=str(self.model_path),
            problem_type='regression',
            eval_metric='rmse'
        )
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fits the AutoGluon TabularPredictor on the provided training data.
        All learned preprocessing (e.g., feature engineering, imputation, encoding)
        is handled internally by AutoGluon and is fitted exclusively
        on the supplied fold training rows (X, y).
        """
        # Set global seeds for Python's random module and NumPy for reproducibility.
        # This is a robust approach given the conflicting AutoGluon seed errors
        # and ensures consistency across different parts of the ecosystem.
        random.seed(GLOBAL_SEED)
        np.random.seed(GLOBAL_SEED)
        # If AutoGluon's specific set_seed utility was successfully imported, use it.
        if _autogluon_set_seed:
            _autogluon_set_seed(GLOBAL_SEED)

        # Drop the 'Id' column if it exists. The contract explicitly states:
        # "Never add row IDs as features." 'errors='ignore'' prevents an error
        # if 'Id' is not present in a particular DataFrame.
        X_processed = X.drop(columns=['Id'], errors='ignore')

        # Combine features and target into a single DataFrame, which is the
        # expected input format for AutoGluon's fit method.
        train_data = X_processed.copy()
        train_data[TARGET_COLUMN] = y

        # Fit the predictor.
        # 'time_limit' is crucial for bounding each fit within the shared deadline.
        # 'presets='best_quality'' aims for high accuracy, which is generally
        # a good starting point for competitive performance.
        # Based on previous error guidance, 'seed' is NOT passed to the fit method.
        self.predictor.fit(
            train_data,
            label=TARGET_COLUMN,
            time_limit=FIT_TIME_LIMIT,
            presets='best_quality',
        )
        self.is_fitted = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generates predictions for the given input features using the fitted model.

        Args:
            X (pd.DataFrame): DataFrame containing features for which to make predictions.

        Returns:
            np.ndarray: A 1D NumPy array of numerical predictions (SalePrice).
        """
        if not self.is_fitted:
            raise RuntimeError("Model has not been fitted yet. Call .fit() before .predict().")

        # Drop the 'Id' column from the prediction features, consistent with training.
        X_processed = X.drop(columns=['Id'], errors='ignore')

        # Generate predictions using the AutoGluon predictor.
        predictions = self.predictor.predict(X_processed)

        # AutoGluon typically returns a pandas Series for predictions.
        # The contract requires a 1D array (e.g., NumPy array or list).
        return predictions.to_numpy()

def build_model(X_train: pd.DataFrame, y_train: pd.Series) -> AutoGluonModelWrapper:
    """
    Factory function to create a new, unfitted AutoGluon model wrapper.

    This function is called by the evaluation harness to obtain a fresh model
    instance for each fold. The provided X_train and y_train are for context
    and should NOT be used to fit the model at this stage. The returned model
    must be unfitted and ready for its .fit(X, y) method to be called later.

    Args:
        X_train (pd.DataFrame): Training features (provided for context, not used for fitting here).
        y_train (pd.Series): Training target (provided for context, not used for fitting here).

    Returns:
        AutoGluonModelWrapper: An unfitted model wrapper object with .fit() and .predict() methods.
    """
    return AutoGluonModelWrapper()