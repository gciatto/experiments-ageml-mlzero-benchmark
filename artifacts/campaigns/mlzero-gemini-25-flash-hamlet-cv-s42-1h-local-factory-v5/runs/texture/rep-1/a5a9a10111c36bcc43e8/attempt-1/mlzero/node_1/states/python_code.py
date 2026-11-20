import os
import numpy as np
import pandas as pd
import logging
import tempfile
import shutil
from autogluon.tabular import TabularPredictor
from autogluon.common.utils.utils import set_seed

# Set a global seed for reproducibility for AutoGluon and other libraries.
# This ensures consistent results across runs when using random processes.
SEED = 42
set_seed(SEED)
np.random.seed(SEED) # For general numpy operations if any are used outside AutoGluon

# Configure logging to suppress excessive AutoGluon output during fitting.
# This helps keep the console output clean during benchmark runs, focusing on essential information.
logging.getLogger('autogluon').setLevel(logging.WARNING)
logging.getLogger('autogluon.core').setLevel(logging.WARNING)
logging.getLogger('autogluon.tabular').setLevel(logging.WARNING)

class AutoGluonModelWrapper:
    """
    A wrapper class for AutoGluon's TabularPredictor to conform to the
    fit/predict interface required by the evaluation harness.
    """
    def __init__(self):
        self.predictor = None
        self.model_path = None
        self.target_column = 'Class' # The target column name as specified in the task description

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Fits the AutoGluon TabularPredictor on the provided training data.
        All necessary preprocessing, feature engineering, and model training
        are handled internally by AutoGluon.
        """
        # Create a unique temporary directory for AutoGluon artifacts for this specific fit call.
        # This ensures "fresh artifact directories per fold" as required, preventing conflicts
        # between different cross-validation folds or subsequent runs.
        self.model_path = tempfile.mkdtemp(prefix="autogluon_model_")
        
        # Combine features (X_train) and target (y_train) into a single DataFrame,
        # which is the expected input format for AutoGluon's TabularPredictor.
        train_data = X_train.copy()
        train_data[self.target_column] = y_train

        # Initialize TabularPredictor.
        # The 'random_state' argument is explicitly removed from the constructor
        # as per previous error feedback. Global reproducibility is managed by
        # autogluon.common.utils.utils.set_seed(SEED) at the module level.
        self.predictor = TabularPredictor(
            label=self.target_column,
            path=self.model_path,
            eval_metric='balanced_accuracy', # The metric specified in the README.md
            verbosity=0 # Suppress verbose output during initialization and fitting
        )

        # Fit the predictor with a time limit.
        # The total remaining search time is 3396 seconds for 10 folds.
        # Setting a time_limit of 250 seconds per fit ensures that all ten fits
        # have a good chance to finish within the shared deadline, leaving sufficient
        # time for data loading, model initialization, prediction, and evaluation overhead.
        self.predictor.fit(
            train_data,
            time_limit=250, # seconds per fit call
            presets='medium_quality_fast_inference', # A balanced preset for benchmarks,
                                                     # offering good quality within reasonable time.
            # Other presets like 'best_quality', 'high_quality', 'good_quality_fast_inference'
            # could be considered if more time is available or higher accuracy is the sole priority.
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predicts class labels for the given input features using the fitted model.
        """
        if self.predictor is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        
        # AutoGluon's predict method typically returns a pandas Series.
        # The contract requires a 1-dimensional NumPy array or Python list.
        predictions = self.predictor.predict(X)
        
        return predictions.to_numpy()

    # Optional: A __del__ method could be added here to clean up the temporary
    # AutoGluon model directory (self.model_path) when the object is garbage collected.
    # However, for benchmark environments, it's often left to the harness or the
    # operating system to manage temporary files. Removing it to simplify the
    # module and avoid potential conflicts with the harness's lifecycle management.
    # def __del__(self):
    #     if self.model_path and os.path.exists(self.model_path):
    #         try:
    #             shutil.rmtree(self.model_path)
    #         except OSError as e:
    #             # Log the error if cleanup fails, but do not raise an exception
    #             pass


def build_model(X_train: pd.DataFrame, y_train: pd.Series) -> AutoGluonModelWrapper:
    """
    Factory function to create a new, unfitted AutoGluon model wrapper.
    This function is called once per cross-validation fold by the evaluation harness.
    The X_train and y_train arguments are provided to indicate the expected data
    format, but the model is returned unfitted.
    """
    return AutoGluonModelWrapper()