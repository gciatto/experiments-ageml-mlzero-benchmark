import pandas as pd
import numpy as np
import tempfile
import shutil
import os
import random
from autogluon.tabular import TabularPredictor
from autogluon.core.metrics import make_scorer
from sklearn.metrics import balanced_accuracy_score

# Global seed for reproducibility across the module.
# Setting a global seed for standard Python 'random' and 'numpy'
# can influence other parts of the code or environment if applicable,
# ensuring broader reproducibility. AutoGluon's internal components,
# especially with 'presets', often manage their own randomness.
GLOBAL_SEED = 42
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)

# Define the target column name as specified in the task description.
TARGET_COLUMN_NAME = 'class'

# Define a custom scorer for balanced accuracy, as specified in the task description.
# AutoGluon's TabularPredictor can optimize for this metric during fitting.
# The 'needs_class=True' is crucial for classification metrics that operate on predicted labels.
ag_balanced_accuracy_scorer = make_scorer(
    name='balanced_accuracy',
    score_func=balanced_accuracy_score,
    optimum=1,  # The best possible score for balanced accuracy is 1
    greater_is_better=True,  # Higher balanced accuracy is better
    needs_class=True  # This metric requires predicted class labels, not probabilities
)

class AutoGluonTabularModel:
    """
    A wrapper class for AutoGluon's TabularPredictor, conforming to the
    fit/predict interface required by the evaluation harness.
    """
    def __init__(self, target_column_name: str, time_limit_per_fit: int):
        """
        Initializes the wrapper.

        Args:
            target_column_name (str): The name of the target column.
            time_limit_per_fit (int): Maximum time in seconds for each AutoGluon fit call.
        """
        self.target_column_name = target_column_name
        self.time_limit_per_fit = time_limit_per_fit
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
        # Using a prefix can help identify these directories if debugging.
        self.temp_dir = tempfile.mkdtemp(prefix='autogluon_tmp_')

        # Initialize TabularPredictor.
        # As per previous errors, 'random_seed' is not a valid argument for the constructor.
        self.predictor = TabularPredictor(
            label=self.target_column_name,
            path=self.temp_dir,
            eval_metric=ag_balanced_accuracy_scorer, # Optimize for balanced accuracy
            # AutoGluon automatically infers the problem type (classification in this case)
            # and handles feature engineering.
            verbosity=0 # Reduce log output for cleaner console in benchmark environment
        )

        # Fit the predictor with a time limit and 'extreme' preset.
        # The 'extreme' preset is recommended for small datasets (<30K samples)
        # and aims for cutting-edge performance. Given the small dataset size (wilt),
        # 'extreme' is a suitable choice.
        # As per previous errors, 'random_state' is not a valid argument for the fit method.
        self.predictor.fit(
            train_data,
            time_limit=self.time_limit_per_fit,
            presets='extreme',
            # For reproducibility with 'extreme' preset, AutoGluon typically
            # manages internal randomness. Explicit seeding for all internal
            # models would require more granular control via ag_args_fit,
            # which is often handled by AutoGluon's internal mechanisms when presets are used.
        )

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
            except OSError as e:
                # In a production environment, proper logging would be used.
                # For this context, a print statement is acceptable for debugging.
                print(f"Error removing temporary directory {self.temp_dir}: {e}")


def build_model(X_train: pd.DataFrame, y_train: pd.Series) -> AutoGluonTabularModel:
    """
    Factory function to create a new, unfitted AutoGluonTabularModel wrapper.

    Args:
        X_train (pd.DataFrame): Training features (not used for building, but part of signature).
        y_train (pd.Series): Training target labels (not used for building, but part of signature).

    Returns:
        AutoGluonTabularModel: An unfitted model wrapper instance.
    """
    # The total remaining shared deadline is 1647 seconds for 10 folds.
    # Allocating 150 seconds per fold (150 * 10 = 1500 seconds) leaves a buffer
    # of 147 seconds for overhead and other operations, ensuring all fits finish
    # inside the remaining shared deadline. This is a conservative estimate for
    # the 'extreme' preset, balancing performance with deadline constraints.
    time_limit_per_fit = 150

    return AutoGluonTabularModel(TARGET_COLUMN_NAME, time_limit_per_fit)