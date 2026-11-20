import pandas as pd
import numpy as np
import os
import random
import uuid
import logging

# Set global random seeds for reproducibility across runs and within AutoGluon's components.
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# AutoGluon uses its own logging. Set verbosity to 0 for minimal output during benchmark.
# This prevents excessive console output from AutoGluon's internal operations.
logging.getLogger('autogluon').setLevel(logging.WARNING) # Suppress INFO and DEBUG messages

# Constants for the task, as specified in the problem description.
TARGET_COLUMN = 'class'
EVAL_METRIC = 'balanced_accuracy'

# Time limit per fit call. The total time for 10 folds is 340 seconds.
# This means an average of 34 seconds per fold, including all overhead (data loading,
# model instantiation, prediction, evaluation by the harness).
# We allocate a conservative 25 seconds for the actual AutoGluon fit process
# to leave room for other operations and ensure all folds complete within the deadline.
# If this proves too slow, 'medium' preset or a lower time_limit would be necessary.
TIME_LIMIT_PER_FIT = 25 # seconds

class AutoGluonTabularWrapper:
    """
    A wrapper for autogluon.tabular.TabularPredictor that conforms to the
    build_model(X_train, y_train) -> unfitted_model.fit(X,y).predict(X) contract.

    This class encapsulates the AutoGluon TabularPredictor, handling its
    initialization, fitting, and prediction steps according to the benchmark's
    requirements.
    """
    def __init__(self, label, eval_metric, random_state, time_limit_per_fit):
        """
        Initializes the wrapper with task-specific parameters.

        Args:
            label (str): The name of the target column.
            eval_metric (str): The evaluation metric for AutoGluon to optimize.
            random_state (int): Seed for reproducibility.
            time_limit_per_fit (int): Maximum time in seconds allowed for the fit operation.
        """
        self.label = label
        self.eval_metric = eval_metric
        self.random_state = random_state
        self.time_limit_per_fit = time_limit_per_fit
        self.predictor = None
        self.model_path = None # Path where AutoGluon saves its artifacts for this specific fit

    def fit(self, X, y):
        """
        Fits the AutoGluon TabularPredictor model.

        All preprocessing and model training are handled internally by AutoGluon
        and are fitted solely on the provided training data (X, y).

        Args:
            X (pd.DataFrame or np.ndarray): Training features.
            y (pd.Series or np.ndarray): Training labels.
        """
        # Import TabularPredictor here to ensure it's loaded only when fit is called,
        # and to keep the module scope cleaner if AG is heavy.
        from autogluon.tabular import TabularPredictor, TabularDataset

        # AutoGluon requires a unique path for each predictor instance to store its
        # models and temporary files. Using a UUID ensures uniqueness across different
        # folds and benchmark runs, preventing conflicts.
        # The harness ensures a clean working directory for each fold, but AutoGluon
        # itself needs distinct paths for its internal operations if multiple predictors
        # were to be instantiated in the same process (though not the case here).
        self.model_path = f"AG_model_{uuid.uuid4().hex}"
        
        # Combine X and y into a single DataFrame, which is the expected input format
        # for AutoGluon's TabularDataset.
        # Ensure X is a DataFrame and y is a Series for consistent behavior.
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y, name=self.label, index=X.index) # Preserve index if X has one

        train_data = X.copy()
        train_data[self.label] = y

        # Initialize TabularPredictor.
        # verbosity=0 suppresses most AutoGluon output, which is beneficial for benchmarks.
        self.predictor = TabularPredictor(
            label=self.label,
            eval_metric=self.eval_metric,
            path=self.model_path,
            verbosity=0,
            # AutoGluon automatically infers the problem_type (e.g., 'binary', 'multiclass')
            # from the target column data, so explicit setting is usually not needed.
        )
        
        # Fit the predictor.
        # 'presets='good'' is chosen as a balance between speed and accuracy.
        # 'medium' is faster but less accurate, while 'best' or 'extreme' are
        # typically much slower and might exceed the time_limit without a GPU.
        # The 'time_limit' parameter is crucial for ensuring the model training
        # completes within the allocated time per fold.
        self.predictor.fit(
            TabularDataset(train_data),
            time_limit=self.time_limit_per_fit,
            presets='good',
            # AutoGluon internally manages random states for its constituent models.
            # The global seeds set at the module level (np.random.seed, random.seed)
            # should influence its stochastic components for reproducibility.
        )

    def predict(self, X):
        """
        Makes predictions using the fitted AutoGluon TabularPredictor model.

        Args:
            X (pd.DataFrame or np.ndarray): Features for which to make predictions.

        Returns:
            np.ndarray: Predicted values.
        """
        if self.predictor is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit(X, y) first.")
        
        # Ensure X is a DataFrame for AutoGluon's predict method.
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Make predictions. AutoGluon's predict method returns a pandas Series.
        predictions = self.predictor.predict(TabularDataset(X))
        
        # Return predictions as a numpy array, as expected by the harness.
        return predictions.values

def build_model(X_train, y_train):
    """
    Factory function to build a new, unfitted AutoGluon Tabular model wrapper.

    This function is called by the benchmark harness for each fold. It should
    return a fresh, unfitted model instance.

    Args:
        X_train (pd.DataFrame): Training features for the current fold.
                                (Not used for fitting at this stage, but can be
                                used for model initialization if needed, e.g.,
                                to infer feature types or target cardinality).
        y_train (pd.Series): Training labels for the current fold.
                             (Not used for fitting at this stage).

    Returns:
        AutoGluonTabularWrapper: An unfitted model wrapper instance that exposes
                                 `fit(X,y)` and `predict(X)` methods.
    """
    # X_train and y_train are provided by the harness but should not be used to fit
    # the model at this stage. They are merely for context if the model needs
    # to know something about the data shape/types before fitting.
    # Our wrapper is generic enough not to need them at instantiation, so we
    # simply pass the predefined constants.
    return AutoGluonTabularWrapper(
        label=TARGET_COLUMN,
        eval_metric=EVAL_METRIC,
        random_state=SEED,
        time_limit_per_fit=TIME_LIMIT_PER_FIT
    )