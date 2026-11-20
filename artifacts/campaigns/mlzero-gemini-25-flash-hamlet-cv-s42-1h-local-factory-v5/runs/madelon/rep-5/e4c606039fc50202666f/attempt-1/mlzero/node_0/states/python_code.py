import pandas as pd
from autogluon.tabular import TabularPredictor, TabularDataset
import tempfile
import shutil
import os
import logging

# Configure logging to suppress excessive AutoGluon output during training.
# The verbosity parameter in TabularPredictor's constructor is also set to 0.
logging.getLogger('autogluon').setLevel(logging.WARNING)
logging.getLogger('autogluon.core').setLevel(logging.WARNING)
logging.getLogger('autogluon.tabular').setLevel(logging.WARNING)

# Helper function to set random seed for AutoGluon's internal operations.
# This ensures reproducibility for each fold's training.
def set_autogluon_seed(seed: int):
    """Sets the random seed for AutoGluon's internal operations."""
    from autogluon.common.utils.utils import set_random_seed
    set_random_seed(seed)

class AutoGluonTabularWrapper:
    """
    A wrapper class for AutoGluon's TabularPredictor, providing a scikit-learn-like
    fit/predict interface and handling artifact management.
    """
    def __init__(self, label_column: str, random_seed: int, time_limit_per_fit: int):
        """
        Initializes the wrapper. The AutoGluon predictor is not fitted here.

        Args:
            label_column (str): The name of the target column.
            random_seed (int): The random seed for reproducibility.
            time_limit_per_fit (int): Maximum time in seconds allowed for each fit call.
        """
        self.label_column = label_column
        self.random_seed = random_seed
        self.time_limit_per_fit = time_limit_per_fit
        self.predictor: TabularPredictor = None
        self.save_path: str = None # Path to store AutoGluon's model artifacts

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fits the AutoGluon TabularPredictor model.

        All learned preprocessing and model fitting occur within this method,
        using only the provided training data (X, y). A unique temporary
        directory is created for each fit call to store AutoGluon's artifacts.

        Args:
            X (pd.DataFrame): Training features.
            y (pd.Series): Training target labels.
        """
        # Set the random seed for AutoGluon for reproducibility for this fold
        set_autogluon_seed(self.random_seed)

        # Combine X and y into a single DataFrame as required by AutoGluon
        train_data = X.copy()
        train_data[self.label_column] = y

        # Create a unique temporary directory for this fold's AutoGluon artifacts.
        # This ensures "fresh artifact directories per fold" as required.
        self.save_path = tempfile.mkdtemp(prefix="autogluon_model_")
        
        # Initialize TabularPredictor.
        # 'balanced_accuracy' is specified as the metric in the problem description.
        # 'best_quality' preset is chosen for a good balance of accuracy and training time
        # within the given per-fold time limit.
        self.predictor = TabularPredictor(
            label=self.label_column,
            eval_metric='balanced_accuracy',
            path=self.save_path,
            verbosity=0 # Suppress verbose output during training
        )

        # Fit the predictor with the specified time limit and presets.
        self.predictor.fit(
            train_data,
            time_limit=self.time_limit_per_fit,
            presets='best_quality',
        )

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Makes predictions using the fitted AutoGluon TabularPredictor model.

        Args:
            X (pd.DataFrame): Features for which to make predictions.

        Returns:
            pd.Series: Predicted class labels.
        """
        if self.predictor is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit(X, y) first.")
        
        # AutoGluon's predict method can directly take a pandas DataFrame.
        predictions = self.predictor.predict(X)
        return predictions

    def __del__(self):
        """
        Attempts to clean up the temporary directory created for AutoGluon artifacts
        when the wrapper object is garbage collected. This is a best-effort cleanup.
        """
        if self.save_path and os.path.exists(self.save_path):
            try:
                shutil.rmtree(self.save_path)
            except OSError as e:
                # Log the error but do not raise, as __del__ should not fail.
                logging.warning(f"Error cleaning up AutoGluon temporary directory {self.save_path}: {e}")

def build_model(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Factory function to build a new, unfitted AutoGluon Tabular model wrapper.

    This function adheres to the contract: it returns a NEW, unfitted model
    with `fit(X,y)` and `predict(X)` methods.

    Args:
        X_train (pd.DataFrame): Training features (provided for context, not used for fitting here).
        y_train (pd.Series): Training target (provided for context, not used for fitting here).

    Returns:
        AutoGluonTabularWrapper: An unfitted model wrapper instance.
    """
    # The target column name is explicitly 'Class' as per the problem description.
    label_column = 'Class'
    
    # The total shared deadline is ~3474 seconds for 10 folds.
    # Allocating 300 seconds per fold for AutoGluon's fit allows for ~47 seconds
    # of overhead per fold (e.g., data loading, wrapper instantiation, cleanup).
    time_limit_per_fit = 300 
    
    # Instantiate the wrapper. It is unfitted at this point.
    return AutoGluonTabularWrapper(
        label_column=label_column,
        random_seed=42, # Seed 42 as required
        time_limit_per_fit=time_limit_per_fit
    )