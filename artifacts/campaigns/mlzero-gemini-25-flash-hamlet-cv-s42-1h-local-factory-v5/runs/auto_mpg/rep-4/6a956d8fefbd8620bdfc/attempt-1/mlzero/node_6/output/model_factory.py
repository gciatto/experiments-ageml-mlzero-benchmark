import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor
import uuid
from pathlib import Path
import logging
import random # Import random for global seed setting

# Configure logging to suppress excessive AutoGluon output if desired.
# For this benchmark, default logging might be acceptable, but often
# setting a higher level can reduce console clutter.
# logging.getLogger('autogluon').setLevel(logging.WARNING)

# Module-level constants
SEED = 42

# The total remaining time for all ten fits and overhead is 1628 seconds.
# To ensure all ten fits finish within the deadline, we set a conservative
# time limit per fold. For 10 folds, 150 seconds per fold means 1500 seconds
# for fitting, leaving 128 seconds for other operations and overhead.
FIT_TIME_LIMIT_PER_FOLD = 150 # Adjusted based on remaining total time

class AutoGluonTabularWrapper:
    """
    A wrapper class for AutoGluon's TabularPredictor to conform to the
    fit/predict interface required by the benchmark harness.
    """
    def __init__(self, seed: int, output_path: Path):
        """
        Initializes the wrapper.

        Args:
            seed (int): Random seed for reproducibility.
            output_path (Path): Unique path for AutoGluon to store its artifacts
                                for this specific model instance (fold).
        """
        self.seed = seed
        self.output_path = output_path
        self.predictor: TabularPredictor = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fits the AutoGluon TabularPredictor model.

        All preprocessing and model training are performed within this method,
        using only the provided training data (X, y).

        Args:
            X (pd.DataFrame): Training features.
            y (pd.Series): Training target values.
        """
        # Set global seeds for reproducibility within this fit call.
        # This is a fallback due to AutoGluon's inconsistent random_seed parameter
        # across versions and its set_seed utility being problematic, as indicated
        # by previous error summaries.
        random.seed(self.seed)
        np.random.seed(self.seed)

        # Combine X and y into a single DataFrame for AutoGluon.
        # AutoGluon expects the target column to be present in the training DataFrame.
        train_df = X.copy()
        
        # Ensure y has a name; if not, assign a default.
        # The problem statement specifies 'class' as the target column name.
        label_column_name = y.name if y.name else 'class'
        train_df[label_column_name] = y

        # Initialize TabularPredictor.
        # Based on the latest error summaries, 'random_seed' is not accepted
        # in the TabularPredictor constructor for the installed AutoGluon version.
        # The path is unique for each fold to prevent data leakage and conflicts.
        self.predictor = TabularPredictor(
            label=label_column_name,
            problem_type='regression',
            path=str(self.output_path), # AutoGluon expects a string path
            # random_seed=self.seed # Removed as per latest error summary
        )

        # Fit the predictor with a time limit to ensure completion within the deadline.
        # AutoGluon handles all internal preprocessing (e.g., missing values,
        # categorical encoding) automatically during the fit process.
        # Based on the latest error summaries, 'random_seed' is not accepted
        # in the fit method either for the installed AutoGluon version.
        self.predictor.fit(
            train_data=train_df,
            time_limit=FIT_TIME_LIMIT_PER_FOLD,
            # For potentially better performance, presets='best_quality' could be used,
            # but it might increase training time. Sticking to default for now to
            # prioritize meeting the time limits.
            # ag_args_fit={'num_gpus': 0} can be added if GPU is available but not desired.
            # random_seed=self.seed # Removed as per latest error summary
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generates predictions for the given input features.

        Args:
            X (pd.DataFrame): Features to make predictions on.

        Returns:
            np.ndarray: A 1-dimensional array of predicted continuous values.
        """
        if self.predictor is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        
        predictions = self.predictor.predict(X)
        
        # AutoGluon's predict method returns a pandas Series or DataFrame.
        # The contract requires a 1-dimensional numpy array.
        return predictions.to_numpy()

def build_model(X_train: pd.DataFrame, y_train: pd.Series) -> AutoGluonTabularWrapper:
    """
    Factory function to create a new, unfitted model wrapper for AutoGluon.

    This function is called by the harness for each cross-validation fold.
    It must return a fresh, unfitted model instance.

    Args:
        X_train (pd.DataFrame): Training features for the current fold.
                                (Used here to infer column names/types, not for fitting).
        y_train (pd.Series): Training target for the current fold.
                             (Used here to infer target name/type, not for fitting).

    Returns:
        AutoGluonTabularWrapper: An unfitted model wrapper with `fit(X,y)` and `predict(X)` methods.
    """
    # Generate a unique output directory path for AutoGluon artifacts for this specific
    # model instance (representing one fold). This ensures "fresh artifact directories per fold"
    # and prevents conflicts or data leakage between folds.
    # The path is relative to the current working directory, which is the node output folder.
    unique_dir_name = f"autogluon_model_{uuid.uuid4()}"
    output_path = Path(unique_dir_name)

    return AutoGluonTabularWrapper(seed=SEED, output_path=output_path)