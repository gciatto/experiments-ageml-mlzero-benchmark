import pandas as pd
from autogluon.tabular import TabularPredictor
from pathlib import Path
import uuid
import logging

# Configure logging to suppress excessive AutoGluon output if desired.
# For benchmark environments, it's often useful to see some progress,
# but too much verbosity can clutter logs.
# logging.getLogger('autogluon').setLevel(logging.WARNING)

# --- Module-level Constants and Configuration ---
# Seed for reproducibility of stochastic elements in AutoGluon.
RANDOM_SEED = 42

# Total time limit for all ten folds is 3420 seconds.
# We allocate 300 seconds per fold for fitting, leaving 420 seconds
# for overall harness overhead and potential variations.
TIME_LIMIT_PER_FOLD = 300


class AutoGluonTabularWrapper:
    """
    A wrapper class for AutoGluon TabularPredictor to conform to the
    `fit(X, y)` and `predict(X)` interface required by the benchmark harness.
    """

    def __init__(self, path: Path, time_limit: int, random_state: int):
        """
        Initializes the wrapper with configuration for the AutoGluon predictor.

        Args:
            path (Path): A unique directory path where AutoGluon will store its
                         models and artifacts for this specific fold.
            time_limit (int): The maximum time in seconds allowed for the
                              AutoGluon `fit` operation.
            random_state (int): The random seed for reproducibility.
        """
        self.path = path
        self.time_limit = time_limit
        self.random_state = random_state
        self.predictor: TabularPredictor = None  # Will be initialized in fit

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fits the AutoGluon TabularPredictor model using the provided training data.
        All data preprocessing, feature engineering, model selection, and
        hyperparameter tuning are handled automatically by AutoGluon within
        this method, using only the supplied `X` and `y`.

        Args:
            X (pd.DataFrame): Training features.
            y (pd.Series): Training target variable.
        """
        # Combine X and y into a single DataFrame as required by TabularPredictor.
        # AutoGluon will automatically identify the target column by its name.
        train_data = X.copy()
        train_data[y.name] = y

        # Initialize TabularPredictor.
        # - `label=y.name`: Specifies the target column.
        # - `path=str(self.path)`: Ensures a fresh, unique artifact directory per fold.
        # - `eval_metric='rmse'`: Sets the evaluation metric for model selection.
        # - `random_state`: Ensures reproducibility.
        self.predictor = TabularPredictor(
            label=y.name,
            path=str(self.path),  # AutoGluon expects a string path
            eval_metric='rmse',  # Root Mean Squared Error for regression
            random_state=self.random_state
        )

        # Fit the predictor.
        # - `time_limit`: Bounds the training time for each fold.
        # - `presets='best'`: Utilizes AutoGluon's best quality preset for high accuracy,
        #                     which includes advanced techniques like bagging and stacking.
        # - `verbosity=1`: Shows some progress updates without being overly verbose.
        self.predictor.fit(
            train_data=train_data,
            time_limit=self.time_limit,
            presets='best',
            verbosity=1  # Set to 0 for silent, 2 for more detailed logs
        )

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Makes predictions using the fitted AutoGluon TabularPredictor.

        Args:
            X (pd.DataFrame): Features for which to make predictions.

        Returns:
            pd.Series: Predicted target values.
        """
        if self.predictor is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit(X, y) first.")

        # AutoGluon automatically handles feature transformation consistent
        # with the training phase.
        predictions = self.predictor.predict(X)
        return predictions


def build_model(X_train: pd.DataFrame, y_train: pd.Series) -> AutoGluonTabularWrapper:
    """
    Factory function to create a new, unfitted AutoGluon Tabular model wrapper.
    This function is called once per cross-validation fold by the harness.

    Args:
        X_train (pd.DataFrame): Training features (used here only to infer target name).
        y_train (pd.Series): Training target (used here only to infer target name).

    Returns:
        AutoGluonTabularWrapper: An unfitted model wrapper instance ready for training.
    """
    # Generate a unique path for AutoGluon artifacts for each fold.
    # This is critical to ensure isolation and prevent interference between
    # models trained on different folds. `uuid.uuid4()` guarantees uniqueness.
    model_dir = Path(f"autogluon_model_{uuid.uuid4()}")

    # Instantiate the wrapper with the unique path and configured parameters.
    model_wrapper = AutoGluonTabularWrapper(
        path=model_dir,
        time_limit=TIME_LIMIT_PER_FOLD,
        random_state=RANDOM_SEED
    )
    return model_wrapper