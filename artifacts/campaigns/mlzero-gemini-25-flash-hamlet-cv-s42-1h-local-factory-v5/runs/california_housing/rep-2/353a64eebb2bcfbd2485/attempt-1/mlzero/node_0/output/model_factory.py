import os
import uuid
import pandas as pd
from autogluon.tabular import TabularPredictor, TabularDataset

# --- Module-level constants and configurations ---
# Seed for reproducibility of stochastic models.
SEED = 42

# Total time for all ten folds is 3251 seconds.
# To ensure all ten fits finish within the shared deadline, we allocate
# approximately 300 seconds per fold, allowing for overhead.
TIME_LIMIT_PER_FOLD = 300

# The name of the target column as specified in the problem description.
TARGET_COLUMN = 'median_house_value'

# The problem type for AutoGluon TabularPredictor.
PROBLEM_TYPE = 'regression'

# The evaluation metric for AutoGluon. RMSE is the default for regression,
# but explicitly setting it is good practice.
EVAL_METRIC = 'root_mean_squared_error'

class AutoGluonTabularWrapper:
    """
    A wrapper class for AutoGluon's TabularPredictor to conform to the
    fit/predict interface required by the benchmark harness.
    """
    def __init__(self, output_dir: str):
        """
        Initializes the wrapper.

        Args:
            output_dir: A unique directory path where AutoGluon will store its artifacts.
                        This ensures fresh artifact directories per fold.
        """
        self.output_dir = output_dir
        self.predictor = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fits the AutoGluon TabularPredictor model.

        All learned preprocessing, model training, and internal validation
        are handled by AutoGluon within this method, using only the provided
        training data (X, y).

        Args:
            X: Training features (pandas DataFrame).
            y: Training target (pandas Series).
        """
        # Combine X and y into a single DataFrame as required by TabularPredictor.
        # AutoGluon handles feature type inference, missing values, and
        # other preprocessing steps automatically.
        train_data = X.copy()
        train_data[TARGET_COLUMN] = y

        # Initialize TabularPredictor.
        # The 'path' parameter ensures that each fold's model artifacts are
        # stored in a unique, fresh directory.
        self.predictor = TabularPredictor(
            label=TARGET_COLUMN,
            problem_type=PROBLEM_TYPE,
            eval_metric=EVAL_METRIC,
            path=self.output_dir,
            random_state=SEED  # Set random_state for reproducibility
        )

        # Fit the predictor.
        # 'presets="best_quality"' is chosen for a strong balance of accuracy
        # and training time. For datasets under 30K rows, 'extreme' could be
        # considered if a GPU is guaranteed and the time limit allows, but
        # 'best_quality' is generally robust.
        # 'time_limit' ensures the model training respects the per-fold deadline.
        self.predictor.fit(
            train_data=TabularDataset(train_data),
            time_limit=TIME_LIMIT_PER_FOLD,
            presets='best_quality',
            # AutoGluon automatically handles internal validation splits;
            # no need to manually provide tuning_data.
        )

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Makes predictions using the fitted AutoGluon TabularPredictor model.

        Args:
            X: Features to predict on (pandas DataFrame).

        Returns:
            A pandas Series of numerical predictions.
        """
        if self.predictor is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit(X, y) first.")

        # AutoGluon's predict method expects a TabularDataset.
        predictions = self.predictor.predict(TabularDataset(X))
        return predictions

def build_model(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Factory function to build a new, unfitted AutoGluon Tabular model wrapper.

    This function is called once per cross-validation fold by the harness.
    It returns a fresh, unfitted model instance.

    Args:
        X_train: A sample of training features (pandas DataFrame).
                 Provided for context, but not used for fitting in this function.
        y_train: A sample of training targets (pandas Series).
                 Provided for context, but not used for fitting in this function.

    Returns:
        An unfitted instance of AutoGluonTabularWrapper, exposing fit(X,y) and predict(X) methods.
    """
    # Generate a unique output directory for each model instance (i.e., per fold).
    # This is crucial to ensure that each fold's model artifacts are stored
    # separately and do not interfere with other folds.
    # os.getcwd() will be the unique output folder for the current attempt/node.
    output_dir = os.path.join(os.getcwd(), f"autogluon_model_{uuid.uuid4()}")
    return AutoGluonTabularWrapper(output_dir)