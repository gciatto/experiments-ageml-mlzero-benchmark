import os
import pandas as pd
from autogluon.tabular import TabularPredictor
import uuid
import logging

# Set the global random seed for AutoGluon reproducibility.
# This must be done at the module level, before any AutoGluon components are initialized.
# As per previous errors, 'seed' is not accepted in TabularPredictor.fit() and
# 'random_state' is not accepted in MultiModalPredictor constructor.
# For TabularPredictor, AG_TABULAR_RANDOM_SEED environment variable is the correct
# way to ensure global reproducibility.
os.environ['AG_TABULAR_RANDOM_SEED'] = '42'

# Optional: Configure logging to control AutoGluon's verbosity.
# By default, AutoGluon can be quite verbose. Setting to WARNING or ERROR can reduce output.
# For this task, it's often fine to let it log, but if excessive output causes issues,
# uncommenting and adjusting the level below can be helpful.
# logging.basicConfig(level=logging.WARNING)


class AutoGluonTabularWrapper:
    """
    A wrapper class for AutoGluon's TabularPredictor to conform to the
    fit/predict interface required by the harness.
    """
    def __init__(self, unique_model_path: str, eval_metric: str, time_limit: int):
        """
        Initializes the AutoGluonTabularWrapper.

        Args:
            unique_model_path (str): A unique file system path where AutoGluon
                                     will store its models and artifacts for this instance.
                                     This ensures isolation between different folds.
            eval_metric (str): The evaluation metric AutoGluon should optimize for.
            time_limit (int): The maximum time in seconds AutoGluon should spend fitting.
        """
        self.model_path = unique_model_path
        self.predictor = None
        self.eval_metric = eval_metric
        self.time_limit = time_limit
        # The target column name is explicitly 'Class' as per the problem description.
        self.label_column = 'Class'

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fits the AutoGluon TabularPredictor using the provided training data.
        All learned preprocessing is fitted within this method, using only
        the training rows provided for the current fold.

        Args:
            X (pd.DataFrame): Training features.
            y (pd.Series): Training target labels.
        """
        # Ensure the unique model path exists before AutoGluon tries to write to it.
        os.makedirs(self.model_path, exist_ok=True)

        # AutoGluon's TabularPredictor expects features and target combined
        # into a single DataFrame.
        train_data = X.copy()
        train_data[self.label_column] = y

        # Initialize TabularPredictor.
        # The 'path' argument ensures artifacts are stored in the unique directory
        # created for this specific model instance (fold).
        self.predictor = TabularPredictor(
            label=self.label_column,
            path=self.model_path,
            eval_metric=self.eval_metric,
            # verbosity=0 # Uncomment to suppress AutoGluon output during fit
        )

        # Fit the predictor.
        # 'time_limit' is crucial for managing execution time per fold, ensuring
        # all ten fits finish within the shared deadline.
        # 'presets='best_quality'' aims for high performance.
        # The global random seed is already set via the AG_TABULAR_RANDOM_SEED
        # environment variable at the module level.
        self.predictor.fit(
            train_data,
            time_limit=self.time_limit,
            presets='best_quality',
        )

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Generates predictions for the given input features.

        Args:
            X (pd.DataFrame): Features to make predictions on.

        Returns:
            pd.Series: Predicted class labels.
        """
        if self.predictor is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit(X, y) first.")

        # AutoGluon's predict method expects a DataFrame.
        predictions = self.predictor.predict(X)
        return predictions


def build_model(X_train: pd.DataFrame, y_train: pd.Series) -> AutoGluonTabularWrapper:
    """
    Factory function to create a new, unfitted AutoGluon model wrapper.

    This function is called by the harness for each cross-validation fold.
    It returns an unfitted model instance that exposes `fit(X,y)` and `predict(X)` methods.

    Args:
        X_train (pd.DataFrame): Training features for the current fold.
                                (Provided by harness, not used for fitting here).
        y_train (pd.Series): Training target for the current fold.
                             (Provided by harness, not used for fitting here).

    Returns:
        AutoGluonTabularWrapper: An unfitted model wrapper instance ready for training.
    """
    # Generate a unique path for AutoGluon artifacts for this specific model instance (fold).
    # This ensures that each fold's model training artifacts are isolated and do not
    # interfere with other folds or previous runs.
    # The 'autogluon_model_artifacts' directory will be created in the current working directory
    # where the model_factory.py is executed.
    unique_model_path = os.path.join("autogluon_model_artifacts", str(uuid.uuid4()))

    # The total remaining search time is 3033 seconds for ALL ten fits.
    # For 10 folds, this is roughly 300 seconds per fold.
    # Setting time_limit to 270 seconds allows for some overhead and ensures
    # all fits can complete within the shared deadline.
    time_limit_per_fold = 270 # seconds

    # Instantiate the wrapper. The actual AutoGluon predictor will be initialized
    # and fitted within the wrapper's .fit() method when called by the harness.
    return AutoGluonTabularWrapper(
        unique_model_path=unique_model_path,
        eval_metric='balanced_accuracy',
        time_limit=time_limit_per_fold
    )