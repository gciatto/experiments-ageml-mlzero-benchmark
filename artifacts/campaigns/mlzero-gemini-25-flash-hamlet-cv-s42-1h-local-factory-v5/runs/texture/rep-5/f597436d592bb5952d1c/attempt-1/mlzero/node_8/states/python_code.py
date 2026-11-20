import pandas as pd
import uuid
import os
import logging
from autogluon.multimodal import MultiModalPredictor

# Configure logging to suppress excessive AutoGluon output if desired.
# For this benchmark, we set it to WARNING to reduce verbosity.
logging.basicConfig(level=logging.WARNING)

# Module-level constants
RANDOM_STATE = 42
# Base directory for AutoGluon models. Each fit will create a unique subdirectory here.
# Using /tmp is generally safe for temporary files in a containerized environment.
AUTOGLUON_MODEL_BASE_DIR = "/tmp/autogluon_multimodal_models"

class AutoGluonMLZeroModel:
    """
    A wrapper class for AutoGluon MultiModalPredictor to conform to the MLZero benchmark API.
    This class handles the fitting and prediction logic for a single fold.
    """
    def __init__(self, label_column: str, random_state: int = RANDOM_STATE):
        """
        Initializes the model wrapper.

        Args:
            label_column: The name of the target column.
            random_state: Seed for reproducibility of stochastic elements.
        """
        self.label_column = label_column
        self.random_state = random_state
        self.predictor = None  # The MultiModalPredictor instance, initialized in fit()
        # Ensure the base directory exists for model artifacts.
        # MultiModalPredictor will create its specific model_path within this base.
        os.makedirs(AUTOGLUON_MODEL_BASE_DIR, exist_ok=True)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fits the AutoGluon MultiModalPredictor model using the provided training data.

        Args:
            X: Training features (pandas DataFrame).
            y: Training target (pandas Series).
        """
        # Combine X and y into a single DataFrame as required by MultiModalPredictor
        train_data = X.copy()
        train_data[self.label_column] = y

        # Create a unique path for this fold's AutoGluon model artifacts.
        # This ensures that each fold's model is trained and stored independently,
        # preventing data leakage and ensuring fresh preprocessing.
        model_path = os.path.join(AUTOGLUON_MODEL_BASE_DIR, f"automm_model_{uuid.uuid4().hex}")

        # Initialize MultiModalPredictor
        # - label: Specifies the target column ('Class' for this task).
        # - eval_metric: Set to 'balanced_accuracy' as per task requirements.
        # - path: Directs AutoGluon to save artifacts in a unique, temporary directory.
        # - presets: "medium_quality" is chosen to balance performance and training speed,
        #            crucial for meeting the shared deadline across 10 folds.
        self.predictor = MultiModalPredictor(
            label=self.label_column,
            eval_metric='balanced_accuracy',
            path=model_path,
            presets="medium_quality"
        )

        # Fit the predictor.
        # The time_limit is critical to ensure that each fold's training completes
        # within a reasonable timeframe, allowing all ten fits to finish within
        # the total shared deadline.
        # With a total remaining time of 867 seconds for 10 folds,
        # setting time_limit to 60 seconds per fold (10 folds * 60s = 600s)
        # leaves 267 seconds for overhead (data loading, model initialization, prediction, evaluation),
        # which should be sufficient.
        self.predictor.fit(
            train_data=train_data,
            time_limit=60,  # seconds per fold
            seed=self.random_state # Pass seed here for MultiModalPredictor's stochastic processes
        )

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Makes predictions using the fitted AutoGluon MultiModalPredictor.

        Args:
            X: Features to predict on (pandas DataFrame).

        Returns:
            Predicted class labels (pandas Series).
        """
        if self.predictor is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        
        # AutoGluon's predict method for classification returns a Series of predicted labels.
        predictions = self.predictor.predict(X)
        return predictions

def build_model(X_train: pd.DataFrame, y_train: pd.Series) -> AutoGluonMLZeroModel:
    """
    Factory function to build a fresh, unfitted AutoGluon MultiModalPredictor wrapper.

    This function is called once per candidate evaluation. It should return an
    unfitted model instance that can then be trained and used for prediction.

    Args:
        X_train: Training features (used for schema inference, not fitting).
        y_train: Training target (used for label column inference, not fitting).

    Returns:
        An unfitted instance of AutoGluonMLZeroModel, ready for fitting.
    """
    # Infer the label column name from the y_train Series.
    # The problem statement specifies 'Class' as the target column.
    label_column = y_train.name
    if label_column is None:
        # Fallback if y_train has no name, though it's expected to be 'Class' per contract.
        logging.warning("y_train Series has no name. Assuming 'Class' as the label column based on task description.")
        label_column = 'Class'

    # Return a new, unfitted model wrapper.
    return AutoGluonMLZeroModel(label_column=label_column, random_state=RANDOM_STATE)