import lightgbm as lgb
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
import numpy as np
import logging
import os

# Module-level initialization
# Set up basic logging to stdout/stderr, which the harness can capture.
# The logging level is INFO, so it will show progress messages during execution.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Seed for reproducibility across runs and within LightGBM.
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED) # For any numpy operations that might be random

class MadelonClassifierWrapper:
    """
    A wrapper class for a LightGBM classifier with StandardScaler preprocessing.
    It handles potential non-0/1 target labels by using a LabelEncoder to ensure
    compatibility with LightGBM's binary objective, and then inverse transforms
    predictions to the original label format.
    """
    def __init__(self, random_state=RANDOM_SEED):
        self.random_state = random_state
        self.label_encoder = None # Initialized in fit if needed

        # Define the machine learning pipeline.
        # 1. StandardScaler: Scales numerical features to zero mean and unit variance.
        #    This can help some models, though tree-based models like LightGBM are
        #    less sensitive to feature scaling. It's included for robustness.
        # 2. LGBMClassifier: A gradient boosting model, chosen for its speed and
        #    strong performance on tabular data, especially with high dimensionality.
        # Parameters for LGBMClassifier are chosen to balance performance and speed,
        # which is crucial for the tight deadline across 10 cross-validation folds.
        # 'binary' objective is suitable for the Madelon dataset's binary classification task.
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('lgbm', lgb.LGBMClassifier(objective='binary',
                                        random_state=self.random_state,
                                        n_estimators=200,      # Number of boosting rounds
                                        learning_rate=0.05,    # Step size shrinkage
                                        num_leaves=20,         # Max number of leaves in one tree
                                        max_depth=5,           # Max tree depth
                                        n_jobs=-1,             # Use all available CPU cores
                                        verbose=-1             # Suppress verbose output during training
                                       ))
        ])
        logger.info("MadelonClassifierWrapper initialized with LightGBM pipeline.")

    def fit(self, X, y):
        """
        Fits the preprocessing pipeline and the LightGBM model.
        Applies LabelEncoder if target labels are not 0/1, as LightGBM's binary
        objective expects target labels to be 0 and 1.
        """
        logger.info(f"Starting model fit for X shape: {X.shape}, y shape: {y.shape}")

        # Check and encode target labels if they are not already 0/1.
        unique_y = np.unique(y)
        # If labels are not exactly [0, 1] or a single class (e.g., [0] or [1]),
        # then apply LabelEncoder. This handles cases like [1, 2] or [-1, 1].
        if not (np.array_equal(unique_y, [0, 1]) or len(unique_y) == 1):
            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(y)
            logger.info(f"Target labels encoded from {unique_y} to {np.unique(y_encoded)}")
        else:
            # If labels are already 0/1 or only one class, no encoding needed.
            y_encoded = y
            logger.info(f"Target labels are already 0/1 or single class. No LabelEncoder applied. Unique values: {unique_y}")

        # Fit the pipeline (StandardScaler and LGBMClassifier) on the training data.
        self.pipeline.fit(X, y_encoded)
        logger.info("Model fitting complete.")
        return self

    def predict(self, X):
        """
        Makes predictions using the fitted model.
        If a LabelEncoder was used during fitting, it inverse transforms the
        predictions to return them in the original label format.
        """
        logger.info(f"Starting prediction for X shape: {X.shape}")
        predictions_encoded = self.pipeline.predict(X)

        # If a LabelEncoder was used, inverse transform the predictions
        # to return them in the original label format (e.g., 2.0 instead of 1).
        if self.label_encoder:
            predictions = self.label_encoder.inverse_transform(predictions_encoded)
            logger.info(f"Predictions decoded from {np.unique(predictions_encoded)} to {np.unique(predictions)}")
        else:
            predictions = predictions_encoded
            logger.info(f"Predictions are {np.unique(predictions)}. No LabelEncoder inverse transform applied.")
        return predictions

def build_model(X_train, y_train):
    """
    Factory function required by the harness.
    Returns a new, unfitted instance of the MadelonClassifierWrapper.
    X_train and y_train are provided for context (e.g., for logging shapes)
    but are NOT used for fitting or any model-specific initialization here.
    The actual fitting occurs within the .fit() method of the returned object.
    """
    logger.info(f"Factory function 'build_model' called. X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    return MadelonClassifierWrapper(random_state=RANDOM_SEED)