import pandas as pd
import uuid
import os
import warnings
from autogluon.multimodal import MultiModalPredictor

# Suppress warnings from AutoGluon or other libraries if they are too verbose
warnings.filterwarnings('ignore')

class AutoGluonMMWrapper:
    """
    A wrapper class for AutoGluon MultiModalPredictor to conform to the
    fit/predict interface required by the evaluation harness.
    """
    def __init__(self, base_output_dir="."):
        """
        Initializes the wrapper.

        Args:
            base_output_dir (str): The base directory where AutoGluon will save its artifacts.
                                   A unique subdirectory will be created within this for each fit.
        """
        self.predictor = None
        self.label_column = 'SalePrice'
        self.problem_type = 'regression'
        self.seed = 42
        # The total shared deadline is 1610 seconds for 10 folds.
        # Setting a time limit of 120 seconds per fold (10 * 120 = 1200 seconds)
        # for the MultiModalPredictor.fit() call. This leaves approximately 410 seconds
        # for overhead (Python startup, data loading, wrapper instantiation, etc.)
        # across all 10 folds, ensuring all fits finish within the deadline.
        self.time_limit_per_fold = 120
        self.base_output_dir = base_output_dir
        self.model_path = None # To store the path of the fitted model for potential loading/prediction

    def fit(self, X, y):
        """
        Fits the AutoGluon MultiModalPredictor model.

        All preprocessing and model fitting are encapsulated here,
        using only the provided training data (X, y).

        Args:
            X (pd.DataFrame): Training features.
            y (pd.Series): Training target.
        """
        # Combine X and y into a single DataFrame as required by MultiModalPredictor.
        # Ensure the 'Id' column is dropped as it should not be used as a feature.
        X_processed = X.copy()
        if 'Id' in X_processed.columns:
            X_processed = X_processed.drop(columns=['Id'])
        
        combined_df = pd.concat([X_processed, y.rename(self.label_column)], axis=1)

        # Create a unique path for this fold's model artifacts.
        # This ensures "fresh artifact directories per fold" as required.
        unique_id = uuid.uuid4().hex
        self.model_path = os.path.join(self.base_output_dir, f"automm_model_{unique_id}")
        
        # Ensure the directory exists for AutoGluon to save its files.
        os.makedirs(self.model_path, exist_ok=True)

        # Instantiate MultiModalPredictor.
        # For purely tabular data, MultiModalPredictor automatically leverages its
        # tabular backend, which includes various models like LightGBM, CatBoost, etc.
        self.predictor = MultiModalPredictor(
            label=self.label_column,
            problem_type=self.problem_type,
            path=self.model_path,
        )

        # Fit the predictor.
        # The 'seed' argument is used for reproducibility, as per the task requirements.
        # The 'time_limit' ensures each fold's training respects the overall deadline.
        self.predictor.fit(
            train_data=combined_df,
            time_limit=self.time_limit_per_fold,
            seed=self.seed,
            # Presets can be specified here, e.g., presets='best_quality'.
            # The problem statement implies presets are already configured, so default is fine.
        )

    def predict(self, X):
        """
        Makes predictions using the fitted AutoGluon MultiModalPredictor model.

        Args:
            X (pd.DataFrame): Features to make predictions on.
            
        Returns:
            pd.Series: Predicted target values.
        """
        if self.predictor is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit(X, y) first.")
        
        # Ensure the 'Id' column is dropped before making predictions.
        X_processed = X.copy()
        if 'Id' in X_processed.columns:
            X_processed = X_processed.drop(columns=['Id'])

        return self.predictor.predict(X_processed)

def build_model(X_train, y_train):
    """
    Factory function to build a new, unfitted model wrapper.
    This function is called once per cross-validation fold.

    Args:
        X_train (pd.DataFrame): Training features for the current fold.
                                (Used for schema inference by the harness, not for fitting here).
        y_train (pd.Series): Training target for the current fold.
                             (Used for schema inference by the harness, not for fitting here).
        
    Returns:
        AutoGluonMMWrapper: An unfitted model wrapper instance with `fit` and `predict` methods.
    """
    # The `base_output_dir="."` ensures that AutoGluon creates its artifact
    # directories within the current working directory, which is the node output
    # folder where `model_factory.py` is saved by the harness.
    return AutoGluonMMWrapper(base_output_dir=".")