import pandas as pd
import numpy as np
import os
import uuid
from autogluon.tabular import TabularPredictor
import logging

# Module-level initialization:
# Create a base directory for AutoGluon artifacts.
# The harness will write model_factory.py in the node output folder,
# so a relative path here will create a subfolder within that output folder.
AUTOGLUON_ARTIFACTS_BASE_DIR = "autogluon_artifacts_for_folds"
os.makedirs(AUTOGLUON_ARTIFACTS_BASE_DIR, exist_ok=True)

class AutoGluonRegressionModelWrapper:
    """
    A wrapper class for AutoGluon's TabularPredictor to conform to the
    fit/predict interface required by the factory contract.
    """
    def __init__(self, artifacts_base_dir: str):
        """
        Initializes the wrapper. The AutoGluon predictor itself is not
        instantiated or fitted until the .fit() method is called.
        
        Args:
            artifacts_base_dir: The base directory where AutoGluon will store
                                its models and temporary files. A unique
                                sub-directory will be created for each fit.
        """
        self.predictor = None
        self.artifacts_base_dir = artifacts_base_dir
        self.model_path = None # To store the specific path for this predictor instance

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fits the AutoGluon TabularPredictor model.
        All preprocessing and model training happens here, using only the provided X and y.
        
        Args:
            X: Training features as a Pandas DataFrame.
            y: Training target as a Pandas Series.
        """
        # AutoGluon expects features and target combined into a single DataFrame.
        # The target column name is 'class' as per the problem description.
        train_data = X.copy()
        train_data['class'] = y

        # Create a unique path for this fold's AutoGluon artifacts.
        # This ensures fresh artifact directories per fold, preventing conflicts
        # and allowing parallel or sequential runs without interference.
        unique_id = str(uuid.uuid4())
        self.model_path = os.path.join(self.artifacts_base_dir, f"autogluon_model_{unique_id}")
        os.makedirs(self.model_path, exist_ok=True)

        # Initialize TabularPredictor for regression.
        # The 'path' argument directs AutoGluon to store its artifacts in the
        # unique directory created for this specific model instance.
        # 'label' specifies the name of the target column.
        self.predictor = TabularPredictor(
            label='class',
            problem_type='regression',
            path=self.model_path
        )

        # Fit the predictor.
        # The time_limit is set to ensure all ten fits finish within the shared deadline.
        # Given a total remaining time of ~1999 seconds for 10 folds,
        # 190 seconds per fold (1900 seconds total) allows for some overhead.
        self.predictor.fit(
            train_data=train_data,
            time_limit=190, # seconds per fit
            presets='medium_quality' # A valid preset for AutoGluon
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Makes predictions using the fitted AutoGluon TabularPredictor.
        
        Args:
            X: Features for which to make predictions, as a Pandas DataFrame.
            
        Returns:
            A 1D NumPy array of predicted numerical values (floats).
        """
        if self.predictor is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit(X, y) first.")
        
        # AutoGluon's predict method expects a DataFrame.
        predictions = self.predictor.predict(X)
        
        # Convert predictions to a 1D NumPy array of floats, as required.
        return predictions.values.astype(float)

def build_model(X_train: pd.DataFrame, y_train: pd.Series) -> AutoGluonRegressionModelWrapper:
    """
    Factory function to create a new, unfitted AutoGluon regression model wrapper.
    
    This function is called by the harness for each cross-validation fold.
    It must return a fresh, unfitted model object.
    
    Args:
        X_train: Training features (provided for context, but not used for
                 fitting within this function, as the model must be unfitted).
        y_train: Training target (provided for context, but not used for
                 fitting within this function).
                 
    Returns:
        An unfitted instance of AutoGluonRegressionModelWrapper, which exposes
        .fit(X, y) and .predict(X) methods.
    """
    # Return a new instance of our wrapper. This instance is initially unfitted.
    # The actual fitting will occur when the harness calls its .fit() method.
    return AutoGluonRegressionModelWrapper(artifacts_base_dir=AUTOGLUON_ARTIFACTS_BASE_DIR)