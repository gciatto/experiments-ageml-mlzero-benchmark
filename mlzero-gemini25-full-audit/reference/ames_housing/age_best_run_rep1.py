from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.compose import ColumnTransformer
import pandas as pd

def build_model(X_train, y_train, imputation_strategy, quantile_range, alpha):
    # Ensure quantile_range is a tuple for RobustScaler
    if isinstance(quantile_range, list):
        quantile_range = tuple(quantile_range)

    # Identify numerical and categorical columns
    numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns

    # Preprocessing for numerical features: imputation and scaling
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=imputation_strategy)),
        ('scaler', RobustScaler(quantile_range=quantile_range))
    ])

    # Preprocessing for categorical features: imputation and one-hot encoding
    # 'most_frequent' is a suitable strategy for categorical data imputation
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Create a preprocessor using ColumnTransformer to apply different transformations to different column types
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='passthrough' # Keep any other columns not explicitly transformed
    )

    # Create the full pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', Ridge(alpha=alpha))
    ])
    return pipeline
