import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector

def build_model(X_train, y_train, max_iter, feature_range):
    # Select numerical features for imputation and scaling
    numerical_features = make_column_selector(dtype_include=np.number)

    # Create a pipeline for numerical features: imputation then normalization
    numerical_transformer = Pipeline(steps=[
        ('imputer', IterativeImputer(max_iter=max_iter, random_state=42)),
        ('scaler', MinMaxScaler(feature_range=tuple(feature_range)))
    ])

    # Create a ColumnTransformer to apply the numerical transformer to numerical features
    # and drop any other columns (e.g., categorical features, as no specific handling
    # for them is mentioned in the pipeline steps).
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features)
        ],
        remainder='drop' # Drop columns not specified in transformers
    )

    # Create the final pipeline: preprocessing followed by linear regression
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])

    return pipeline
