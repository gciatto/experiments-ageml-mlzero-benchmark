from sklearn.pipeline import Pipeline
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
import pandas as pd

def build_model(X_train, y_train, max_iter, feature_range):
    numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns

    # Ensure feature_range is a tuple for MinMaxScaler
    if isinstance(feature_range, list):
        feature_range_tuple = tuple(feature_range)
    else:
        feature_range_tuple = feature_range

    numerical_transformer = Pipeline(steps=[
        ('imputer', IterativeImputer(max_iter=max_iter, random_state=42)),
        ('scaler', MinMaxScaler(feature_range=feature_range_tuple))
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])

    return pipeline
