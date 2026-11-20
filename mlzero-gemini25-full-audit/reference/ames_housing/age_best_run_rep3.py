import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Ridge
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer

def build_model(X_train, y_train, quantile_range, alpha):
    numerical_features_selector = make_column_selector(dtype_include=np.number)

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('robust_scaler', RobustScaler(quantile_range=tuple(quantile_range)))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features_selector)
        ],
        remainder='drop'
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('ridge', Ridge(alpha=alpha))
    ])
    return pipeline
