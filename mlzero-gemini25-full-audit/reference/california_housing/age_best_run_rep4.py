from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Ridge
from sklearn.compose import ColumnTransformer

def build_model(X_train, y_train, imputation_strategy, quantile_range, alpha):
    numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns

    # Ensure quantile_range is a tuple as required by RobustScaler
    quantile_range_tuple = tuple(quantile_range)

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=imputation_strategy)),
        ('scaler', RobustScaler(quantile_range=quantile_range_tuple))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols)
        ],
        remainder='drop'
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', Ridge(alpha=alpha))
    ])
    return pipeline
