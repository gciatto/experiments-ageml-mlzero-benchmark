from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

def build_model(X_train, y_train, quantile_range, alpha):
    # Identify numerical and categorical features
    numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns

    # Create preprocessing steps for numerical features
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler(quantile_range=tuple(quantile_range)))
    ])

    # Create preprocessing steps for categorical features
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Create a column transformer to apply different transformations to different column types
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough' # Keep any other columns not explicitly transformed
    )

    # Create the full pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('ridge', Ridge(alpha=alpha))
    ])
    return pipeline
