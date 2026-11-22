from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor
from sklearn.compose import ColumnTransformer

def build_model(X_train, y_train, imputation_strategy, quantile_range, n_components, n_neighbors, weights):
    # Identify numerical and categorical features
    numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns

    # Create preprocessing steps for numerical features
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=imputation_strategy)),
        ('scaler', RobustScaler(quantile_range=tuple(quantile_range)))
    ])

    # Create preprocessing steps for categorical features
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine numerical and categorical transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Create the full pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('pca', PCA(n_components=n_components, random_state=42)),
        ('regressor', KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights))
    ])
    return pipeline
