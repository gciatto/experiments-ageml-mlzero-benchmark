from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import PowerTransformer
from sklearn.svm import SVR

def build_model(X_train, y_train, max_iter, method, kernel, C, epsilon):
    pipeline = Pipeline([
        ('imputation', IterativeImputer(max_iter=max_iter, initial_strategy='mean', random_state=42)),
        ('normalization', PowerTransformer(method=method)),
        ('regression', SVR(kernel=kernel, C=C, epsilon=epsilon))
    ])
    return pipeline
