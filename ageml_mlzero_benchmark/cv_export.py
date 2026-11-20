"""Frozen-fold CV verification; float64 scores with bounded float32 compatibility.

Older evaluators score native float32 predictions before CSV serialization.
Verification recomputes in float64. Regression-only tolerance accommodates that
rounding, not wrong folds, missing predictions, altered sources or score changes.
"""
import math
import numpy as np
import pandas as pd
from . import read_json, sha256_file
from .evaluation import score

REVISION = 'rmse-float32-compat-v1'
REGRESSION_RTOL = 1e-6  # About eight float32 epsilons; observed errors < 5e-8.
REGRESSION_ATOL = 1e-8


def trusted_cv(output, prepared):
    contract = read_json(prepared / 'cv_contract.json')
    ev = read_json(output / 'cv_evaluation.json')
    if (
        ev.get('contract_sha256') != sha256_file(prepared / 'cv_contract.json')
        or ev.get('source_sha256') != sha256_file(prepared / 'data.csv')
        or ev.get('factory_sha256') != sha256_file(output / 'model_factory.py')
        or ev.get('evaluator_sha256') != sha256_file(prepared / 'evaluate_cv.py')
        or ev.get('fit_seed') != 42
    ):
        raise ValueError('CV evaluation provenance mismatch')
    pred = pd.read_csv(output / 'cv_predictions.csv', dtype={'prediction': str}, keep_default_na=False)
    frame = pd.read_csv(prepared / 'data.csv')
    target = contract['target_column']
    folds = contract['cv_folds']
    if len(folds) != 10:
        raise ValueError('Expected exactly ten prescribed folds')
    if (
        list(pred.columns) != ['row_id', 'fold', 'prediction']
        or pred.row_id.duplicated().any()
        or set(pred.row_id) != set(range(len(frame)))
    ):
        raise ValueError('Invalid OOF prediction identities')
    if not set(pred.fold) <= set(range(10)):
        raise ValueError('Unexpected fold IDs')
    scores = []
    classification = contract['task'] == 'classification'
    for fold, part in enumerate(folds):
        observed = pred.loc[pred.fold.eq(fold)].set_index('row_id')
        if set(observed.index) != set(part['validation']):
            raise ValueError('Wrong fold assignment')
        y = frame.iloc[part['validation']][target]
        p = observed.loc[part['validation'], 'prediction']
        if classification and not set(p) <= set(frame[target].astype(str)):
            raise ValueError('Unknown predicted labels')
        scores.append(score('balanced_accuracy' if classification else 'rmse', y, p))
    mean = float(np.mean(scores))
    declared = np.asarray(ev.get('fold_scores', []), dtype=np.float64)
    declared_mean = float(ev.get('cv_score', float('nan')))
    rtol, atol = (1e-10, 1e-10) if classification else (REGRESSION_RTOL, REGRESSION_ATOL)
    if (
        not np.isfinite(mean)
        or declared.shape != (10,)
        or not np.isfinite(declared).all()
        or not math.isfinite(declared_mean)
        or not np.allclose(scores, declared, rtol=rtol, atol=atol)
        or not math.isclose(mean, declared_mean, rel_tol=rtol, abs_tol=atol)
        # A reported mean must also agree with its own declared fold scores.
        or not math.isclose(float(np.mean(declared)), declared_mean, rel_tol=1e-10, abs_tol=1e-10)
    ):
        raise ValueError('Declared scores disagree with recomputed folds')
    return {'cv_score': mean, 'cv_fold_scores': scores,
            'score_role': 'selection_cv_not_independent_test', 'test_score': None}
