"""Common CV export verifier, used before MCTS scoring and after selection."""

import math

import numpy as np
import pandas as pd

from . import read_json, sha256_file
from .evaluation import score


def trusted_cv(output, prepared):
    contract = read_json(prepared / "cv_contract.json")
    ev = read_json(output / "cv_evaluation.json")
    if (
        ev.get("contract_sha256") != sha256_file(prepared / "cv_contract.json")
        or ev.get("source_sha256") != sha256_file(prepared / "data.csv")
        or ev.get("factory_sha256") != sha256_file(output / "model_factory.py")
        or ev.get("evaluator_sha256") != sha256_file(prepared / "evaluate_cv.py")
        or ev.get("fit_seed") != 42
    ):
        raise ValueError("CV evaluation provenance mismatch")
    pred = pd.read_csv(
        output / "cv_predictions.csv", dtype={"prediction": str}, keep_default_na=False
    )
    frame = pd.read_csv(prepared / "data.csv")
    target = contract["target_column"]
    if (
        list(pred.columns) != ["row_id", "fold", "prediction"]
        or pred.row_id.duplicated().any()
        or set(pred.row_id) != set(range(len(frame)))
    ):
        raise ValueError("Invalid OOF prediction identities")
    if not set(pred.fold) <= set(range(10)):
        raise ValueError("Unexpected fold IDs")
    scores = []
    for fold, part in enumerate(contract["cv_folds"]):
        observed = pred.loc[pred.fold.eq(fold)].set_index("row_id")
        if set(observed.index) != set(part["validation"]):
            raise ValueError("Wrong fold assignment")
        y = frame.iloc[part["validation"]][target]
        p = observed.loc[part["validation"], "prediction"]
        if contract["task"] == "classification" and not set(p) <= set(
            frame[target].astype(str)
        ):
            raise ValueError("Unknown predicted labels")
        scores.append(
            score(
                "balanced_accuracy" if contract["task"] == "classification" else "rmse",
                y,
                p,
            )
        )
    mean = float(np.mean(scores))
    if (
        not np.isfinite(mean)
        or len(ev.get("fold_scores", [])) != 10
        or not np.allclose(scores, ev["fold_scores"], rtol=1e-10, atol=1e-10)
        or not math.isclose(mean, ev["cv_score"], rel_tol=1e-10, abs_tol=1e-10)
    ):
        raise ValueError("Declared scores disagree with recomputed folds")
    return {
        "cv_score": mean,
        "cv_fold_scores": scores,
        "score_role": "selection_cv_not_independent_test",
        "test_score": None,
    }
