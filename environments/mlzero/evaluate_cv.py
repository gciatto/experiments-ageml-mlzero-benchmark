"""Trusted ten-fold candidate evaluator, executed inside MLZero's node environment.

Generated factory contract: build_model(X_train, y_train) returns a NEW unfitted
estimator exposing fit/predict. AutoGluon may be used through a compatible wrapper.
No generated code is executed by the host-side benchmark orchestration.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, mean_squared_error


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluate(factory_path, contract_path, output):
    contract_path = Path(contract_path)
    contract = json.loads(contract_path.read_text())
    data = contract_path.parent / "data.csv"
    if digest(data) != contract["source_sha256"]:
        raise ValueError("CSV checksum mismatch")
    frame = pd.read_csv(data)
    target = contract["target_column"]
    random.seed(42)
    np.random.seed(42)
    spec = importlib.util.spec_from_file_location("candidate_factory", factory_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    scores = []
    rows = []
    seeds = []
    start = time.monotonic()
    for fold, part in enumerate(contract["cv_folds"]):
        random.seed(42)
        np.random.seed(42)
        train = frame.iloc[part["train"]]
        validation = frame.iloc[part["validation"]]
        X = train.drop(columns=[target])
        y = train[target]
        model = module.build_model(X.copy(), y.copy())
        applied = {}
        if hasattr(model, "get_params") and hasattr(model, "set_params"):
            applied = {
                k: 42
                for k in model.get_params(deep=True)
                if k == "random_state" or k.endswith("__random_state")
            }
            if applied:
                model.set_params(**applied)
        model.fit(X, y)
        prediction = np.asarray(model.predict(validation.drop(columns=[target])))
        if prediction.ndim != 1 or len(prediction) != len(validation):
            raise ValueError("Invalid prediction shape")
        score = (
            float(balanced_accuracy_score(validation[target], prediction))
            if contract["task"] == "classification"
            else float(np.sqrt(mean_squared_error(validation[target], prediction)))
        )
        if not np.isfinite(score):
            raise ValueError("Non-finite fold score")
        scores.append(score)
        seeds.append(applied)
        rows.extend(
            {"row_id": row, "fold": fold, "prediction": pred}
            for row, pred in zip(part["validation"], prediction.tolist())
        )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values("row_id").to_csv(
        output / "cv_predictions.csv", index=False
    )
    record = {
        "score_role": "selection_cv_not_independent_test",
        "fold_scores": scores,
        "cv_score": float(np.mean(scores)),
        "fit_seed": 42,
        "effective_random_states": seeds,
        "source_sha256": digest(data),
        "contract_sha256": digest(contract_path),
        "factory_sha256": digest(factory_path),
        "evaluator_sha256": digest(__file__),
        "wall_seconds": time.monotonic() - start,
    }
    # Write last: a deadline-interrupted candidate is never eligible.
    temporary = output / "cv_evaluation.json.tmp"
    temporary.write_text(json.dumps(record, sort_keys=True, allow_nan=False))
    os.replace(temporary, output / "cv_evaluation.json")
    print(
        "MEAN_10_FOLD_"
        + ("BALANCED_ACCURACY" if contract["task"] == "classification" else "RMSE")
        + "="
        + str(record["cv_score"])
    )
    return record


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--factory", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument(
        "--contract",
        default=str(Path(__file__).with_name("cv_contract.json")),
        type=Path,
    )
    a = p.parse_args()
    evaluate(a.factory, a.contract, a.output)
