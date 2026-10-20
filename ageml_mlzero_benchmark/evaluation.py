"""Trusted evaluator: prediction checks and independent metrics.

Ground-truth labels never leave this process. Predictions are normalized to
``row_id,prediction`` before scoring.
"""

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, mean_squared_error

from . import OBJECTIVES


def normalize_predictions(
    native_csv: Path, row_ids: list[int], target: str | None = None
) -> pd.DataFrame:
    """Map a native prediction file (one row per input row, same order) to ``row_id,prediction``.

    The prediction column is ``prediction``, else the target name, else the only column.
    Raises ``ValueError`` when the file cannot be aligned.
    """
    df = pd.read_csv(native_csv, dtype=str, keep_default_na=False)
    if "prediction" in df.columns:
        col = "prediction"
    elif target is not None and target in df.columns:
        col = target
    elif df.shape[1] == 1:
        col = df.columns[0]
    else:
        raise ValueError(
            f"cannot identify prediction column among {list(df.columns)[:10]}"
        )
    if len(df) != len(row_ids):
        raise ValueError(
            f"prediction file has {len(df)} rows, input had {len(row_ids)}"
        )
    return pd.DataFrame({"row_id": row_ids, "prediction": df[col].to_numpy()})


def check_predictions(
    pred: pd.DataFrame, truth: pd.DataFrame, task: str, classes: list[str] | None
) -> list[str]:
    """Return a list of contract violations (empty = valid).

    ``pred`` and ``truth`` both have a ``row_id`` column; ``truth`` has ``target``.
    """
    problems = []
    if list(pred.columns) != ["row_id", "prediction"]:
        problems.append(f"schema must be row_id,prediction, got {list(pred.columns)}")
        return problems
    ids = pred["row_id"]
    if ids.duplicated().any():
        problems.append(f"{int(ids.duplicated().sum())} duplicated row_id")
    expected, got = set(truth["row_id"]), set(ids)
    if expected - got:
        problems.append(f"{len(expected - got)} missing row_id")
    if got - expected:
        problems.append(f"{len(got - expected)} unexpected row_id")
    values = pred["prediction"].astype(str)
    if task == "classification":
        assert classes is not None
        bad = ~values.isin(classes)
        if bad.any():
            problems.append(
                f"{int(bad.sum())} predictions outside class set, e.g. {values[bad].iloc[0]!r}"
            )
    else:
        nums = pd.to_numeric(values, errors="coerce")
        if (~np.isfinite(nums.to_numpy(dtype=float))).any():
            problems.append("non-finite or non-numeric regression predictions")
    return problems


def score(objective: str, y_true: Any, y_pred: Any) -> float:
    if objective == "balanced_accuracy":
        return float(
            balanced_accuracy_score(
                np.asarray(y_true, dtype=str), np.asarray(y_pred, dtype=str)
            )
        )
    if objective == "rmse":
        return math.sqrt(
            mean_squared_error(
                np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
            )
        )
    raise ValueError(f"unknown objective {objective!r}")


def evaluate(
    pred: pd.DataFrame,
    truth: pd.DataFrame,
    task: str,
    objective: str,
    classes: list[str] | None,
) -> dict[str, Any]:
    """Validate then score. Invalid predictions are never scored."""
    problems = check_predictions(pred, truth, task, classes)
    if problems:
        return {"valid": False, "problems": problems, "score": None}
    merged = truth.merge(pred, on="row_id", validate="one_to_one")
    return {
        "valid": True,
        "problems": [],
        "score": score(objective, merged["target"], merged["prediction"]),
        "maximize": OBJECTIVES[objective],
    }


def better(objective: str, a: float, b: float) -> bool:
    return a > b if OBJECTIVES[objective] else a < b
