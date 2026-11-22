import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold
from ageml_mlzero_benchmark import sha256_file

ROOT = Path(__file__).parents[2]
spec = importlib.util.spec_from_file_location(
    "ageml_mlzero_benchmark.cv_export_repair", ROOT / "scripts/cv_export_corrected.py"
)
assert spec is not None and spec.loader is not None
cv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cv)


class NumericTests(unittest.TestCase):
    def fixture(self, root, classification=False, dtype=np.float32):
        prepared, output = root / "input", root / "output"
        prepared.mkdir()
        output.mkdir()
        rng = np.random.default_rng(42)
        y = (
            np.array([0, 1] * 100)
            if classification
            else rng.integers(50000, 500000, 200)
        )
        pred = (
            y.copy()
            if classification
            else (y + rng.normal(0, 25000, 200)).astype(dtype)
        )
        data = pd.DataFrame({"x": range(200), "y": y})
        data.to_csv(prepared / "data.csv", index=False)
        splitter = StratifiedKFold(10) if classification else KFold(10)
        folds = [
            {"train": a.tolist(), "validation": b.tolist()}
            for a, b in splitter.split(data, y)
        ]
        contract = {
            "task": "classification" if classification else "regression",
            "target_column": "y",
            "source_sha256": sha256_file(prepared / "data.csv"),
            "cv_folds": folds,
        }
        (prepared / "cv_contract.json").write_text(json.dumps(contract))
        (prepared / "evaluate_cv.py").write_text("frozen original evaluator")
        (output / "model_factory.py").write_text("original model source")
        rows = []
        scores = []
        for fold, part in enumerate(folds):
            ids = part["validation"]
            if classification:
                value = float(balanced_accuracy_score(y[ids], pred[ids]))
            elif dtype == np.float32:
                # Reproduce native float32 arithmetic independently of the host
                # sklearn version (node environments can use a different one).
                errors = y[ids].astype(np.float32) - pred[ids]
                value = math.sqrt(float(np.mean(errors**2, dtype=np.float32)))
            else:
                value = float(np.sqrt(mean_squared_error(y[ids], pred[ids])))
            scores.append(value)
            rows.extend(
                {"row_id": i, "fold": fold, "prediction": p}
                for i, p in zip(ids, pred[ids].tolist())
            )
        pd.DataFrame(rows).sort_values("row_id").to_csv(
            output / "cv_predictions.csv", index=False
        )
        ev = {
            "fit_seed": 42,
            "fold_scores": scores,
            "cv_score": float(np.mean(scores)),
            "contract_sha256": sha256_file(prepared / "cv_contract.json"),
            "source_sha256": sha256_file(prepared / "data.csv"),
            "evaluator_sha256": sha256_file(prepared / "evaluate_cv.py"),
            "factory_sha256": sha256_file(output / "model_factory.py"),
        }
        self.save(output, ev)
        return prepared, output, ev

    def save(self, output, ev):
        (output / "cv_evaluation.json").write_text(json.dumps(ev))

    def test_real_float32_export_accepted_and_float64_score_returned(self):
        with tempfile.TemporaryDirectory() as d:
            p, o, e = self.fixture(Path(d))
            result = cv.trusted_cv(o, p)
            self.assertTrue(math.isfinite(result["cv_score"]))
            self.assertFalse(
                math.isclose(
                    e["cv_score"], result["cv_score"], rel_tol=1e-10, abs_tol=1e-10
                )
            )
            self.assertTrue(
                math.isclose(e["cv_score"], result["cv_score"], rel_tol=1e-6)
            )
            self.assertIsNone(result["test_score"])

    def test_float64_and_classification_exports_accepted(self):
        for classification in (False, True):
            with tempfile.TemporaryDirectory() as d:
                p, o, e = self.fixture(
                    Path(d), classification=classification, dtype=np.float64
                )
                self.assertAlmostEqual(
                    cv.trusted_cv(o, p)["cv_score"], e["cv_score"], places=9
                )

    def test_genuine_score_errors_rejected(self):
        for offset in (0.1, 100.0):
            with tempfile.TemporaryDirectory() as d:
                p, o, e = self.fixture(Path(d))
                e["fold_scores"] = [v + offset for v in e["fold_scores"]]
                e["cv_score"] = float(np.mean(e["fold_scores"]))
                self.save(o, e)
                with self.assertRaisesRegex(ValueError, "disagree"):
                    cv.trusted_cv(o, p)

    def test_classification_does_not_inherit_regression_tolerance(self):
        with tempfile.TemporaryDirectory() as d:
            p, o, e = self.fixture(Path(d), classification=True)
            e["fold_scores"] = [v - 1e-8 for v in e["fold_scores"]]
            e["cv_score"] = float(np.mean(e["fold_scores"]))
            self.save(o, e)
            with self.assertRaisesRegex(ValueError, "disagree"):
                cv.trusted_cv(o, p)

    def test_altered_factory_evaluator_and_data_rejected(self):
        for which in ("factory", "evaluator", "data"):
            with tempfile.TemporaryDirectory() as d:
                p, o, e = self.fixture(Path(d))
                path = {
                    "factory": o / "model_factory.py",
                    "evaluator": p / "evaluate_cv.py",
                    "data": p / "data.csv",
                }[which]
                path.write_text(path.read_text() + "\nchanged")
                with self.assertRaisesRegex(ValueError, "provenance"):
                    cv.trusted_cv(o, p)

    def test_wrong_missing_or_duplicate_rows_rejected(self):
        for mode in ("fold", "missing", "duplicate"):
            with tempfile.TemporaryDirectory() as d:
                p, o, e = self.fixture(Path(d))
                f = pd.read_csv(o / "cv_predictions.csv")
                if mode == "fold":
                    f.loc[0, "fold"] = 9
                elif mode == "missing":
                    f = f.iloc[1:]
                else:
                    f.loc[0, "row_id"] = 1
                f.to_csv(o / "cv_predictions.csv", index=False)
                with self.assertRaises(ValueError):
                    cv.trusted_cv(o, p)

    def test_nonfinite_or_partial_fold_exports_rejected(self):
        for mode in ("nan", "inf", "partial"):
            with tempfile.TemporaryDirectory() as d:
                p, o, e = self.fixture(Path(d))
                if mode == "partial":
                    e["fold_scores"] = e["fold_scores"][:-1]
                else:
                    e["fold_scores"][0] = float(mode)
                self.save(o, e)
                with self.assertRaises(ValueError):
                    cv.trusted_cv(o, p)

    def test_fabricated_mean_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p, o, e = self.fixture(Path(d))
            e["cv_score"] += 10
            self.save(o, e)
            with self.assertRaises(ValueError):
                cv.trusted_cv(o, p)
