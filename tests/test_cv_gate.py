"""Real node-environment fold evaluation must precede native candidate selection."""

import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from sklearn.model_selection import KFold, StratifiedKFold

from ageml_mlzero_benchmark import cv_gate as gate
from ageml_mlzero_benchmark import sha256_file, write_json


@pytest.fixture
def node(tmp_path, monkeypatch):
    def create(task="classification"):
        prepared = tmp_path / "input"
        prepared.mkdir()
        output = tmp_path / "node/output"
        output.mkdir(parents=True)
        frame = pd.DataFrame(
            {
                "x": range(40),
                "y": [0] * 20 + [10] * 20 if task == "regression" else [0, 1] * 20,
            }
        )
        frame.to_csv(prepared / "data.csv", index=False)
        splitter = (
            KFold(10, shuffle=False)
            if task == "regression"
            else StratifiedKFold(10, shuffle=False)
        )
        write_json(
            prepared / "cv_contract.json",
            {
                "task": task,
                "target_column": "y",
                "source_sha256": sha256_file(prepared / "data.csv"),
                "cv_folds": [
                    {"train": a.tolist(), "validation": b.tolist()}
                    for a, b in splitter.split(frame[["x"]], frame.y)
                ],
            },
        )
        evaluator = (
            Path(__file__).resolve().parents[1] / "environments/mlzero/evaluate_cv.py"
        )
        shutil.copy2(evaluator, prepared / "evaluate_cv.py")
        python = tmp_path / "node/conda_env/bin/python"
        python.parent.mkdir(parents=True)
        python.symlink_to(sys.executable)
        code = (
            'from sklearn.dummy import DummyRegressor\ndef build_model(X_train,y_train):\n return DummyRegressor(strategy="constant",constant=0)\n'
            if task == "regression"
            else 'from sklearn.dummy import DummyClassifier\ndef build_model(X_train,y_train):\n return DummyClassifier(strategy="most_frequent")\n'
        )
        manager = SimpleNamespace(
            current_node=SimpleNamespace(python_code=code),
            get_iteration_folder=lambda n: str(tmp_path / "node"),
            get_per_iteration_output_folder=lambda n: str(output),
        )
        for key, value in {
            "AGEML_CV_INPUT": str(prepared),
            "AGEML_CV_DEADLINE": str(time.time() + 60),
            "AGEML_CV_DATA_SHA256": sha256_file(prepared / "data.csv"),
            "AGEML_CV_CONTRACT_SHA256": sha256_file(prepared / "cv_contract.json"),
            "AGEML_CV_EVALUATOR_SHA256": sha256_file(prepared / "evaluate_cv.py"),
        }.items():
            monkeypatch.setenv(key, value)
        return manager, output, prepared

    return create


@pytest.mark.parametrize(
    "task,expected", [("classification", 0.5), ("regression", -5.0)]
)
def test_actual_ten_fold_score_replaces_judge_guess(node, task, expected):
    manager, output, _ = node(task)
    assert gate.candidate_gate(
        manager, "FIX", "judge found no holdout results", 0.999
    ) == ("SUCCESS", "", expected)
    assert (output / "model_factory.py").is_file()
    assert (output / "cv_evaluation.json").is_file()
    assert len(pd.read_csv(output / "cv_predictions.csv")) == 40


def test_old_training_script_rejected_before_execution(node):
    manager, output, _ = node()
    manager.current_node.python_code = 'print("fit and claim holdout score 0.99")'
    assert "MATCHED_CV_REJECTED" in gate.pre_execution_error(manager)
    decision, feedback, score = gate.candidate_gate(manager, "SUCCESS", "", 0.99)
    assert decision == "FAILURE" and score is None and "build_model" in feedback
    assert not (output / "cv_evaluation.json").exists()


def test_missing_node_environment_is_debug_failure(node):
    manager, output, _ = node()
    (output.parent / "conda_env/bin/python").unlink()
    decision, feedback, score = gate.candidate_gate(manager, "SUCCESS", "", 0.99)
    assert decision == "FAILURE" and score is None and "conda_env" in feedback


def test_expired_deadline_cannot_admit_candidate(node, monkeypatch):
    manager, output, _ = node()
    monkeypatch.setenv("AGEML_CV_DEADLINE", str(time.time() - 1))
    decision, feedback, score = gate.candidate_gate(manager, "SUCCESS", "", 0.99)
    assert decision == "FAILURE" and score is None and "deadline" in feedback
    assert not (output / "cv_evaluation.json").exists()


def test_changed_official_inputs_abort_search(node):
    manager, _, prepared = node()
    (prepared / "cv_contract.json").write_text("{}")
    with pytest.raises(RuntimeError, match="Trusted CV input changed"):
        gate.candidate_gate(manager, "SUCCESS", "", 0.99)


@pytest.mark.parametrize(
    "code",
    [
        "from sklearn.dummy import DummyClassifier\nmodel=DummyClassifier()\ndef build_model(x,y): return model",
        'def build_model(x,y): return None\nif __name__=="__main__": print("fit")',
    ],
)
def test_import_time_training_or_calls_rejected(code):
    with pytest.raises(ValueError):
        gate.factory_source(code)


def test_prompt_override_only_for_coders(monkeypatch):
    monkeypatch.setenv("AGEML_CV_INPUT", "/input")
    assert "MANDATORY MATCHED-CV" in gate.prompt_contract(None, "PythonCoderPrompt")
    assert "conda_env" in gate.prompt_contract(None, "BashCoderPrompt")
    assert gate.prompt_contract(None, "ExecuterPrompt") == ""
    monkeypatch.delenv("AGEML_CV_INPUT")
    assert gate.prompt_contract(None, "PythonCoderPrompt") == ""
    assert gate.candidate_gate(None, "SUCCESS", "original", 0.8) == (
        "SUCCESS",
        "original",
        0.8,
    )


def test_wrapper_class_with_dataclass_annotations_is_importable(node):
    manager, _, _ = node()
    manager.current_node.python_code = """from __future__ import annotations
from dataclasses import dataclass
from sklearn.dummy import DummyClassifier
@dataclass
class Wrapper:
    estimator: object = None
    def fit(self, X, y):
        self.estimator = DummyClassifier(strategy="most_frequent").fit(X,y)
        return self
    def predict(self, X):
        return self.estimator.predict(X)
def build_model(X_train,y_train):
    return Wrapper()
"""
    assert gate.candidate_gate(manager, "SUCCESS", "", 0.999) == ("SUCCESS", "", 0.5)
