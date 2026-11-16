"""Execute the patched native selection method with real ten-fold toy fits, no LLM."""

import argparse
import ast
import json
import logging
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ageml_mlzero_benchmark import sha256_file, write_json


def native_simulate(source):
    tree = ast.parse(
        (source / "src/autogluon/assistant/managers/node_manager.py").read_text()
    )
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "NodeManager"
    )
    method = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "simulate"
    )
    if "candidate_gate" not in ast.unparse(
        method
    ) or "pre_execution_error" not in ast.unparse(method):
        raise RuntimeError("Native NodeManager lacks the mandatory CV gate patch")
    namespace = {"os": os, "logger": logging.getLogger("cv-gate-smoke")}
    exec(  # noqa: S102 -- compile the hash-verified pinned native method, not generated code.
        compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    return namespace["simulate"]


def run_check(source):
    import pandas as pd
    from sklearn.model_selection import KFold, StratifiedKFold

    simulate = native_simulate(source)
    results = {}
    for task in ("classification", "regression", "legacy-holdout", "broken-factory"):
        with tempfile.TemporaryDirectory(prefix="mlzero-cv-gate-") as tmp:
            root = Path(tmp)
            prepared = root / "input"
            prepared.mkdir()
            node_dir = root / "node"
            output = node_dir / "output"
            output.mkdir(parents=True)
            python = node_dir / "conda_env/bin/python"
            python.parent.mkdir(parents=True)
            python.symlink_to(sys.executable)
            regression = task == "regression"
            frame = pd.DataFrame(
                {
                    "x": range(40),
                    "y": [0] * 20 + [10] * 20 if regression else [0, 1] * 20,
                }
            )
            frame.to_csv(prepared / "data.csv", index=False)
            splitter = (
                KFold(10, shuffle=False)
                if regression
                else StratifiedKFold(10, shuffle=False)
            )
            write_json(
                prepared / "cv_contract.json",
                {
                    "task": "regression" if regression else "classification",
                    "target_column": "y",
                    "fit_seed": 42,
                    "source_sha256": sha256_file(prepared / "data.csv"),
                    "cv_folds": [
                        {"train": a.tolist(), "validation": b.tolist()}
                        for a, b in splitter.split(frame[["x"]], frame.y)
                    ],
                },
            )
            shutil.copy2(
                ROOT / "environments/mlzero/evaluate_cv.py", prepared / "evaluate_cv.py"
            )
            code = (
                'from sklearn.dummy import DummyRegressor\ndef build_model(X_train,y_train):\n return DummyRegressor(strategy="constant",constant=0)\n'
                if regression
                else 'from sklearn.dummy import DummyClassifier\ndef build_model(X_train,y_train):\n return DummyClassifier(strategy="most_frequent")\n'
            )
            if task == "legacy-holdout":
                code = 'raise RuntimeError("This legacy holdout script must never execute")'
            if task == "broken-factory":
                code = 'def build_model(X_train,y_train):\n raise ValueError("invalid candidate")\n'
            node = SimpleNamespace(
                python_code=code,
                bash_script="",
                stage="evolve",
                id=0,
                tool_used="machine learning",
            )
            calls = []
            manager = SimpleNamespace(
                current_node=node,
                task_description="",
                data_prompt="",
                _best_node=None,
                _best_validation_score=None,
                _worst_validation_score=None,
                last_successful_node=None,
                time_step=0,
                _all_error_analyses=[],
                get_iteration_folder=lambda n, folder=node_dir: str(folder),
                get_per_iteration_output_folder=lambda n, folder=output: str(folder),
                save_and_log_states=lambda *a, **k: None,
                error_analyzer=lambda: "debug feedback",
            )

            def execute(calls=calls, **kwargs):
                calls.append(True)
                return "SUCCESS", "", 0.999, "", "", ""

            manager.executer = execute
            env = {
                "AGEML_CV_INPUT": str(prepared),
                "AGEML_CV_DATA_SHA256": sha256_file(prepared / "data.csv"),
                "AGEML_CV_CONTRACT_SHA256": sha256_file(prepared / "cv_contract.json"),
                "AGEML_CV_EVALUATOR_SHA256": sha256_file(prepared / "evaluate_cv.py"),
                "AGEML_CV_DEADLINE": str(time.time() + 60),
                "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""),
            }
            previous = {k: os.environ.get(k) for k in env}
            os.environ.update(env)
            try:
                result = simulate(manager)
                if task in ("classification", "regression"):
                    expected = -5.0 if regression else 0.5
                    assert result == (expected, True, False), result
                    assert (
                        manager._best_node is node
                        and manager.last_successful_node is node
                    )
                    assert (
                        node.validation_score == expected
                        and (output / "cv_evaluation.json").is_file()
                    )
                else:
                    assert result == (None, False, True), result
                    assert (
                        manager._best_node is None
                        and manager.last_successful_node is None
                    )
                    if task == "legacy-holdout":
                        assert not calls
                    assert "MATCHED_CV_REJECTED" in node.error_message
                results[task] = {
                    "score": result[0],
                    "selected": manager._best_node is not None,
                }
            finally:
                for key, value in previous.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps({"cv_gate_smoke": "passed", "cases": run_check(args.source)}))
