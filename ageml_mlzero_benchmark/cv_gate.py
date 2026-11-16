"""Mandatory matched-CV interface for MLZero nodes; no LLM judge sets scores."""

import ast
import os
import subprocess
import time
from pathlib import Path

from . import read_json, sha256_file
from .cv_export import trusted_cv


def factory_source(code):
    tree = ast.parse(code)
    definitions = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "build_model"
    ]
    if len(definitions) != 1:
        raise ValueError(
            "Return a Python module with exactly one top-level def build_model(X_train, y_train)"
        )
    for node in tree.body:
        if isinstance(
            node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef)
        ):
            continue
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            continue
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            ast.literal_eval(node.value)
            continue
        raise ValueError(
            "Factory module top level may only contain imports, definitions, docstrings and literal constants; move data loading/fit/predict into methods"
        )
    return code


def prompt_contract(manager, prompt_name):
    if not os.environ.get("AGEML_CV_INPUT") or prompt_name not in {
        "PythonCoderPrompt",
        "BashCoderPrompt",
    }:
        return ""
    return """\n\nMANDATORY MATCHED-CV CONTRACT (overrides holdout/tutorial score examples):
Return the Python MODULE implementing def build_model(X_train, y_train), returning
one fresh unfitted fit/predict model per call. Include all wrapper classes in that
module. Module top level: imports, definitions, literal constants only. NO top-level
CSV reading, training, validation, predictions, external evaluator calls or main block.
All preprocessing/feature selection is fitted inside each returned model, on fold
training data only. Use seed/random_state 42. AutoGluon wrappers remain allowed.
All ten fits share the remaining one-hour search deadline: bound each fit suitably
and use a fresh temporary model directory per fold. Never allocate one hour PER fold.
The harness saves your generated module as output/model_factory.py and runs the
trusted ten-fold evaluator AFTER your Bash script installs dependencies into this
node's conda_env. Bash must create conda_env with numpy/pandas/scikit-learn and the
chosen model library. Do not run any full-data/holdout fit or invent results.csv.
Missing factory/dependencies or invalid CV output are execution failures to debug.
Only the harness's actual mean-fold CV metric will enter the existing MCTS reward.
The executor judge's guessed score and ordinary holdout scores are ignored.
"""


def pre_execution_error(manager):
    if not os.environ.get("AGEML_CV_INPUT"):
        return None
    try:
        factory_source(manager.current_node.python_code)
        return None
    except (ValueError, SyntaxError) as error:
        return "MATCHED_CV_REJECTED before execution: " + str(error)


def candidate_gate(manager, decision, feedback, guessed_score):
    """Before _simulate updates best-node/UCT: run CV within the shared deadline."""
    if not os.environ.get("AGEML_CV_INPUT"):
        return decision, feedback, guessed_score
    prepared = Path(os.environ["AGEML_CV_INPUT"])
    for name, key in [
        ("data.csv", "AGEML_CV_DATA_SHA256"),
        ("cv_contract.json", "AGEML_CV_CONTRACT_SHA256"),
        ("evaluate_cv.py", "AGEML_CV_EVALUATOR_SHA256"),
    ]:
        if sha256_file(prepared / name) != os.environ[key]:
            raise RuntimeError("Trusted CV input changed: " + name)
    node = manager.current_node
    output = Path(manager.get_per_iteration_output_folder(node))
    output.mkdir(parents=True, exist_ok=True)
    try:
        source = factory_source(node.python_code)
        factory = output / "model_factory.py"
        factory.write_text(source)
        for name in ("cv_evaluation.json", "cv_predictions.csv"):
            (output / name).unlink(missing_ok=True)
        python = Path(manager.get_iteration_folder(node)) / "conda_env/bin/python"
        if not python.is_file():
            raise ValueError(
                "Bash script must create this node conda_env with numpy, pandas, scikit-learn and model dependencies"
            )
        remaining = float(os.environ["AGEML_CV_DEADLINE"]) - time.time()
        if remaining <= 0:
            raise ValueError("Shared exploration deadline expired before CV")
        # Stream subprocess logs to file, avoiding unbounded buffered model output.
        log = output / "cv_execution.log"
        with log.open("w") as stream:
            result = subprocess.run(
                [
                    str(python),
                    str(prepared / "evaluate_cv.py"),
                    "--factory",
                    str(factory),
                    "--output",
                    str(output),
                ],
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=remaining,
                check=False,
                env={**os.environ, "PYTHONHASHSEED": "42"},
            )
        if result.returncode:
            tail = log.read_text(errors="replace")[-6000:]
            raise ValueError(f"Trusted CV evaluator exited {result.returncode}: {tail}")
        if time.time() > float(os.environ["AGEML_CV_DEADLINE"]):
            raise ValueError("CV completed after shared exploration deadline")
        checked = trusted_cv(output, prepared)
        value = checked["cv_score"]
        if read_json(prepared / "cv_contract.json")["task"] == "regression":
            value = -value
        return "SUCCESS", "", value
    except (OSError, ValueError, SyntaxError, subprocess.TimeoutExpired) as error:
        # Preserve artifacts/logs, give native debug mechanism concrete feedback;
        # invalid nodes never update _best_node or last_successful_node.
        return "FAILURE", "MATCHED_CV_REJECTED: " + str(error), None
