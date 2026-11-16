"""Mandatory matched-CV interface for MLZero nodes; no LLM judge sets scores."""

import ast
import os
import subprocess
import time
from pathlib import Path

from . import read_json, sha256_bytes, sha256_file
from .cv_export import trusted_cv


def _static_text_value(node, bindings):
    if isinstance(node, ast.Name):
        if node.id not in bindings:
            raise ValueError("Unresolved static factory variable: " + node.id)
        return bindings[node.id]
    if isinstance(node, ast.JoinedStr):
        parts = []
        for part in node.values:
            if isinstance(part, ast.Constant):
                parts.append(part.value)
            elif isinstance(part, ast.FormattedValue):
                value = _static_text_value(part.value, bindings)
                if part.conversion in (115, 114, 97):
                    value = {115: str, 114: repr, 97: ascii}[part.conversion](value)
                spec = (
                    _static_text_value(part.format_spec, bindings)
                    if part.format_spec
                    else ""
                )
                parts.append(format(value, spec))
            else:
                raise TypeError("Unsupported factory string expression")
        return "".join(parts)
    if isinstance(node, ast.Call) and not node.keywords:
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "strip"
            and not node.args
        ):
            value = _static_text_value(node.func.value, bindings)
            if isinstance(value, str):
                return value.strip()
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "textwrap"
            and node.func.attr == "dedent"
            and len(node.args) == 1
        ):
            import textwrap

            value = _static_text_value(node.args[0], bindings)
            if isinstance(value, str):
                return textwrap.dedent(value)
        raise ValueError(
            "Factory string must be static; arbitrary calls are not executed"
        )
    return ast.literal_eval(node)


def factory_source(code, allow_generator=True):
    tree = ast.parse(code)
    definitions = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "build_model"
    ]
    if not definitions and allow_generator:
        bindings = {}
        payloads = []
        for node in tree.body:
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = [t.id for t in targets if isinstance(t, ast.Name)]
            try:
                value = _static_text_value(node.value, bindings)
            except (ValueError, TypeError) as error:
                if "MODEL_FACTORY_CONTENT" in names:
                    raise ValueError(
                        "MODEL_FACTORY_CONTENT must be statically resolvable; return the factory module directly. Cause: "
                        + str(error)
                    ) from None
                continue
            for name in names:
                bindings[name] = value
            if "MODEL_FACTORY_CONTENT" in names:
                payloads.append(value)
        if len(payloads) == 1 and isinstance(payloads[0], str):
            return factory_source(payloads[0], allow_generator=False)
    # Match AGE-ML's import semantics. AST is used only to locate the factory
    # (or extract an existing static payload), never to whitelist Python style.
    # Actual import, construction, fit and prediction errors belong to execution.
    return code


PYTHON_TEMPLATE = """Generate a SELF-CONTAINED Python factory MODULE using {selected_tool}.
Return the module code directly, not a program that writes another Python file.
Define def build_model(X_train, y_train), returning a fresh unfitted fit/predict
wrapper. Include imports and wrapper classes in the same module. Module scope
may use normal Python initialization, including scorer construction, logging,
computed constants and helper objects. Do not fit,
read CSVs, predict, execute an evaluator, or use an if __name__ main block there.
The harness handles saving model_factory.py and ten-fold evaluation.
All learned preprocessing must be fitted within each fold's fit(X,y), using only
its training rows. Seed 42. AutoGluon and the selected library remain allowed.
Use fresh artifact directories per fold and bound each fit so ALL ten fits finish
inside the remaining shared deadline; do not assume 360 seconds per fold.
Return Python code in a python fenced block.

### Task
{task_description}
### Data and factory contract
{data_prompt}
### Library guidance (ignore incompatible standalone/holdout/export examples)
{tool_prompt}
### Native debug/evolution context
{code_improvement_prompt}
### Previous errors
{all_previous_error_analyses}
### Reference tutorials (library usage, not the evaluation/export contract)
{tutorial_prompt}
"""

BASH_TEMPLATE = """Generate a minimal BASH INSTALLATION script for the factory module below.
{environment_prompt}
Prepare this node's conda_env with numpy, pandas, scikit-learn and the selected
ML library. Use its interpreter/pip, not the orchestration or AGE-ML environment.
Do NOT run training, predictions, holdout evaluation, file-generation scripts,
or the factory module itself. The harness invokes the trusted ten-fold evaluator
after installation. Propagate installation errors. Return a bash fenced block.
### Factory module (dependencies must support these imports)
{python_code}
### Previous errors
{all_previous_error_analyses}
### Previous installation attempt
{previous_bash_script}
"""


def factory_template(prompt_name):
    return {"PythonCoderPrompt": PYTHON_TEMPLATE, "BashCoderPrompt": BASH_TEMPLATE}.get(
        prompt_name
    )


def prompt_contract(manager, prompt_name):
    if not os.environ.get("AGEML_CV_INPUT") or factory_template(prompt_name) is None:
        return ""
    if prompt_name == "PythonCoderPrompt":
        text = "\nMANDATORY MATCHED-CV: return the factory MODULE directly, not MODEL_FACTORY_CONTENT or a file writer. Imports, comments, blank lines and helper classes/functions are allowed. No module-level fitting or main block."
    else:
        text = "\nMANDATORY MATCHED-CV: return only a conda_env dependency-installation Bash script. Do not execute training or factory-generation code."
    if manager is not None:
        folder = Path(manager.get_iteration_folder(manager.current_node))
        text += "\nExact candidate environment: " + str(folder / "conda_env")
    if os.environ.get("AGEML_CV_DEADLINE"):
        remaining = max(0, int(float(os.environ["AGEML_CV_DEADLINE"]) - time.time()))
        text += f"\nRemaining TOTAL search time, including all ten fits and overhead: {remaining} seconds."
    return text


def normalize_manager_source(manager):
    if not os.environ.get("AGEML_CV_INPUT"):
        return
    original = manager.current_node.python_code
    try:
        normalized = factory_source(original)
    except (ValueError, SyntaxError):
        return  # pre_execution_error sends the concrete issue to native debugging
    if normalized != original:
        manager.current_node.python_code = normalized
        manager.save_and_log_states(
            content={
                "normalization": "static MODEL_FACTORY_CONTENT extraction",
                "original_sha256": sha256_bytes(original.encode()),
                "factory_sha256": sha256_bytes(normalized.encode()),
            },
            save_name="factory_source_normalization.json",
            per_iteration=True,
            add_uuid=False,
        )


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
