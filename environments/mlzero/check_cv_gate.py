"""Execute the patched native selection method with real ten-fold toy fits, no LLM."""

import argparse
import ast
import json
import logging
import os
import shlex
import shutil
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


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


def native_prompt_smoke(source):
    """Exercise real patched build/_build methods and the exact prompt saved to disk."""

    def method(file, cls_name, name):
        tree = ast.parse(
            (source / "src/autogluon/assistant/prompts" / file).read_text()
        )
        cls = next(
            n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls_name
        )
        node = next(
            n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name
        )
        ns = {"logger": logging.getLogger("cv-prompt-smoke")}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), ns)  # noqa: S102 -- verified native source only
        return ns[name]

    build = method("base_prompt.py", "BasePrompt", "build")
    old = os.environ.get("AGEML_CV_INPUT")
    os.environ["AGEML_CV_INPUT"] = "/synthetic-input"
    try:
        for role, file in [
            ("PythonCoderPrompt", "python_coder_prompt.py"),
            ("BashCoderPrompt", "bash_coder_prompt.py"),
        ]:
            saved = {}

            def save(content, save_name, saved=saved, **kwargs):
                saved[save_name] = content

            manager = SimpleNamespace(
                time_step=0,
                current_node=object(),
                get_iteration_folder=lambda n: "/synthetic-node",
                save_and_log_states=save,
            )
            cls = type(
                role, (), {"build": build, "_build": method(file, role, "_build")}
            )
            prompt = cls()
            prompt.manager = manager
            prompt.template = "OLD FULL-SCRIPT TEMPLATE MUST NOT REMAIN"
            prompt.maybe_apply_meta_prompting = lambda: None
            prompt._generate_code_improvement_prompt = lambda: (
                "native debug/evolution context"
            )
            prompt._generate_validation_prompt = lambda: (
                "LEGACY HOLDOUT MUST NOT REMAIN"
            )
            prompt.get_env_prompt = lambda: "install into /synthetic-node/conda_env"
            prompt.llm_config = SimpleNamespace(add_coding_format_instruction=False)
            values = {
                "selected_tool": "selected-tool",
                "task_description": "classification task",
                "data_prompt": "dataset metadata",
                "tool_prompt": "library guidance",
                "all_previous_error_analyses": "errors",
                "tutorial_prompt": "library tutorials",
                "python_code": "factory module",
                "previous_bash_script": "old installer",
            }
            prompt.render = lambda additional, prompt=prompt, values=values: (
                prompt.template.format_map({**values, **additional})
            )
            rendered = prompt.build()
            name = (
                "python_coder_prompt.txt"
                if role == "PythonCoderPrompt"
                else "bash_coder_prompt.txt"
            )
            assert saved[name] == rendered and "MANDATORY MATCHED-CV" in rendered
            assert (
                "OLD FULL-SCRIPT" not in rendered and "LEGACY HOLDOUT" not in rendered
            )
            assert "/synthetic-node/conda_env" in rendered
    finally:
        if old is None:
            os.environ.pop("AGEML_CV_INPUT", None)
        else:
            os.environ["AGEML_CV_INPUT"] = old


def run_check(source):
    import pandas as pd
    from sklearn.model_selection import KFold, StratifiedKFold

    from ageml_mlzero_benchmark import sha256_file, write_json

    simulate = native_simulate(source)
    native_prompt_smoke(source)
    results = {}
    for task in (
        "classification",
        "regression",
        "legacy-holdout",
        "broken-factory",
        "wrapped-factory",
        "unresolved-wrapper",
        "module-initialization",
    ):
        with tempfile.TemporaryDirectory(prefix="mlzero-cv-gate-") as tmp:
            root = Path(tmp)
            prepared = root / "input"
            prepared.mkdir()
            node_dir = root / "node"
            output = node_dir / "output"
            output.mkdir(parents=True)
            python = node_dir / "conda_env/bin/python"
            python.parent.mkdir(parents=True)
            # Keep this toy candidate in the active environment on Python 3.13.
            python.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} "$@"\n')
            python.chmod(0o755)
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
            if task == "module-initialization":
                code = (
                    "import logging\n"
                    "from sklearn.metrics import make_scorer, balanced_accuracy_score\n"
                    "SCORER = make_scorer(balanced_accuracy_score)\n"
                    "logging.getLogger('autogluon').setLevel(logging.WARNING)\n"
                    "CONFIG = dict(seed=42)\n" + code
                )
            if task == "wrapped-factory":
                code = (
                    "MODEL_FACTORY_CONTENT = "
                    + repr(code)
                    + "\nraise RuntimeError('Never execute this generator')\n"
                )
            if task == "unresolved-wrapper":
                code = "MODEL_FACTORY_CONTENT = f'def build_model(x,y): return {undefined_model}'"
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
                if task in (
                    "classification",
                    "regression",
                    "wrapped-factory",
                    "module-initialization",
                ):
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
                    if task == "unresolved-wrapper":
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
