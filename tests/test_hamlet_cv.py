"""Matched protocol tests: exact folds, means, seeds, tampering and deadline export."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold, StratifiedKFold

from ageml_mlzero_benchmark import hamlet_cv as cv
from ageml_mlzero_benchmark import mlzero, sha256_file, write_json


def make_contract(frame, task):
    splitter = (
        StratifiedKFold(10, shuffle=False)
        if task == "classification"
        else KFold(10, shuffle=False)
    )
    return {
        "evaluation_protocol": "hamlet_cv",
        "split_seed": 42,
        "source_row_count": len(frame),
        "task": task,
        "target_column": "y",
        "cv_folds": [
            {"train": a.tolist(), "validation": b.tolist()}
            for a, b in splitter.split(frame[["x"]], frame.y)
        ],
    }


def evaluator():
    spec = importlib.util.spec_from_file_location("trusted_cv_evaluator", cv.EVALUATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluated(tmp_path, task="regression"):
    frame = pd.DataFrame(
        {
            "x": range(40),
            "y": ([0] * 20 + [10] * 20) if task == "regression" else ["a", "b"] * 20,
        }
    )
    data = tmp_path / "data.csv"
    frame.to_csv(data, index=False)
    contract = make_contract(frame, task)
    contract["source_sha256"] = sha256_file(data)
    write_json(tmp_path / "cv_contract.json", contract)
    (tmp_path / "evaluate_cv.py").write_bytes(cv.EVALUATOR.read_bytes())
    output = tmp_path / "output"
    output.mkdir()
    source = (
        'from sklearn.dummy import DummyRegressor\ndef build_model(X_train,y_train):\n return DummyRegressor(strategy="constant",constant=0)\n'
        if task == "regression"
        else "from sklearn.tree import DecisionTreeClassifier\ndef build_model(X_train,y_train):\n return DecisionTreeClassifier(random_state=999,max_depth=1)\n"
    )
    factory = output / "model_factory.py"
    factory.write_text(source)
    record = evaluator().evaluate(factory, tmp_path / "cv_contract.json", output)
    # Executed module path and copied trusted evaluator contain identical bytes.
    return frame, output, record


def test_mean_fold_rmse_is_not_pooled_rmse(tmp_path):
    frame, out, record = evaluated(tmp_path)
    assert record["cv_score"] == 5.0
    assert record["cv_score"] != np.sqrt(np.mean(frame.y**2))
    checked = cv.trusted_cv(out, tmp_path)
    assert checked["cv_score"] == 5.0 and checked["test_score"] is None
    assert len(checked["cv_fold_scores"]) == 10


def test_fit_random_states_are_reset_for_every_fold(tmp_path):
    _, out, record = evaluated(tmp_path, "classification")
    assert record["effective_random_states"] == [{"random_state": 42}] * 10
    assert (
        cv.trusted_cv(out, tmp_path)["score_role"]
        == "selection_cv_not_independent_test"
    )


def test_oof_fold_tampering_is_rejected(tmp_path):
    _, out, _ = evaluated(tmp_path)
    p = pd.read_csv(out / "cv_predictions.csv")
    p.loc[0, "fold"] = 9
    p.to_csv(out / "cv_predictions.csv", index=False)
    with pytest.raises(ValueError, match="fold assignment"):
        cv.trusted_cv(out, tmp_path)


def test_modified_factory_is_rejected(tmp_path):
    _, out, _ = evaluated(tmp_path)
    (out / "model_factory.py").write_text("changed")
    with pytest.raises(ValueError, match="provenance"):
        cv.trusted_cv(out, tmp_path)


def test_bootstrap_freezes_current_campaign_not_old_reference(tmp_path, monkeypatch):
    monkeypatch.setattr(mlzero, "ROOT", tmp_path)
    source = tmp_path / "age-data/regression/toy.csv"
    source.parent.mkdir(parents=True)
    frame = pd.DataFrame({"x": range(40), "y": range(40)})
    frame.to_csv(source, index=False)
    contract = make_contract(frame, "regression")
    contract["source_sha256"] = sha256_file(source)
    frozen = {
        "campaign_identity": "actual-campaign",
        "model": {"model_name": "gemini-2.5-flash", "seed": 42, "temperature": 0.0},
        "config": {
            "run_seed": 42,
            "time_budget_seconds": 3600,
            "repetitions": [1, 2, 3, 4, 5],
            "datasets": {"regression": {"toy": 1}},
        },
        "source_evidence": {
            "toy": {
                "source_sha256": sha256_file(source),
                "split_identity": "exact-age-split",
                "manifest": contract,
            }
        },
    }
    age = tmp_path / "age-campaign"
    write_json(age / "campaign.json", frozen)
    cfg = cv.bootstrap(age, tmp_path / "age-data", tmp_path / "config.yml")
    assert cfg["budget"]["search_wall_seconds"] == 3600
    assert cfg["llm"]["settings"] == {"seed": 42, "temperature": 0.0}
    rendered = mlzero.render_config(cfg)
    assert all(
        rendered[a]["provider"] == "gemini"
        and rendered[a]["seed"] == 42
        and rendered[a]["temperature"] == 0
        for a in mlzero.AGENTS
    )
    cv.validate_inputs(cfg)
    prepared = Path(cfg["dataset_entries"][0]["prepared"])
    (prepared / "data.csv").write_text("changed")
    with pytest.raises(ValueError, match="Changed input"):
        cv.validate_inputs(cfg)


def test_original_holdout_protocol_is_rejected():
    frame = pd.DataFrame({"x": range(40), "y": range(40)})
    contract = make_contract(frame, "regression")
    contract["evaluation_protocol"] = "holdout"
    with pytest.raises(ValueError, match="hamlet_cv"):
        cv.validate_contract(contract, frame)


@pytest.mark.parametrize("late_commit", [False, True])
def test_native_cv_scheduler_deadline_and_resume(tmp_path, monkeypatch, late_commit):
    import subprocess
    from types import SimpleNamespace

    monkeypatch.setattr(mlzero, "ROOT", tmp_path)
    frame = pd.DataFrame({"x": range(40), "y": [0] * 20 + [10] * 20})
    source = tmp_path / "data/regression/toy.csv"
    source.parent.mkdir(parents=True)
    frame.to_csv(source, index=False)
    contract = make_contract(frame, "regression")
    contract["source_sha256"] = sha256_file(source)
    age = tmp_path / "age"
    write_json(
        age / "campaign.json",
        {
            "campaign_identity": "age-actual",
            "model": {"model_name": "gemini-2.5-flash", "seed": 42, "temperature": 0.0},
            "config": {
                "run_seed": 42,
                "time_budget_seconds": 3600,
                "repetitions": [1],
                "datasets": {"regression": {"toy": 1}},
            },
            "source_evidence": {
                "toy": {
                    "source_sha256": sha256_file(source),
                    "split_identity": "age-split",
                    "manifest": contract,
                }
            },
        },
    )
    cfg = cv.bootstrap(age, tmp_path / "data", tmp_path / "config.yml")
    prepared = Path(cfg["dataset_entries"][0]["prepared"])
    monkeypatch.setenv("GEMINI_API_KEY", "test-only")
    monkeypatch.setattr(cv, "check_image", lambda cfg: {})
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=0)
    )
    monkeypatch.setattr(
        subprocess, "check_output", lambda *a, **kw: "sha256:test-image"
    )
    monkeypatch.setattr(
        cv.shutil, "disk_usage", lambda root: SimpleNamespace(free=100 * 1024**3)
    )
    captured = {}
    calls = []

    def supervise(argv, env, timeout, log, name, grace):
        attempt = log.parent
        output = attempt / "mlzero/node_0/output"
        output.mkdir(parents=True)
        factory = output / "model_factory.py"
        factory.write_text(
            'from sklearn.dummy import DummyRegressor\ndef build_model(X_train,y_train):\n return DummyRegressor(strategy="constant",constant=0)\n'
        )
        evaluator().evaluate(factory, prepared / "cv_contract.json", output)
        incumbent = {
            "output_folder": "/bench/run/mlzero/node_0/output",
            "native_validation_score": -5.0,
            "ts": 5000 if late_commit else 1001,
        }
        captured["collection"] = {
            "commits": [incumbent],
            "events": [
                {
                    "event": "chat_created",
                    "settings": {
                        "provider": "gemini",
                        "model": "gemini-2.5-flash",
                        "seed": 42,
                        "temperature": 0.0,
                    },
                }
            ],
            "usage_rows": [],
            "resources": None,
            "returned": True,
            "native_stop": "done",
        }
        calls.append((argv, timeout, env["PYTHONHASHSEED"]))
        return {
            "started_at": 1000,
            "wall_s": 10,
            "exit_code": 0,
            "timed_out": False,
            "launch_error": None,
        }

    monkeypatch.setattr(cv.runner, "_supervise", supervise)
    monkeypatch.setattr(mlzero, "collect", lambda *a: captured["collection"])
    result = cv.run(cfg)
    assert result[0]["status"] == ("no_artifact" if late_commit else "completed")
    assert result[0]["test_score"] is None
    assert calls[0][1:] == (3600, "42")
    assert "timeout" in calls[0][0] and "3600s" in calls[0][0]
    assert cv.run(cfg)[0]["run_id"] == result[0]["run_id"]
    assert len(calls) == 1
