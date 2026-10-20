"""Data, evaluation, import and accounting contracts (no processes launched)."""

import json
from pathlib import Path

import pandas as pd
import pytest

from ageml_mlzero_benchmark import (
    accounting,
    datasets,
    evaluation,
    importers,
    read_json,
)
from ageml_mlzero_benchmark.reference import pair_checks, V1_PROTOCOL
from ageml_mlzero_benchmark.reporting import outcome

from tests.conftest import synthetic


def prepared(project: Path, key: str) -> Path:
    return datasets.find_prepared(project / "artifacts", key)


def test_split_disjoint_complete_and_stable(project: Path) -> None:
    d = prepared(project, "toyclf")
    split, m = read_json(d / "split.json"), read_json(d / "manifest.json")
    parts = [set(split[k]) for k in ("train", "validation", "test")]
    assert not (parts[0] & parts[1] or parts[0] & parts[2] or parts[1] & parts[2])
    assert set().union(*parts) == set(range(200))
    assert m["counts"] == {"train": 128, "validation": 32, "test": 40}  # 64/16/20
    df = pd.read_csv(d / "private" / "export.csv")
    again = datasets.make_split(df, "class", "classification")
    assert datasets.split_hash(again) == m["split_hash"]


def test_split_matches_ageml_two_stage_dataframe_split() -> None:
    """Row-id splitting reproduces AGE-ML's train_test_split calls on the frame."""
    from sklearn.model_selection import train_test_split

    df = synthetic("classification")
    dev, test = train_test_split(
        df, test_size=0.2, stratify=df["class"], random_state=42
    )
    train, val = train_test_split(
        dev, test_size=0.2, stratify=dev["class"], random_state=42
    )
    s = datasets.make_split(df, "class", "classification")
    assert (s["train"], s["validation"], s["test"]) == (
        list(train.index),
        list(val.index),
        list(test.index),
    )


def test_reference_split_hash_mismatch_is_refused(project: Path) -> None:
    from tests.conftest import entry

    e = entry("toyclf", "classification") | {"split_hash": "0" * 64}
    with pytest.raises(ValueError, match="split hash"):
        datasets.prepare_dataset(
            e, project / "art2", export_csv=project / "exports" / "toyclf.csv"
        )


def test_explicit_target_class_mapping_and_units(project: Path) -> None:
    m = read_json(prepared(project, "toyclf") / "manifest.json")
    assert m["target"] == "class" and m["class_mapping"] == {"neg": 0, "pos": 1}
    r = read_json(prepared(project, "toyreg") / "manifest.json")
    assert (
        r["class_mapping"] is None and r["target_summary"]["min"] == 100.0
    )  # original units
    with pytest.raises(ValueError, match="no explicit target"):
        datasets.prepare_dataset(
            {"key": "k", "target": None, "task": "regression"}, project
        )


def test_search_input_excludes_test_and_full_data(project: Path) -> None:
    d = prepared(project, "toyclf")
    search = {p.name for p in (d / "search").iterdir()}
    assert search == datasets.SEARCH_FILES
    feats = pd.read_csv(d / "search" / "val_features.csv")
    assert "class" not in feats.columns and "row_id" not in feats.columns
    n = sum(len(pd.read_csv(d / "search" / f)) for f in ("train.csv", "val.csv"))
    assert n == 160  # no test rows
    (d / "search" / "export.csv").write_text("x")
    with pytest.raises(RuntimeError):
        datasets.assert_search_isolated(d / "search")


def test_prediction_checks() -> None:
    truth = pd.DataFrame({"row_id": [5, 6, 7], "target": ["a", "b", "a"]})
    ok = pd.DataFrame(
        {"row_id": [7, 5, 6], "prediction": ["a", "a", "b"]}
    )  # shuffled is fine
    assert (
        evaluation.evaluate(
            ok, truth, "classification", "balanced_accuracy", ["a", "b"]
        )["score"]
        == 1.0
    )
    for bad, msg in [
        (ok.iloc[:2], "missing"),
        (pd.concat([ok, ok.iloc[:1]]), "duplicated"),
        (ok.assign(prediction=["a", "z", "b"]), "outside class set"),
        (ok.assign(row_id=[5, 6, 99]), "unexpected"),
    ]:
        res = evaluation.evaluate(
            bad, truth, "classification", "balanced_accuracy", ["a", "b"]
        )
        assert (
            not res["valid"]
            and res["score"] is None
            and any(msg in p for p in res["problems"])
        )
    rt = pd.DataFrame({"row_id": [1, 2], "target": ["1.0", "3.0"]})
    nan = pd.DataFrame({"row_id": [1, 2], "prediction": ["1", "nan"]})
    assert not evaluation.evaluate(nan, rt, "regression", "rmse", None)["valid"]


def test_metric_direction() -> None:
    assert evaluation.score("rmse", [0, 0], [3, 4]) == pytest.approx((12.5) ** 0.5)
    assert evaluation.better("rmse", 1.0, 2.0) and evaluation.better(
        "balanced_accuracy", 0.9, 0.8
    )


def test_normalize_requires_row_alignment(tmp_path: Path) -> None:
    f = tmp_path / "p.csv"
    f.write_text("prediction\na\nb\n")
    assert list(evaluation.normalize_predictions(f, [10, 11]).row_id) == [10, 11]
    with pytest.raises(ValueError):
        evaluation.normalize_predictions(f, [1, 2, 3])


def test_usage_null_and_no_double_counting() -> None:
    assert accounting.summarize_usage([])["input_tokens"] is None
    assert (
        accounting.summarize_usage([{"usage_metadata": None}])["usage_coverage"]
        == "none"
    )
    rows: list[dict] = [
        # reasoning inside completion tokens (total = in + out): reported, not added
        {
            "usage_metadata": {
                "input_tokens": 100,
                "output_tokens": 20,
                "input_token_details": {"cache_read": 40},
                "output_token_details": {"reasoning": 5},
            },
            "token_usage": {"total_tokens": 120},
        },
        # reasoning billed outside completion tokens (total = in + out + reasoning)
        {
            "usage_metadata": {
                "input_tokens": 10,
                "output_tokens": 2,
                "output_token_details": {"reasoning": 3},
            },
            "token_usage": {"total_tokens": 15},
        },
        {"usage_metadata": None},
    ]
    u = accounting.summarize_usage(rows)
    assert (
        u["input_tokens"],
        u["output_tokens"],
        u["cache_read_tokens"],
        u["reasoning_tokens"],
    ) == (110, 22, 40, 8)
    assert u["billable_output_tokens"] == 25 and u["usage_coverage"] == "partial"
    price = {
        "version": "v",
        "models": {
            "m": {
                "input_per_mtok": 1.0,
                "cached_input_per_mtok": 0.1,
                "output_per_mtok": 10.0,
            }
        },
    }
    assert accounting.estimate_cost(u, "m", price)["cost"] == pytest.approx(
        (70 + 4 + 250) / 1e6
    )
    assert accounting.estimate_cost(u, "unknown", price)["cost"] is None


AGE_CSV = """OpenML ID,Dataset,Problem,Best Pipeline,Validation Metric,Best Test-set Score,Baseline,Features,Instances,Numeric Features,Discrete Features,Actual Time,Equivalent Time,LLM Inference Time,ML Training Time,Workers,Pipeline Budget,Actual Pipelines,Input Tokens,Output Tokens,Total Tokens,Putative Cost ($)
40983,wilt,classification,Pipeline 28,balanced_accuracy,0.96,hamlet,5,4839,5,0,0:03:24,0:30:28,0:15:26,0:15:01,8,30,30,622179,31941,654120,0.20
196,auto_mpg,regression,Pipeline 25,rmse,,to fill,7,398,7,0,2:00:38,,,,8,30,0,,,,
"""


def test_historical_import_cannot_masquerade(tmp_path: Path) -> None:
    f = (
        tmp_path
        / "results"
        / "gemini-x"
        / "specification"
        / "general"
        / "run-3"
        / "results.csv"
    )
    f.parent.mkdir(parents=True)
    f.write_text(AGE_CSV)
    recs = importers.import_results(tmp_path / "results", "c")
    wilt, mpg = recs
    assert (
        wilt["repetition_id"] == 3
        and wilt["comparability_status"] == "historical_protocol"
    )
    assert (
        wilt["source_commit"] is None
        and wilt["protocol_id"] is None
        and wilt["dataset_hash"] is None
    )
    assert (
        wilt["requested_model"] is None
        and wilt["extra"]["model_from_path"] == "gemini-x"
    )
    assert wilt["search_wall_s"] == 204 and wilt["extra"]["equivalent_time_s"] == 1828
    assert (
        mpg["artifact_status"] == "no_artifact" and mpg["input_tokens"] is None
    )  # null, not 0
    mlz = dict(
        wilt,
        system="mlzero",
        protocol_id="ageml-holdout-v2",
        dataset_hash="h",
        split_hash="s",
        target="class",
        requested_model="gemini-x",
    )
    chk = pair_checks(wilt, mlz, "ageml-holdout-v2")
    assert not chk["predictive_eligible"] and chk["checks"]["protocol"] == "unknown"
    v1 = dict(wilt, protocol_id=V1_PROTOCOL)
    assert pair_checks(v1, mlz, "ageml-holdout-v2")["checks"]["protocol"] == "fail"


def test_corrected_manifest_import_is_eligible(tmp_path: Path) -> None:
    run = tmp_path / "run-1"
    run.mkdir()
    (run / "results.csv").write_text(AGE_CSV)
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "protocol_id": "ageml-holdout-v2",
                "source_commit": "abc",
                "llm": {"requested_model": "m", "reported_model": "m"},
                "datasets": [
                    {
                        "key": "wilt",
                        "export_sha256": "h",
                        "split_hash": "s",
                        "target": "class",
                    }
                ],
            }
        )
    )
    wilt = importers.import_results(run / "results.csv", "c")[0]
    assert wilt["comparability_status"] == "corrected_protocol"
    mlz = dict(wilt, system="mlzero", requested_model="m")
    assert pair_checks(wilt, mlz, "ageml-holdout-v2")["predictive_eligible"]


def test_preregistered_outcome_rule() -> None:
    v = {"artifact_status": "valid", "objective": "rmse", "test_score": 1.0}
    w = dict(v, test_score=2.0)
    inv = {"artifact_status": "export_failed"}
    assert outcome(v, w) == "ageml" and outcome(w, v) == "mlzero"
    assert (
        outcome(inv, v) == "mlzero"
        and outcome(inv, inv) == "tie"
        and outcome(None, inv) == "tie"
    )
