"""End-to-end runner behaviour against the fake MLZero CLI (local executor)."""

import copy
from pathlib import Path

import pytest
import yaml

from ageml_mlzero_benchmark import mlzero, read_json, runner
from ageml_mlzero_benchmark.reporting import write_report

from tests.conftest import ENDPOINT


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_KEY", "secret")
    monkeypatch.setenv("FAKE_MLZERO_MODE", "ok")


def run(resolved: dict, only: str = "toyclf", **kw) -> dict:
    return runner.run_campaign(resolved, only=[only], **kw)[0]


def attempt_dir(resolved: dict, rec: dict) -> Path:
    return Path(resolved["outputs"]["artifact_root"]) / rec["evidence"]["attempt_dir"]


def test_end_to_end_valid(resolved: dict) -> None:
    rec = run(resolved)
    assert rec["artifact_status"] == "valid", rec["protocol_checks"]
    assert rec["termination_reason"] == "safety_ceiling"  # fake stops at -n
    assert rec["selected_candidate_id"] == "node_2"
    assert rec["test_score"] == pytest.approx(0.5)  # majority class -> BA 0.5
    assert rec["protocol_checks"]["llm_settings"]["status"] == "pass"
    assert rec["protocol_checks"]["inference_fit_calls"] == []
    assert rec["comparability_status"] == "provisional_protocol"  # reference is v1
    assert (
        rec["llm_calls"] == 2
        and rec["output_tokens"] == 20
        and rec["reasoning_tokens"] == 8
    )
    assert (
        rec["first_valid_artifact_s"] is not None and rec["first_valid_artifact_s"] >= 0
    )
    pred = (
        attempt_dir(resolved, rec) / "inference" / "out" / "predictions.normalized.csv"
    )
    assert pred.read_text().startswith("row_id,prediction")


def test_regression_units(resolved: dict) -> None:
    rec = run(resolved, "toyreg")
    assert (
        rec["artifact_status"] == "valid"
        and rec["objective"] == "rmse"
        and not rec["maximize"]
    )
    chk = rec["protocol_checks"]["native_vs_trusted_validation"]
    assert chk["expected_native"] == pytest.approx(
        -rec["validation_score"]
    )  # native is negated RMSE


def test_runner_launches_documented_cli(resolved: dict, tmp_path: Path) -> None:
    argv, env = mlzero.search_command(
        resolved, tmp_path, tmp_path / "in", tmp_path, "c"
    )
    tail = argv[argv.index("--") + 3 :]
    assert (
        tail[:2] == ["-i", str(tmp_path / "in")] and "--continuous_improvement" in tail
    )
    assert tail[tail.index("-n") + 1] == "3" and tail[tail.index("-c") + 1].endswith(
        "mlzero.yaml"
    )
    assert "secret" not in " ".join(argv) and env["OPENAI_BASE_URL"] == ENDPOINT


def test_every_agent_gets_requested_model(resolved: dict) -> None:
    resolved = copy.deepcopy(resolved)
    resolved["llm"]["settings"] = {"temperature": 0.3, "reasoning_effort": "low"}
    cfg = mlzero.render_config(resolved)
    for agent in mlzero.AGENTS:
        assert (
            cfg[agent]["model"] == "gemini-test" and cfg[agent]["provider"] == "openai"
        )
        assert (
            cfg[agent]["proxy_url"] == ENDPOINT
            and cfg[agent]["reasoning_effort"] == "low"
        )
        assert cfg[agent]["temperature"] == 0.3


def test_native_selection_survives_trusted_scores(
    resolved: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_MLZERO_MODE", "native_vs_trusted")
    rec = run(resolved)
    # node_1 has the higher native score but worse predictions: it stays selected
    assert rec["selected_candidate_id"] == "node_1"
    assert rec["artifact_status"] == "valid"


def test_timeout_keeps_last_committed_incumbent(
    resolved: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_MLZERO_MODE", "hang")
    resolved["budget"]["search_wall_seconds"] = 4
    rec = run(resolved)
    assert rec["termination_reason"] == "timeout"
    assert rec["selected_candidate_id"] == "node_1"  # node 2 was never committed
    assert rec["artifact_status"] == "valid"
    assert rec["llm_calls"] == 2  # usage of the unfinished node survives the kill
    assert 4 <= rec["search_wall_s"] < 30


def test_failed_export_is_failure_without_fallback(
    resolved: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_MLZERO_MODE", "no_export")
    rec = run(resolved)
    assert rec["artifact_status"] == "export_failed" and rec["test_score"] is None
    assert not (attempt_dir(resolved, rec) / "inference").exists()


def test_fit_during_inference_is_violation(
    resolved: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_MLZERO_MODE", "fit_in_predict")
    rec = run(resolved)
    assert rec["artifact_status"] == "protocol_violation" and rec["test_score"] is None
    assert any("fit" in c for c in rec["protocol_checks"]["inference_fit_calls"])


def test_resume_identity_and_retries(
    resolved: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = run(resolved)
    with pytest.raises(RuntimeError, match="--resume"):
        run(resolved)
    assert (
        run(resolved, resume=True)["attempt_id"] == first["attempt_id"]
    )  # skipped, not duplicated
    ids = {i["run_id"] for i in runner.plan(resolved)}
    for path, value in [
        (("budget", "search_wall_seconds"), 61),
        (("llm", "model"), "other"),
        (("mlzero", "commit"), "deadbeef"),
        (("repetitions",), [2]),
    ]:
        changed = copy.deepcopy(resolved)
        d = changed
        for k in path[:-1]:
            d = d[k]
        d[path[-1]] = value
        assert not ids & {i["run_id"] for i in runner.plan(changed)}, path
    # a crash of the system under test is an outcome, never retried
    monkeypatch.setenv("FAKE_MLZERO_MODE", "crash")
    rec = run(resolved, "toyreg")
    assert (
        rec["termination_reason"] == "process_error"
        and rec["artifact_status"] == "no_artifact"
    )
    assert rec["extra"]["final"] and rec["attempt_id"].endswith("attempt-1")


def test_infrastructure_error_retried_once_and_kept(resolved: dict) -> None:
    resolved["mlzero"]["python"] = "/nonexistent/python"
    resolved["mlzero"]["command"] = ["/nonexistent/mlzero"]
    rec = run(resolved)
    assert rec["termination_reason"] == "infrastructure_error"
    assert rec["attempt_id"].endswith("attempt-2")
    assert (
        len(list(attempt_dir(resolved, rec).parent.glob("attempt-*/record.json"))) == 2
    )


def test_interrupted_attempt_gets_new_attempt(resolved: dict) -> None:
    item = next(i for i in runner.plan(resolved) if i["entry"]["key"] == "toyclf")
    stale = runner.run_dir(resolved, item) / "attempt-1"
    (stale / "bench").mkdir(parents=True)
    (stale / "bench" / "usage.jsonl").write_text(
        '{"usage_metadata": {"input_tokens": 7, "output_tokens": 1}}\n'
    )
    rec = run(resolved, resume=True)
    old = read_json(stale / "record.json")
    assert (
        old["termination_reason"] == "harness_interrupted" and old["input_tokens"] == 7
    )
    assert rec["attempt_id"].endswith("attempt-2") and rec["artifact_status"] == "valid"


def test_report_without_credentials(
    resolved: dict, monkeypatch: pytest.MonkeyPatch, project: Path
) -> None:
    run(resolved)
    monkeypatch.setenv("FAKE_MLZERO_MODE", "no_export")
    run(resolved, "toyreg")
    monkeypatch.delenv("FAKE_KEY")
    out = write_report(runner.campaign_dir(resolved), "ageml-holdout-v2")
    import pandas as pd

    summary = pd.read_csv(out / "comparison.csv")
    reg = summary[summary.dataset == "toyreg"].iloc[0]
    assert (
        reg.planned_runs == 1 and reg.valid_runs == 0 and reg.n_export_failed == 1
    )  # denominator kept
    assert (out / "table_balanced_accuracy.tex").exists() and (
        out / "test_rmse.pdf"
    ).exists()
    assert (
        yaml.safe_load((project / "configs" / "c.yml").read_text())["campaign_id"]
        == "test"
    )
