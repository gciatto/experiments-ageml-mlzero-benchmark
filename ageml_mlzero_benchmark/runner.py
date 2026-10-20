"""MLZero scheduling, deadline supervision, final inference and resume.

Layout: ``<root>/campaigns/<campaign>/runs/<dataset>/rep-<r>/<run_id>/attempt-<n>/``
with ``record.json`` as the terminal, authoritative record of each attempt.
"""

import os
import platform
import shutil
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import pandas as pd

from . import (
    append_jsonl,
    canonical_hash,
    new_record,
    read_json,
    sha256_file,
    utcnow,
    write_json,
    OBJECTIVES,
)
from . import accounting, datasets, evaluation, mlzero
from .config import is_corrected

#: retried automatically (new attempt, prior attempts and charges kept)
RETRYABLE = {"infrastructure_error", "harness_interrupted"}


def campaign_dir(resolved: dict[str, Any]) -> Path:
    return (
        Path(resolved["outputs"]["artifact_root"])
        / "campaigns"
        / resolved["campaign_id"]
    )


def protocol_id(resolved: dict[str, Any]) -> str:
    req = resolved["required_protocol_id"]
    return req if is_corrected(resolved) else f"{req}-provisional"


def plan(resolved: dict[str, Any]) -> list[dict[str, Any]]:
    """One entry per logical run. Identity changes with anything that changes the run."""
    root = Path(resolved["outputs"]["artifact_root"])
    m = resolved["mlzero"]
    items = []
    for entry in resolved["dataset_entries"]:
        try:
            prepared = datasets.find_prepared(root, entry["key"])
            manifest = read_json(prepared / "manifest.json")
        except FileNotFoundError:
            prepared, manifest = None, None
        for rep in resolved["repetitions"]:
            ident = {
                "system": "mlzero",
                "mlzero_commit": m["commit"],
                "image": m.get("image"),
                "patches": mlzero.patch_hashes(),
                "dataset": entry["key"],
                "dataset_hash": manifest and manifest["export_sha256"],
                "split_hash": manifest and manifest["split_hash"],
                "protocol_id": protocol_id(resolved),
                "llm": resolved["llm"] | {"api_key_env": None},
                "resources": resolved["resources"],
                "budget": resolved["budget"],
                "inference_timeout": resolved["evaluation"][
                    "inference_timeout_seconds"
                ],
                "mlzero_config": mlzero.render_config(resolved),
                "max_iterations": m["max_iterations"],
                "task_template": canonical_hash(
                    mlzero.TASK_TEMPLATE + mlzero.INSTRUCTION
                ),
                "repetition": rep,
            }
            items.append(
                {
                    "entry": entry,
                    "repetition": rep,
                    "prepared": prepared,
                    "manifest": manifest,
                    "identity": ident,
                    "run_id": canonical_hash(ident)[:16],
                }
            )
    return items


def run_dir(resolved: dict[str, Any], item: dict[str, Any]) -> Path:
    return (
        campaign_dir(resolved)
        / "runs"
        / item["entry"]["key"]
        / f"rep-{item['repetition']}"
        / item["run_id"]
    )


def _supervise(
    argv: list[str],
    env: dict[str, str],
    timeout: float,
    log: Path,
    docker_name: str | None,
    grace: float,
) -> dict[str, Any]:
    """Run until exit or deadline; the deadline covers the whole process tree."""
    started = time.time()
    t0 = time.monotonic()
    with open(log, "ab") as out:
        try:
            proc = subprocess.Popen(
                argv,
                env={**os.environ, **env},
                stdout=out,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except OSError as e:
            return {
                "started_at": started,
                "wall_s": 0.0,
                "exit_code": None,
                "timed_out": False,
                "launch_error": str(e),
            }
        timed_out = False
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            if docker_name:
                subprocess.run(
                    ["docker", "stop", "-t", str(int(grace)), docker_name],
                    capture_output=True,
                )
            else:
                os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=grace + 10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
    res = {
        "started_at": started,
        "wall_s": time.monotonic() - t0,
        "exit_code": proc.returncode,
        "timed_out": timed_out,
        "launch_error": None,
    }
    if docker_name:
        st = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.OOMKilled}}", docker_name],
            capture_output=True,
            text=True,
        )
        res["oom_killed"] = st.stdout.strip() == "true"
        subprocess.run(["docker", "rm", "-f", docker_name], capture_output=True)
    return res


def _termination(sup: dict[str, Any], col: dict[str, Any], docker: bool) -> str:
    if sup["launch_error"] or (docker and sup["exit_code"] == 125):
        return "infrastructure_error"
    if sup["timed_out"]:
        return "timeout"
    if sup.get("oom_killed"):
        return "resource_exhaustion"
    if sup["exit_code"] == 0 and col["returned"]:
        return (
            "safety_ceiling"
            if col["native_stop"] == "max_iterations"
            else "native_completed"
        )
    return "process_error"


def _environment_digest(resolved: dict[str, Any], env_freeze: Path | None) -> str:
    m = resolved["mlzero"]
    if m["executor"] == "docker":
        img = (
            subprocess.run(
                ["docker", "image", "inspect", "-f", "{{.Id}}", m["image"]],
                capture_output=True,
                text=True,
            ).stdout.strip()
            or None
        )
    else:
        img = f"local:{platform.platform()}:python{platform.python_version()}"
    freeze = sha256_file(env_freeze) if env_freeze and env_freeze.exists() else None
    return canonical_hash({"image": img, "node_env_freeze": freeze})


def _score_validation(col: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
    entry, manifest, prepared = item["entry"], item["manifest"], item["prepared"]
    if not col.get("native_results"):
        return {
            "validation_score": None,
            "validation_problems": ["no native results.csv"],
        }
    split = read_json(prepared / "split.json")
    truth = datasets.load_labels(prepared / "private" / "val_labels.csv")
    try:
        pred = evaluation.normalize_predictions(
            col["native_results"], split["validation"], entry["target"]
        )
    except ValueError as e:
        return {"validation_score": None, "validation_problems": [str(e)]}
    classes = list(manifest["class_mapping"]) if manifest["class_mapping"] else None
    ev = evaluation.evaluate(pred, truth, entry["task"], entry["objective"], classes)
    return {"validation_score": ev["score"], "validation_problems": ev["problems"]}


def _infer(
    resolved: dict[str, Any], item: dict[str, Any], attempt: Path, col: dict[str, Any]
) -> dict[str, Any]:
    """Run the frozen selected artifact once on the outer test features."""
    m = resolved["mlzero"]
    docker = m["executor"] == "docker"
    inf = attempt / "inference"
    feats, out = inf / "input", inf / "out"
    feats.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        item["prepared"] / "private" / "test_features.csv", feats / "test_features.csv"
    )
    name = f"ageml-inf-{uuid.uuid4().hex[:10]}"
    argv = mlzero.inference_command(
        resolved, attempt, col["incumbent"], feats, out, name
    )
    env = (
        {}
        if docker
        else {
            "PYTHONPATH": str(mlzero.ROOT / "environments/mlzero/guard"),
            "AGEML_BENCH_FIT_LOG": str(out / "fit_calls.jsonl"),
            # no credentials reach inference
            "OPENAI_API_KEY": "",
            resolved["llm"]["api_key_env"]: "",
        }
    )
    before = col["asset_hashes"]
    sup = _supervise(
        argv,
        env,
        resolved["evaluation"]["inference_timeout_seconds"],
        inf / "inference.log",
        name if docker else None,
        10,
    )
    after = mlzero.sha256_tree(col["node_folder_host"] / "output")
    modified = sorted(k for k, v in before.items() if after.get(k) != v)
    fit_calls = [r["call"] for r in mlzero.read_jsonl(out / "fit_calls.jsonl")]
    result: dict[str, Any] = {
        "inference_s": sup["wall_s"],
        "inference_exit": sup["exit_code"],
        "inference_timed_out": sup["timed_out"],
        "assets_modified": modified,
        "fit_calls": fit_calls,
        "env_freeze": out / "env-freeze.txt",
        "inference_resources": accounting.summarize_resources(
            read_json(out / "resources.json")
            if (out / "resources.json").exists()
            else None
        ),
    }
    if (
        sup["exit_code"] != 0
        or sup["timed_out"]
        or not (out / "predictions.csv").exists()
    ):
        return {**result, "status": "inference_failed"}
    split = read_json(item["prepared"] / "split.json")
    try:
        pred = evaluation.normalize_predictions(
            out / "predictions.csv", split["test"], item["entry"]["target"]
        )
    except ValueError as e:
        return {**result, "status": "invalid_predictions", "problems": [str(e)]}
    pred.to_csv(out / "predictions.normalized.csv", index=False)
    truth = datasets.load_labels(item["prepared"] / "private" / "test_labels.csv")
    classes = (
        list(item["manifest"]["class_mapping"])
        if item["manifest"]["class_mapping"]
        else None
    )
    ev = evaluation.evaluate(
        pred, truth, item["entry"]["task"], item["entry"]["objective"], classes
    )
    if not ev["valid"]:
        return {**result, "status": "invalid_predictions", "problems": ev["problems"]}
    status = "protocol_violation" if (fit_calls or modified) else "valid"
    return {**result, "status": status, "test_score": ev["score"], "problems": []}


def build_record(
    resolved: dict[str, Any],
    item: dict[str, Any],
    attempt: Path,
    attempt_id: str,
    sup: dict[str, Any] | None,
    col: dict[str, Any],
    termination: str,
    inference: dict[str, Any] | None,
) -> dict[str, Any]:
    m, entry, manifest = resolved["mlzero"], item["entry"], item["manifest"]
    root = Path(resolved["outputs"]["artifact_root"])
    usage = accounting.summarize_usage(col["usage_rows"])
    pricing = (
        accounting.load_pricing(Path(resolved["pricing"]))
        if resolved.get("pricing")
        else {}
    )
    cost = accounting.estimate_cost(usage, resolved["llm"]["model"], pricing)
    inc = col.get("incumbent")
    start = sup["started_at"] if sup else None
    first_valid = next(
        (
            c["ts"] - start
            for c in col["commits"]
            if c.get("export_status") == "present" and start
        ),
        None,
    )
    if inc is None:
        status = "no_artifact"
    elif not col.get("export_present"):
        status = "export_failed"
    else:
        status = inference["status"] if inference else "inference_failed"
    val = (
        _score_validation(col, item)
        if inc
        else {"validation_score": None, "validation_problems": []}
    )
    native = inc.get("native_validation_score") if inc else None
    expected = None
    if val["validation_score"] is not None:
        expected = (
            val["validation_score"]
            if OBJECTIVES[entry["objective"]]
            else -val["validation_score"]
        )
    llm_check = mlzero.verify_llm(
        attempt / "bench" / "effective_config.yaml", col["events"], resolved
    )
    rel = lambda p: os.path.relpath(p, root)  # noqa: E731
    return new_record(
        campaign_id=resolved["campaign_id"],
        run_id=item["run_id"],
        attempt_id=attempt_id,
        system="mlzero",
        dataset_key=entry["key"],
        repetition_id=item["repetition"],
        source_commit=m["commit"],
        patch_hash=canonical_hash(mlzero.patch_hashes()),
        environment_digest=_environment_digest(
            resolved, inference["env_freeze"] if inference else None
        ),
        config_hash=canonical_hash(item["identity"]),
        reference_campaign_id=resolved["reference"]["campaign_id"],
        openml_id=entry["openml_id"],
        dataset_version=entry.get("dataset_version"),
        dataset_hash=manifest["export_sha256"],
        target=entry["target"],
        task=entry["task"],
        split_hash=manifest["split_hash"],
        protocol_id=protocol_id(resolved),
        split_seed=manifest.get("split_seed"),
        search_seed=None,  # MLZero exposes no search seed; LLM sampling is not seeded
        provider="google-openai-compat",
        endpoint=resolved["llm"]["endpoint"],
        requested_model=resolved["llm"]["model"],
        reported_model=",".join(usage["reported_models"]) or None,
        agent_settings=mlzero.render_config(resolved)["llm"]
        | {"proxy_url": resolved["llm"]["endpoint"]},
        artifact_status=status,
        termination_reason=termination,
        selected_candidate_id=inc and f"node_{inc['node_id']}",
        selected_artifact_hash=canonical_hash(col["asset_hashes"])
        if inc and col.get("asset_hashes")
        else None,
        comparability_status="corrected_protocol"
        if is_corrected(resolved)
        else "provisional_protocol",
        objective=entry["objective"],
        maximize=OBJECTIVES[entry["objective"]],
        native_validation_score=native,
        validation_score=val["validation_score"],
        test_score=inference.get("test_score")
        if inference and status == "valid"
        else None,
        search_wall_s=sup and sup["wall_s"],
        search_budget_s=resolved["budget"]["search_wall_seconds"],
        first_valid_artifact_s=first_valid,
        inference_s=inference and inference["inference_s"],
        **{
            k: v
            for k, v in accounting.summarize_resources(col["resources"]).items()
            if k in ("cpu_s", "peak_rss_bytes")
        },
        resource_limits={
            k: resolved["resources"].get(k) for k in ("cpus", "memory_gib", "gpus")
        },
        timing_boundary_verified=True,  # clock starts before launch, whole process tree
        llm_calls=usage["llm_calls"],
        input_tokens=usage["input_tokens"],
        output_tokens=usage["output_tokens"],
        cache_read_tokens=usage["cache_read_tokens"],
        reasoning_tokens=usage["reasoning_tokens"],
        usage_coverage=usage["usage_coverage"],
        **cost,
        evidence={
            "attempt_dir": rel(attempt),
            "events": rel(attempt / "bench" / "events.jsonl"),
            "usage": rel(attempt / "bench" / "usage.jsonl"),
            "effective_config": rel(attempt / "bench" / "effective_config.yaml"),
            "incumbent": rel(attempt / "bench" / "incumbent.json") if inc else None,
            "predictions": rel(
                attempt / "inference" / "out" / "predictions.normalized.csv"
            )
            if inference
            and (attempt / "inference/out/predictions.normalized.csv").exists()
            else None,
            "native_predictions": rel(attempt / "inference" / "out" / "predictions.csv")
            if inference and (attempt / "inference/out/predictions.csv").exists()
            else None,
            "search_log": rel(attempt / "search.log"),
        },
        protocol_checks={
            "llm_settings": llm_check,
            "search_input_isolated": "pass",  # asserted before launch
            "generated_code": col.get("code_audit"),
            "inference_fit_calls": inference and inference["fit_calls"],
            "inference_assets_modified": inference and inference["assets_modified"],
            "validation_problems": val["validation_problems"],
            "test_problems": inference and inference.get("problems"),
            "native_vs_trusted_validation": None
            if native is None or expected is None
            else {
                "native": native,
                "expected_native": expected,
                "discrepancy": native - expected,
            },
        },
        candidates={
            "nodes_executed": len(col["steps"]),
            "successful_nodes": sum(1 for s in col["steps"] if s.get("success")),
            "incumbent_commits": len(col["commits"]),
        },
        extra={
            "native_stop": col["native_stop"],
            "exit_code": sup and sup["exit_code"],
            "oom_killed": sup and sup.get("oom_killed"),
            "billable_output_tokens": usage["billable_output_tokens"],
            "resources": accounting.summarize_resources(col["resources"]),
            "inference_resources": inference and inference["inference_resources"],
            "selected_by": inc and inc.get("selected_by"),
            "trajectory": [
                [c["ts"] - start, c.get("native_validation_score")]
                for c in col["commits"]
            ]
            if start
            else [],
            "final": termination not in RETRYABLE,
            "finished_at": utcnow(),
        },
    )


def execute_attempt(
    resolved: dict[str, Any], item: dict[str, Any], attempt: Path
) -> dict[str, Any]:
    m = resolved["mlzero"]
    docker = m["executor"] == "docker"
    attempt_id = f"{item['run_id']}-{attempt.name}"
    host_events = attempt / "host_events.jsonl"
    attempt.mkdir(parents=True)
    (attempt / "bench").mkdir()

    inp = attempt / "input"
    inp.mkdir()
    for f in datasets.SEARCH_FILES:
        shutil.copy2(item["prepared"] / "search" / f, inp / f)
    (inp / mlzero.TASK_FILE).write_text(
        mlzero.task_text(item["entry"], item["manifest"])
    )
    datasets.assert_search_isolated(inp)
    import yaml

    (attempt / "mlzero.yaml").write_text(
        yaml.safe_dump(mlzero.render_config(resolved), sort_keys=True)
    )
    write_json(attempt / "identity.json", item["identity"])

    cache = Path(resolved["outputs"]["artifact_root"]).resolve() / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    name = f"ageml-mlz-{uuid.uuid4().hex[:10]}"
    argv, env = mlzero.search_command(
        resolved, attempt.resolve(), inp.resolve(), cache, name
    )
    env["OPENAI_API_KEY"] = os.environ.get(resolved["llm"]["api_key_env"], "")
    append_jsonl(host_events, {"event": "launch", "at": utcnow(), "argv": argv})
    sup = _supervise(
        argv,
        env,
        resolved["budget"]["search_wall_seconds"],
        attempt / "search.log",
        name if docker else None,
        resolved["budget"].get("stop_grace_seconds", 30),
    )
    append_jsonl(host_events, {"event": "search_end", "at": utcnow(), **sup})
    col = mlzero.collect(attempt, docker)
    termination = _termination(sup, col, docker)
    inference = None
    if col.get("incumbent") and col.get("export_present"):
        append_jsonl(host_events, {"event": "inference_start", "at": utcnow()})
        inference = _infer(resolved, item, attempt, col)
        append_jsonl(
            host_events,
            {"event": "inference_end", "at": utcnow(), "status": inference["status"]},
        )
    record = build_record(
        resolved, item, attempt, attempt_id, sup, col, termination, inference
    )
    write_json(attempt / "record.json", record)
    return record


def _close_interrupted(
    resolved: dict[str, Any], item: dict[str, Any], attempt: Path
) -> None:
    """An attempt the harness never finished: keep its partial accounting."""
    col = mlzero.collect(attempt, resolved["mlzero"]["executor"] == "docker")
    rec = build_record(
        resolved,
        item,
        attempt,
        f"{item['run_id']}-{attempt.name}",
        None,
        col,
        "harness_interrupted",
        None,
    )
    write_json(attempt / "record.json", rec)


def run_campaign(
    resolved: dict[str, Any], resume: bool = False, only: list[str] | None = None
) -> list[dict[str, Any]]:
    retries = resolved.get("retries", {}).get("max_infrastructure_retries", 1)
    records = []
    for item in plan(resolved):
        if only and item["entry"]["key"] not in only:
            continue
        if item["prepared"] is None:
            raise FileNotFoundError(
                f"{item['entry']['key']}: not prepared; run `prepare`"
            )
        d = run_dir(resolved, item)
        attempts = sorted(d.glob("attempt-*"), key=lambda p: int(p.name.split("-")[1]))
        if attempts and not resume:
            raise RuntimeError(f"{d} already has attempts; use --resume")
        for a in attempts:
            if not (a / "record.json").exists():
                _close_interrupted(resolved, item, a)
        recs = [read_json(a / "record.json") for a in attempts]
        if any(r["extra"]["final"] for r in recs):
            records.append(next(r for r in recs if r["extra"]["final"]))
            continue
        while len(recs) < 1 + retries:
            print(
                f"[ageml-bench] {item['entry']['key']} rep {item['repetition']} "
                f"attempt {len(recs) + 1}",
                file=sys.stderr,
            )
            rec = execute_attempt(resolved, item, d / f"attempt-{len(recs) + 1}")
            recs.append(rec)
            if rec["extra"]["final"]:
                break
        records.append(recs[-1])
    return records


def all_attempt_records(campaign: Path) -> list[dict[str, Any]]:
    return [
        read_json(p)
        for p in sorted(campaign.glob("runs/*/rep-*/*/attempt-*/record.json"))
    ]


def dry_run(resolved: dict[str, Any]) -> pd.DataFrame:
    items = plan(resolved)
    rows = [
        {
            "dataset": i["entry"]["key"],
            "repetition": i["repetition"],
            "run_id": i["run_id"],
            "prepared": i["prepared"] is not None,
            "done": any(
                read_json(r)["extra"]["final"]
                for r in run_dir(resolved, i).glob("attempt-*/record.json")
            ),
        }
        for i in items
    ]
    return pd.DataFrame(rows)
