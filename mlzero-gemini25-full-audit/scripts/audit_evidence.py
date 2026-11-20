"""Audit saved predictions, original targets, logs and events; no model execution."""

import argparse
import gzip
import hashlib
import io
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score


def existing(path):
    if path.is_file():
        return path
    compressed = path.with_name(path.name + ".gz")
    return compressed if compressed.is_file() else None


def contents(path):
    p = existing(path)
    if p is None:
        raise FileNotFoundError(path)
    return gzip.decompress(p.read_bytes()) if p.suffix == ".gz" else p.read_bytes()


def text(path):
    return contents(path).decode("utf-8", errors="replace")


def load(path):
    return json.loads(contents(path))


def jsonl(path):
    return [json.loads(line) for line in text(path).splitlines() if line.strip()]


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def category(message):
    if not message:
        return "No returned diagnostic"
    if any(
        s in message
        for s in (
            "No module named",
            "Bash script must create",
            "CondaError",
            "activation script not found",
        )
    ):
        return "Environment / dependencies"
    if any(
        s in message
        for s in (
            "cannot import name",
            "Unknown `.fit`",
            "Invalid kwargs",
            "unexpected keyword argument",
            "not found in the config",
            "Preset '",
            "Invalid problem_type",
            "has no attribute 'set_seed'",
            "Unknown model type",
        )
    ):
        return "Library API / imports"
    if "Declared scores disagree" in message:
        return "Score verification"
    if any(s in message for s in ("exited -11", "Segmentation fault", "Fatal Python")):
        return "Native crash"
    if "ray/_raylet.so" in message and "ray::" in message:
        return "Native runtime backtrace"
    return "Other returned error"


def error_excerpt(message):
    if "Error summary:" in message:
        summary = message.rsplit("Error summary:", 1)[1].strip()
    else:
        summary = message.strip()
    lines = [
        line.strip()
        for line in summary.splitlines()
        if re.match(r"^(?:[A-Za-z][\w.]*Error:|[A-Za-z]*Exception:)", line.strip())
    ]
    return (lines[-1] if lines else summary.splitlines()[0] if summary else "")[:1800]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifacts", type=Path, required=True)
    p.add_argument("--outdir", type=Path, required=True)
    a = p.parse_args()
    artifacts, out = a.artifacts.resolve(), a.outdir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    root = (
        artifacts / "campaigns/mlzero-gemini-25-flash-hamlet-cv-s42-1h-local-factory-v5"
    )
    cache, inputs, metrics, node_rows, run_rows = {}, [], [], [], []
    for record_path in sorted(root.glob("runs/*/rep-*/*/attempt-*/record.json")):
        record = load(record_path)
        if not record.get("final", True):
            continue
        ds, rep = record["dataset"], record["repetition"]
        attempt = record_path.parent
        identity = load(attempt / "identity.json")
        entry = next(e for e in identity["config"]["dataset_entries"] if e["key"] == ds)
        prepared = artifacts / "datasets-cv" / ds / Path(entry["prepared"]).name
        if ds not in cache:
            checks = {}
            for name, expected in entry["input_hashes"].items():
                checks[name] = digest(contents(prepared / name)) == expected
            contract = load(prepared / "cv_contract.json")
            raw = contents(prepared / "data.csv")
            target_frame = pd.read_csv(
                io.BytesIO(raw), usecols=[contract["target_column"]]
            )
            header = pd.read_csv(io.BytesIO(raw), nrows=0).columns.tolist()
            del raw
            y = target_frame[contract["target_column"]]
            assert all(checks.values()), (ds, checks)
            cache[ds] = contract, y
            inputs.append(
                dict(
                    dataset=ds,
                    rows=len(y),
                    features=len(header) - 1,
                    target_column=contract["target_column"],
                    original_data_hash_verified=checks["data.csv"],
                    original_contract_hash_verified=checks["cv_contract.json"],
                    evaluator_hash_verified=checks["evaluate_cv.py"],
                    readme_hash_verified=checks["README.md"],
                )
            )
        contract, y = cache[ds]
        classification = contract["task"] == "classification"
        selected_id = (record.get("selected") or {}).get("node_id")
        events = jsonl(attempt / "bench/events.jsonl")
        usage = jsonl(attempt / "bench/usage.jsonl")
        steps = {
            e["node_id"]: e
            for e in events
            if e.get("event") == "step" and e.get("node_id") is not None
        }
        stopped = next((e for e in events if e.get("event") == "search_stopped"), {})
        nodes = sorted(
            (attempt / "mlzero").glob("node_[0-9]*"), key=lambda n: int(n.name[5:])
        )
        last_id = int(nodes[-1].name[5:])
        start = record["finished_at"] - record["search_wall_s"]
        final_llm = max(usage, key=lambda e: e["ts"])
        last_response_offset = final_llm["ts"] - start
        metric_rows = []
        for node in nodes:
            node_id = int(node.name[5:])
            step = steps.get(node_id, {})
            prompt = existing(node / "states/error_analyzer_prompt.txt")
            message = ""
            if prompt:
                message = (
                    text(prompt)
                    .partition("### Error Message")[2]
                    .partition("### Task Description")[0]
                    .strip()
                )
            err_category = category(message)
            export = existing(node / "output/cv_evaluation.json")
            predictions = existing(node / "output/cv_predictions.csv")
            node_rows.append(
                dict(
                    dataset=ds,
                    repetition=rep,
                    node_id=node_id,
                    run_status=record["status"],
                    selected=node_id == selected_id,
                    step_returned=node_id in steps,
                    step_success=step.get("success"),
                    step_score=step.get("validation_score"),
                    node_status=(
                        "CV exported"
                        if export
                        else "Error returned"
                        if message
                        else "Unfinished at deadline"
                        if node_id == last_id
                        and record["termination_reason"] == "timeout"
                        else "No returned diagnostic"
                    ),
                    error_category=err_category,
                    error_excerpt=error_excerpt(message),
                    evidence_path=str(prompt.relative_to(artifacts)) if prompt else "",
                )
            )
            if not export:
                continue
            assert predictions, (ds, rep, node_id, "No OOF predictions")
            ev = load(export)
            pred = pd.read_csv(
                io.BytesIO(contents(predictions)),
                dtype={"prediction": str},
                keep_default_na=False,
            )
            assert not pred.row_id.duplicated().any() and set(pred.row_id) == set(
                range(len(y))
            )
            observed = pred.set_index("row_id")
            scores = []
            for fold, part in enumerate(contract["cv_folds"]):
                ids = part["validation"]
                assert set(pred.loc[pred.fold.eq(fold), "row_id"]) == set(ids)
                truth = y.iloc[ids]
                pp = observed.loc[ids, "prediction"]
                if classification:
                    assert set(pp) <= set(y.astype(str))
                    value = float(
                        balanced_accuracy_score(
                            np.asarray(truth, dtype=str), np.asarray(pp, dtype=str)
                        )
                    )
                else:
                    value = float(
                        np.sqrt(
                            np.mean(
                                (
                                    truth.to_numpy(dtype=np.float64)
                                    - pp.to_numpy(dtype=np.float64)
                                )
                                ** 2
                            )
                        )
                    )
                scores.append(value)
            value = float(np.mean(scores))
            rtol = 1e-10 if classification else 1e-6
            assert np.allclose(
                scores,
                ev["fold_scores"],
                rtol=rtol,
                atol=1e-10 if classification else 1e-8,
            )
            is_selected = node_id == selected_id
            record_error = abs(value - record["cv_score"]) if is_selected else None
            if is_selected:
                assert np.isclose(value, record["cv_score"], rtol=1e-12, atol=1e-10)
            rr = dict(
                dataset=ds,
                repetition=rep,
                node_id=node_id,
                selected=is_selected,
                metric="balanced_accuracy" if classification else "RMSE",
                recomputed_cv=value,
                evaluator_declared_cv=ev["cv_score"],
                difference_from_declared=value - ev["cv_score"],
                selected_record_difference=record_error,
                folds=json.dumps(scores),
                prediction_path=str(predictions.relative_to(artifacts)),
            )
            if classification:
                rr.update(
                    unique_prediction_labels=int(pred.prediction.nunique()),
                    single_class_folds=int(
                        pred.groupby("fold").prediction.nunique().eq(1).sum()
                    ),
                    overall_dominant_prediction_fraction=float(
                        pred.prediction.value_counts(normalize=True).iloc[0]
                    ),
                )
            metric_rows.append(rr)
            metrics.append(rr)
        run_mode = (
            "Valid CV export"
            if record["status"] == "completed"
            else (
                "Repair branches exhausted"
                if stopped.get("reason") == "all_nodes_terminal"
                else "Timeout during setup"
                if not existing(nodes[-1] / "output/model_factory.py")
                else "Timeout with evaluation unfinished"
            )
        )
        rr = dict(
            dataset=ds,
            repetition=rep,
            status=record["status"],
            termination_reason=record["termination_reason"],
            terminal_phase=run_mode,
            search_wall_minutes=record["search_wall_s"] / 60,
            generated_nodes=len(nodes),
            completed_node_steps=len(steps),
            cv_exports=len(metric_rows),
            last_node=last_id,
            last_llm_session=final_llm["session"],
            last_llm_response_offset_minutes=last_response_offset / 60,
            interval_after_last_llm_response_minutes=(
                record["finished_at"] - final_llm["ts"]
            )
            / 60,
            search_stopped_reason=stopped.get("reason"),
            log_path=str(existing(attempt / "search.log").relative_to(artifacts)),
            selected_node=selected_id,
            selected_score=record.get("cv_score"),
        )
        if metric_rows:
            best = (max if classification else min)(
                r["recomputed_cv"] for r in metric_rows
            )
            rr["best_saved_candidate_score"] = best
            rr["selected_matches_best_saved_candidate"] = np.isclose(
                record["cv_score"], best, rtol=1e-10, atol=1e-10
            )
        run_rows.append(rr)
    pd.DataFrame(inputs).to_csv(out / "verified_input_files.csv", index=False)
    pd.DataFrame(metrics).to_csv(out / "recomputed_candidate_metrics.csv", index=False)
    selected_metrics = pd.DataFrame([r for r in metrics if r["selected"]])
    selected_metrics = selected_metrics[
        [
            "dataset",
            "repetition",
            "node_id",
            "recomputed_cv",
            "selected_record_difference",
        ]
    ].rename(columns={"node_id": "recomputed_selected_node"})
    for name in ("winner_audit.csv", "regression_winners.csv", "export_checks.csv"):
        path = out / name
        if not path.exists():
            continue
        table = pd.read_csv(path).drop(
            columns=[
                "target_values_available",
                "independent_metric_recomputation",
                "recomputed_selected_node",
                "recomputed_cv",
                "selected_record_difference",
            ],
            errors="ignore",
        )
        table = table.merge(
            selected_metrics, on=["dataset", "repetition"], validate="one_to_one"
        )
        table["target_values_available"] = True
        table["independent_metric_recomputation"] = True
        table.to_csv(path, index=False)
    pd.DataFrame(node_rows).to_csv(out / "node_diagnostics.csv", index=False)
    pd.DataFrame(run_rows).to_csv(out / "run_diagnostics.csv", index=False)
    failed_rows = []
    for run in run_rows:
        if run["status"] == "completed":
            continue
        relevant = [
            n
            for n in node_rows
            if n["dataset"] == run["dataset"] and n["repetition"] == run["repetition"]
        ]
        failed_rows.append(
            {
                **run,
                "observed_candidate_categories": json.dumps(
                    dict(Counter(n["error_category"] for n in relevant))
                ),
                "observed_error_examples": " | ".join(
                    n["error_excerpt"] for n in relevant if n["error_excerpt"]
                ),
                "interpretation_limit": "Terminal phase and returned errors are recorded. Exact active import/fit/fold or waiting mechanism at timeout is not logged.",
            }
        )
    pd.DataFrame(failed_rows).to_csv(out / "failure_audit.csv", index=False)
    summary = dict(
        datasets_with_verified_original_inputs=len(inputs),
        independently_recomputed_candidate_exports=len(metrics),
        independently_recomputed_selected_winners=sum(r["selected"] for r in metrics),
        maximum_selected_record_recomputation_error=max(
            r["selected_record_difference"] for r in metrics if r["selected"]
        ),
        terminal_phases=dict(Counter(r["terminal_phase"] for r in run_rows)),
        failed_run_candidate_errors=dict(
            Counter(
                r["error_category"] for r in node_rows if r["run_status"] != "completed"
            )
        ),
        winner_selection_discrepancies=[
            r
            for r in run_rows
            if r.get("selected_matches_best_saved_candidate") is not None
            and not r["selected_matches_best_saved_candidate"]
        ],
        experiment_reruns=0,
    )
    (out / "full_verification_summary.json").write_text(
        json.dumps(summary, indent=2, default=lambda x: bool(x)) + "\n"
    )
    base_summary_path = out / "verification_summary.json"
    base_summary = load(base_summary_path) if base_summary_path.exists() else {}
    base_summary.update(summary)
    base_summary.update(
        raw_targets_in_archive=True, runtime_failure_logs_in_archive=True
    )
    base_summary_path.write_text(
        json.dumps(base_summary, indent=2, default=lambda x: bool(x)) + "\n"
    )
    print(json.dumps(summary, indent=2, default=lambda x: bool(x)))


if __name__ == "__main__":
    main()
