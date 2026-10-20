"""AGE-ML result and provenance import.

Reads AGE-ML ``results.csv`` summaries (any subset of the inspected
``experiments/results_csv.py::SUMMARY_COLUMNS``) plus an optional sibling
``manifest.json`` (the proposed ``ageml-holdout-v2`` export). Without a
manifest the rows are historical: no commit, data hash, split hash or protocol
is assumed, whatever the directory names suggest.
"""

import csv
import re
from pathlib import Path
from typing import Any

from . import OBJECTIVES, canonical_hash, new_record, read_json, sha256_file
from .reference import V1_PROTOCOL


def parse_duration(text: str | None) -> float | None:
    """``H:MM:SS`` or ``N day(s), H:MM:SS`` (``str(timedelta)``) -> seconds."""
    if not text:
        return None
    m = re.fullmatch(r"(?:(\d+) days?, )?(\d+):(\d{2}):(\d{2}(?:\.\d+)?)", text.strip())
    if not m:
        return None
    d, h, mi, s = m.groups()
    return int(d or 0) * 86400 + int(h) * 3600 + int(mi) * 60 + float(s)


def _num(text: str | None) -> float | None:
    try:
        return float(text) if text else None
    except ValueError:
        return None


def _int(text: str | None) -> int | None:
    v = _num(text)
    return None if v is None else int(v)


def import_results_file(
    path: Path, campaign_id: str, reference: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    path = Path(path)
    manifest_path = path.parent / "manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    rep = re.fullmatch(r"run-(\d+)", path.parent.name)
    repetition = manifest.get("repetition_id") or (int(rep.group(1)) if rep else None)
    # experiments/results/<model>/<axis>/<config>/run-N/results.csv
    parts = path.parts
    model_hint = parts[parts.index("results") + 1] if "results" in parts[:-1] else None
    file_hash = sha256_file(path)
    ref_datasets = {d["key"]: d for d in (reference or {}).get("datasets", [])}
    per_dataset = {d["key"]: d for d in manifest.get("datasets", [])}
    protocol_id = manifest.get("protocol_id")
    if protocol_id is None:
        status = "historical_protocol"
    elif protocol_id == V1_PROTOCOL:
        status = "historical_protocol"
    else:
        status = (
            "corrected_protocol"
            if protocol_id == "ageml-holdout-v2"
            else "unverified_protocol"
        )

    records = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = row.get("Dataset")
            ds = per_dataset.get(key, {})
            objective = row.get("Validation Metric") or ref_datasets.get(key, {}).get(
                "objective"
            )
            test = _num(row.get("Best Test-set Score"))
            in_tok, out_tok = (
                _int(row.get("Input Tokens")),
                _int(row.get("Output Tokens")),
            )
            records.append(
                new_record(
                    campaign_id=campaign_id,
                    run_id=canonical_hash(["ageml", file_hash, key, repetition])[:16],
                    attempt_id=None,
                    system="ageml",
                    dataset_key=key,
                    repetition_id=repetition,
                    source_commit=manifest.get("source_commit"),
                    environment_digest=manifest.get("environment_digest"),
                    config_hash=manifest.get("specification_sha256"),
                    reference_campaign_id=manifest.get("campaign_id"),
                    openml_id=_int(row.get("OpenML ID")),
                    dataset_hash=ds.get("export_sha256"),
                    target=ds.get("target"),
                    task=row.get("Problem"),
                    split_hash=ds.get("split_hash"),
                    protocol_id=protocol_id,
                    split_seed=ds.get("split_seed", manifest.get("split_seed")),
                    search_seed=None,
                    provider=(manifest.get("llm") or {}).get("provider"),
                    requested_model=(manifest.get("llm") or {}).get("requested_model"),
                    reported_model=(manifest.get("llm") or {}).get("reported_model"),
                    artifact_status="valid" if test is not None else "no_artifact",
                    termination_reason=ds.get("termination_reason"),
                    selected_candidate_id=row.get("Best Pipeline") or None,
                    comparability_status=status,
                    objective=objective,
                    maximize=OBJECTIVES.get(objective or ""),
                    validation_score=_num(ds.get("validation_score")),
                    test_score=test,
                    search_wall_s=parse_duration(row.get("Actual Time")),
                    search_budget_s=manifest.get("search_budget_s"),
                    resource_limits=manifest.get("resource_limits"),
                    timing_boundary_verified=manifest.get(
                        "timing_boundary_verified", False
                    ),
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    cost=_num(row.get("Putative Cost ($)")),
                    currency="USD" if row.get("Putative Cost ($)") else None,
                    usage_coverage="ageml_source_instrumentation",
                    evidence={
                        "source_file": str(path),
                        "source_sha256": file_hash,
                        "manifest": str(manifest_path) if manifest else None,
                    },
                    candidates={
                        "pipeline_budget": _int(row.get("Pipeline Budget")),
                        # counts non-null MLflow run ids (results_csv.py)
                        "actual_pipelines": _int(row.get("Actual Pipelines")),
                        "pipelines_found": _int(row.get("Pipelines Found")),
                        "semantic_validation_failures": _int(
                            row.get("Semantic Validation Failures")
                        ),
                    },
                    extra={
                        "model_from_path": model_hint,
                        "workers": _int(row.get("Workers")),
                        # sum of per-pipeline durations, not wall or CPU time
                        "equivalent_time_s": parse_duration(row.get("Equivalent Time")),
                        "llm_inference_time_s": parse_duration(
                            row.get("LLM Inference Time")
                        ),
                        "ml_training_time_s": parse_duration(
                            row.get("ML Training Time")
                        ),
                        "features": _int(row.get("Features")),
                        "instances": _int(row.get("Instances")),
                        "numeric_features": _int(row.get("Numeric Features")),
                        "discrete_features": _int(row.get("Discrete Features")),
                        "specification": row.get("Specification") or None,
                        "test_selected_winner": protocol_id in (None, V1_PROTOCOL),
                    },
                )
            )
    return records


def import_results(
    root: Path, campaign_id: str, reference: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    root = Path(root)
    files = [root] if root.is_file() else sorted(root.rglob("results.csv"))
    return [r for f in files for r in import_results_file(f, campaign_id, reference)]
