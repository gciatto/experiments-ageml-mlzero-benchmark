"""MLZero baseline experiments for the AGE-ML comparison.

Shared helpers live here: hashing, atomic JSON writes, timestamps.
"""

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

#: Objective name -> whether higher is better.
OBJECTIVES: dict[str, bool] = {"balanced_accuracy": True, "rmse": False}


#: Canonical run record. Unavailable values are None, never 0.
RECORD_FIELDS: tuple[str, ...] = (
    # identity
    "schema_version",
    "campaign_id",
    "run_id",
    "attempt_id",
    "system",
    "dataset_key",
    "repetition_id",
    # source
    "source_commit",
    "patch_hash",
    "environment_digest",
    "config_hash",
    "reference_campaign_id",
    # data / protocol
    "openml_id",
    "dataset_version",
    "dataset_hash",
    "target",
    "task",
    "split_hash",
    "protocol_id",
    "split_seed",
    "search_seed",
    # llm
    "provider",
    "endpoint",
    "requested_model",
    "reported_model",
    "agent_settings",
    # outcome
    "artifact_status",
    "termination_reason",
    "selected_candidate_id",
    "selected_artifact_hash",
    "comparability_status",
    # metrics
    "objective",
    "maximize",
    "native_validation_score",
    "validation_score",
    "test_score",
    # time / resources
    "search_wall_s",
    "search_budget_s",
    "first_valid_artifact_s",
    "inference_s",
    "cpu_s",
    "peak_rss_bytes",
    "resource_limits",
    "timing_boundary_verified",
    # usage
    "llm_calls",
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "reasoning_tokens",
    "cost",
    "currency",
    "pricing_version",
    "usage_coverage",
    # evidence
    "evidence",
    "protocol_checks",
    "candidates",
    "extra",
)


def new_record(**values: Any) -> dict[str, Any]:
    unknown = set(values) - set(RECORD_FIELDS)
    if unknown:
        raise KeyError(f"unknown record fields: {sorted(unknown)}")
    rec = dict.fromkeys(RECORD_FIELDS)
    rec["schema_version"] = SCHEMA_VERSION
    rec.update(values)
    return rec


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_tree(root: Path) -> dict[str, str]:
    """Hash every regular file under ``root``, keyed by POSIX relative path."""
    return {
        p.relative_to(root).as_posix(): sha256_file(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and not p.is_symlink()
    }


def canonical_hash(obj: Any) -> str:
    """Hash of the canonical JSON form of ``obj`` (sorted keys, no whitespace)."""
    return sha256_bytes(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()
    )


def write_json(path: Path, obj: Any) -> None:
    """Write JSON atomically: a crash leaves either the old or the new file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=str)
        f.write("\n")
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def append_jsonl(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(obj, sort_keys=True, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path: Path) -> list[Any]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            # a hard kill can truncate the last line; everything before it is kept
            continue
    return out
