"""AGE-ML reference capture and comparison eligibility.

The AGE-ML checkout is read as data (AST, YAML, git); its modules are never
imported because importing them can initialize LLM clients.
"""

import ast
import subprocess
import tomllib
from pathlib import Path
from typing import Any

import yaml

from . import SCHEMA_VERSION, read_json, sha256_file, utcnow

AGEML_REPO = "https://github.com/Mala1180/age-ml"
OBJECTIVE_BY_TASK = {"classification": "balanced_accuracy", "regression": "rmse"}
V1_PROTOCOL = "ageml-holdout-v1-test-selected"


def _literal(path: Path, name: str) -> Any:
    """Value of a module-level ``name = <literal>`` (annotated or not)."""
    for node in ast.parse(path.read_text()).body:
        targets: list[ast.expr]
        if isinstance(node, ast.AnnAssign):
            targets, value = [node.target], node.value
        elif isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        else:
            continue
        if value is not None and any(
            isinstance(t, ast.Name) and t.id == name for t in targets
        ):
            return ast.literal_eval(value)
    raise KeyError(f"{name} not found in {path}")


def _function_source(path: Path, name: str) -> str:
    src = path.read_text()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(src, node) or ""
    return ""


def _git(source: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(source), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def detect_protocol(source: Path) -> dict[str, Any]:
    """Classify the selection protocol implemented by the checkout."""
    selector = _function_source(
        source / "ageml/evaluation/agent.py", "get_best_run_across_pipelines"
    )
    if "test_df" in selector:
        return {
            "protocol_id": V1_PROTOCOL,
            "comparability_status": "historical_protocol",
            "evidence": "ageml/evaluation/agent.py::get_best_run_across_pipelines "
            "scores pipeline finalists on test_df to pick the global winner",
        }
    return {
        "protocol_id": None,
        "comparability_status": "unverified_protocol",
        "evidence": "test-based global selection not found; supply the AGE-ML "
        "protocol manifest (--protocol-manifest) to establish the protocol",
    }


def openml_targets(ids: list[int]) -> dict[int, dict[str, Any]]:
    import openml

    out = {}
    for i in ids:
        ds = openml.datasets.get_dataset(
            i,
            download_data=False,
            download_qualities=False,
            download_features_meta_data=False,
        )
        out[i] = {
            "name": ds.name,
            "dataset_version": ds.version,
            "default_target_attribute": ds.default_target_attribute,
        }
    return out


def build_reference(
    age_source: Path,
    campaign_id: str,
    protocol_manifest: Path | None = None,
    resources_file: Path | None = None,
    fetch_openml: bool = True,
) -> dict[str, Any]:
    age_source = age_source.resolve()
    spec_path = age_source / "resources/general-specification.yml"
    spec = yaml.safe_load(spec_path.read_text())
    by_kind = _literal(
        age_source / "experiments/download_datasets.py", "DEFAULT_OPENML_DATASETS"
    )
    model = _literal(age_source / "ageml/common/__init__.py", "DEFAULT_MODEL_NAME")
    try:
        repetitions = _literal(
            age_source / "experiments/ablation/sweep.py", "DEFAULT_REPETITIONS"
        )
    except (KeyError, FileNotFoundError):
        repetitions = None
    release = tomllib.loads((age_source / "pyproject.toml").read_text())
    release = release.get("project", release.get("tool", {}).get("poetry", {})).get(
        "version"
    )

    ids = [i for kind in by_kind.values() for i in kind.values()]
    meta = openml_targets(ids) if fetch_openml else {}
    datasets = []
    for task, entries in by_kind.items():
        for key, openml_id in entries.items():
            m = meta.get(openml_id, {})
            datasets.append(
                {
                    "key": key,
                    "openml_id": openml_id,
                    "task": task,
                    "objective": OBJECTIVE_BY_TASK[task],
                    "openml_name": m.get("name"),
                    "dataset_version": m.get("dataset_version"),
                    # AGE-ML v1 asks an LLM to identify the target; the benchmark
                    # uses OpenML's declared target and checks the two agree.
                    "target": m.get("default_target_attribute"),
                    "target_source": "openml_default_target_attribute" if m else None,
                    "target_units": None,
                    "split_seed": 42,
                    "split_hash": None,
                    "export_sha256": None,
                }
            )

    budgets = spec.get("budgets", {})
    t = budgets.get("time", {})
    nominal = t.get("hours", 0) * 3600 + t.get("minutes", 0) * 60 + t.get("seconds", 0)
    ref: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "reference_campaign_id": campaign_id,
        "captured_at": utcnow(),
        "source": {
            "repository": AGEML_REPO,
            "commit": _git(age_source, "rev-parse", "HEAD"),
            "dirty": bool(_git(age_source, "status", "--porcelain")),
            "release": release,
        },
        "protocol": detect_protocol(age_source),
        "specification": {
            "path": "resources/general-specification.yml",
            "sha256": sha256_file(spec_path),
            "budgets": budgets,
            "technical_details": spec.get("technical_details"),
        },
        "budget": {
            "nominal_search_seconds": nominal or None,
            "pipelines": budgets.get("pipelines"),
            "workers": budgets.get("workers"),
            "generation_attempts": budgets.get("generation_attempts"),
            # the inspected deadline wraps pipeline jobs only (ageml/app.py);
            # final test evaluation and reporting run after it
            "timing_boundary": "pipeline jobs only (inspected v1); unverified for this campaign",
        },
        "llm": {
            "provider": "google",
            "client": "langchain_google_genai.ChatGoogleGenerativeAI(model=...) with provider defaults",
            "requested_model": model,
            "settings": None,
        },
        "resources": None,
        "repetitions": {
            "count": repetitions,
            "split_seed": 42,
            "search_seed": None,
            "note": "repetitions re-run the search on one fixed split",
        },
        "datasets": datasets,
    }
    if resources_file is not None:
        ref["resources"] = yaml.safe_load(resources_file.read_text())
    if protocol_manifest is not None:
        apply_protocol_manifest(ref, read_json(protocol_manifest))
    return ref


def apply_protocol_manifest(ref: dict[str, Any], manifest: dict[str, Any]) -> None:
    """Overlay an AGE-ML-exported protocol manifest.

    The ``ageml-holdout-v2`` schema is not yet defined upstream; the keys read
    here are the proposed ones. Unknown keys are kept verbatim for audit.
    """
    ref["protocol"] = {
        "protocol_id": manifest.get("protocol_id"),
        "comparability_status": "corrected_protocol"
        if manifest.get("protocol_id") == "ageml-holdout-v2"
        else "unverified_protocol",
        "evidence": "AGE-ML protocol manifest",
        "manifest": manifest,
    }
    by_key = {d["key"]: d for d in ref["datasets"]}
    for d in manifest.get("datasets", []):
        if d.get("key") in by_key:
            by_key[d["key"]].update(
                {
                    k: d[k]
                    for k in (
                        "target",
                        "target_units",
                        "split_hash",
                        "export_sha256",
                        "split_seed",
                    )
                    if k in d
                }
            )
            by_key[d["key"]]["target_source"] = "ageml_protocol_manifest"
    for k in ("llm", "resources"):
        if manifest.get(k):
            ref[k] = manifest[k]


# --- comparison eligibility ----------------------------------------------------

PREDICTIVE_CHECKS = ("data", "split", "protocol", "target", "objective", "model")
RESOURCE_CHECKS = ("resources", "timing")


def _cmp(a: Any, b: Any) -> str:
    if a is None or b is None:
        return "unknown"
    return "pass" if a == b else "fail"


def pair_checks(
    age: dict[str, Any], mlz: dict[str, Any], required_protocol: str
) -> dict[str, Any]:
    """Compatibility of one AGE-ML record with one MLZero record (same dataset)."""
    checks = {
        "data": _cmp(age.get("dataset_hash"), mlz.get("dataset_hash")),
        "split": _cmp(age.get("split_hash"), mlz.get("split_hash")),
        "protocol": "pass"
        if age.get("protocol_id") == mlz.get("protocol_id") == required_protocol
        else ("unknown" if age.get("protocol_id") is None else "fail"),
        "target": _cmp(age.get("target"), mlz.get("target")),
        "objective": _cmp(age.get("objective"), mlz.get("objective")),
        "model": _cmp(age.get("reported_model") or None, mlz.get("requested_model")),
        "resources": _cmp(age.get("resource_limits"), mlz.get("resource_limits")),
        "timing": _cmp(age.get("search_budget_s"), mlz.get("search_budget_s"))
        if age.get("timing_boundary_verified")
        else "unknown",
    }
    predictive = all(checks[k] == "pass" for k in PREDICTIVE_CHECKS)
    return {
        "checks": checks,
        "predictive_eligible": predictive,
        "resource_eligible": predictive
        and all(checks[k] == "pass" for k in RESOURCE_CHECKS),
    }
