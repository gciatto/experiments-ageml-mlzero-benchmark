"""Campaign configuration: load, validate, resolve against the reference manifest."""

import copy
from pathlib import Path
from typing import Any

import yaml

from . import read_json

REQUIRED = (
    "campaign_id",
    "reference_manifest",
    "required_protocol_id",
    "system",
    "repetitions",
    "datasets",
    "mlzero",
    "llm",
    "resources",
    "budget",
    "evaluation",
    "outputs",
)


class ConfigError(ValueError):
    pass


def load(path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(path).read_text())
    missing = [k for k in REQUIRED if k not in cfg]
    if missing:
        raise ConfigError(f"{path}: missing keys {missing}")
    if cfg["system"] != "mlzero":
        raise ConfigError("only system: mlzero is implemented")
    if cfg["evaluation"].get("refit", False):
        raise ConfigError("evaluation.refit must be false (no train+validation refit)")
    cfg["_path"] = str(path)
    return cfg


def resolve(cfg: dict[str, Any], base: Path | None = None) -> dict[str, Any]:
    """Replace ``source: reference_manifest`` sections by the actual values.

    Raises ConfigError when a value the run depends on is unresolved.
    """
    base = base or Path(cfg["_path"]).parent.parent
    r = copy.deepcopy(cfg)
    ref = read_json(base / cfg["reference_manifest"])
    r["reference"] = {
        "path": cfg["reference_manifest"],
        "campaign_id": ref["reference_campaign_id"],
        "protocol": ref["protocol"],
        "source": ref["source"],
    }

    known = {d["key"]: d for d in ref["datasets"]}
    unknown = [k for k in cfg["datasets"] if k not in known]
    if unknown:
        raise ConfigError(f"datasets not in reference: {unknown}")
    r["dataset_entries"] = [known[k] for k in cfg["datasets"]]

    llm = r["llm"]
    if llm.get("source") == "reference_manifest":
        if llm.get("model"):
            raise ConfigError(
                "llm.model must be empty when llm.source is reference_manifest"
            )
        llm["model"] = ref["llm"]["requested_model"]
    if not llm.get("model") or not llm.get("endpoint"):
        raise ConfigError("llm.model and llm.endpoint must resolve")

    res = r["resources"]
    if res.get("source") == "reference_manifest":
        if not ref.get("resources"):
            raise ConfigError(
                "reference manifest has no measured resources; pass "
                "`reference --resources-file` or use explicit provisional resources"
            )
        r["resources"] = {**ref["resources"], "source": "reference_manifest"}
    for k in ("cpus", "memory_gib"):
        if r["resources"].get(k) is None:
            raise ConfigError(f"resources.{k} unresolved")
    return r


def is_corrected(resolved: dict[str, Any]) -> bool:
    return (
        resolved["reference"]["protocol"].get("protocol_id")
        == resolved["required_protocol_id"]
    )
