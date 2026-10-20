"""Exact dataset preparation and immutable partition manifests.

Layout under ``<artifact_root>/datasets/<key>/<split_hash[:12]>/``::

    manifest.json        dataset + split provenance (no data)
    split.json           ordered row ids per partition
    search/              the ONLY directory mounted into MLZero search
        train.csv            labeled fitting rows
        val.csv              labeled validation rows
        val_features.csv     validation features, for native results.csv
    private/             trusted-evaluator side, never mounted into search
        export.csv           the exported all-row CSV both systems read
        test_features.csv
        test_labels.csv      row_id,target
        val_labels.csv       row_id,target

Row ids are positions in ``export.csv``; they are kept as sidecars and never
written as feature columns.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

from . import SCHEMA_VERSION, canonical_hash, read_json, sha256_file, utcnow, write_json

SEARCH_FILES = {"train.csv", "val.csv", "val_features.csv"}

#: Two-stage split of the inspected AGE-ML code: app.py (outer 80/20) then
#: execution/agent.py (inner 80/20 of development), each stratified with an
#: unstratified fallback on ValueError, random_state=seed both times.
AGEML_V1_SPLIT = "ageml-v1-two-stage-train_test_split"


def export_openml(openml_id: int, dest: Path, cache_dir: Path) -> dict[str, Any]:
    """Download one OpenML dataset and export it exactly as AGE-ML's downloader does.

    AGE-ML calls ``get_data(dataset_format="dataframe")`` without a target, so
    the CSV holds every column in OpenML order, written with ``index=False``.
    """
    import openml  # heavy, and only needed here

    openml.config.set_root_cache_directory(str(cache_dir))
    ds = openml.datasets.get_dataset(
        openml_id, download_data=True, download_qualities=False
    )
    X, y, _, _ = ds.get_data(dataset_format="dataframe")
    df = pd.concat([X, y], axis=1)  # y is None here, as in AGE-ML
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    raw_file = getattr(ds, "parquet_file", None) or getattr(ds, "data_file", None)
    return {
        "openml_id": openml_id,
        "name": ds.name,
        "dataset_version": ds.version,
        "default_target_attribute": ds.default_target_attribute,
        "openml_md5_checksum": ds.md5_checksum,
        "raw_file": Path(raw_file).name if raw_file else None,
        "raw_sha256": sha256_file(Path(raw_file))
        if raw_file and Path(raw_file).exists()
        else None,
        "openml_package_version": openml.__version__,
    }


def schema_of(df: pd.DataFrame, target: str) -> dict[str, Any]:
    missing = df.isna().sum()
    return {
        "features": [
            {"name": c, "dtype": str(df[c].dtype), "missing": int(missing[c])}
            for c in df.columns
            if c != target
        ],
        "target": {"name": target, "dtype": str(df[target].dtype)},
        # pandas writes NaN as an empty field and reads empty fields back as NaN
        "missing_value_representation": "empty CSV field",
        "rows": int(len(df)),
    }


def _split_once(
    ids: np.ndarray, y: np.ndarray | None, test_size: float, seed: int
) -> tuple[np.ndarray, np.ndarray, bool]:
    try:
        a, b = train_test_split(ids, test_size=test_size, stratify=y, random_state=seed)
        return a, b, y is not None
    except ValueError:
        a, b = train_test_split(ids, test_size=test_size, random_state=seed)
        return a, b, False


def make_split(
    df: pd.DataFrame, target: str, task: str, seed: int = 42
) -> dict[str, Any]:
    """Regenerate the AGE-ML v1 partition (~64/16/20) over export row ids."""
    ids = np.arange(len(df))
    # raw values, as AGE-ML passes df[target]: class order affects the allocation
    y = df[target].to_numpy() if task == "classification" else None
    dev, test, strat_outer = _split_once(ids, y, 0.2, seed)
    y_dev = y[dev] if y is not None else None
    train, val, strat_inner = _split_once(dev, y_dev, 0.2, seed)
    return {
        "algorithm": AGEML_V1_SPLIT,
        "seed": seed,
        "stratified": {"outer": strat_outer, "inner": strat_inner},
        "sklearn_version": sklearn.__version__,
        "train": [int(i) for i in train],
        "validation": [int(i) for i in val],
        "test": [int(i) for i in test],
    }


def split_hash(split: dict[str, Any]) -> str:
    return canonical_hash({k: split[k] for k in ("train", "validation", "test")})


def validate_split(
    split: dict[str, Any], df: pd.DataFrame, target: str, task: str
) -> list[str]:
    problems = []
    parts = {k: set(split[k]) for k in ("train", "validation", "test")}
    for k in parts:
        if len(parts[k]) != len(split[k]):
            problems.append(f"{k} has duplicated row ids")
    if (
        parts["train"] & parts["validation"]
        or parts["train"] & parts["test"]
        or (parts["validation"] & parts["test"])
    ):
        problems.append("partitions overlap")
    if set().union(*parts.values()) != set(range(len(df))):
        problems.append("partitions do not cover every row exactly once")
    if df[target].isna().any():
        problems.append(f"{int(df[target].isna().sum())} rows with missing target")
    if task == "classification":
        train_classes = set(df[target].iloc[split["train"]].astype(str))
        for k in ("validation", "test"):
            unseen = set(df[target].iloc[split[k]].astype(str)) - train_classes
            if unseen:
                problems.append(f"{k} has classes absent from train: {sorted(unseen)}")
    return problems


def write_partitions(
    df: pd.DataFrame, target: str, split: dict[str, Any], out: Path
) -> None:
    search, private = out / "search", out / "private"
    search.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    tr, va, te = (df.iloc[split[k]] for k in ("train", "validation", "test"))
    tr.to_csv(search / "train.csv", index=False)
    va.to_csv(search / "val.csv", index=False)
    va.drop(columns=[target]).to_csv(search / "val_features.csv", index=False)
    te.drop(columns=[target]).to_csv(private / "test_features.csv", index=False)
    for name, part, ids in (
        ("test_labels.csv", te, split["test"]),
        ("val_labels.csv", va, split["validation"]),
    ):
        pd.DataFrame({"row_id": ids, "target": part[target].to_numpy()}).to_csv(
            private / name, index=False
        )


def load_labels(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"target": str}, keep_default_na=False)


def prepare_dataset(
    entry: dict[str, Any],
    artifact_root: Path,
    split_manifest: Path | None = None,
    export_csv: Path | None = None,
) -> Path:
    """Prepare one reference dataset; returns the prepared directory.

    ``entry`` is a reference-manifest dataset (key, openml_id, task, objective,
    target). ``split_manifest`` is an exported AGE-ML split (preferred); if
    absent the AGE-ML v1 split is regenerated and, when the reference records a
    split hash, it must match. ``export_csv`` skips the download (tests, or
    reusing AGE-ML's own exported bytes).
    """
    key, target, task = entry["key"], entry["target"], entry["task"]
    if not target:
        raise ValueError(f"{key}: no explicit target in the reference manifest")
    staging = artifact_root / "datasets" / key
    if export_csv is None:
        export_csv = staging / "export.csv"
        source = export_openml(
            entry["openml_id"], export_csv, artifact_root / "openml-cache"
        )
    else:
        source = {
            "openml_id": entry.get("openml_id"),
            "export_supplied": str(export_csv),
        }
    if source.get("default_target_attribute") not in (None, target):
        source["target_note"] = (
            f"reference target {target!r} differs from OpenML default "
            f"{source['default_target_attribute']!r}"
        )
    df = pd.read_csv(export_csv)  # exactly how AGE-ML reads it
    if target not in df.columns:
        raise ValueError(f"{key}: target {target!r} not in exported columns")

    if split_manifest is not None:
        split = read_json(split_manifest)
        split.setdefault("algorithm", "imported")
        split["source"] = str(split_manifest)
    else:
        split = make_split(df, target, task, entry.get("split_seed", 42))
    problems = validate_split(split, df, target, task)
    if problems:
        raise ValueError(f"{key}: invalid split: {problems}")
    h = split_hash(split)
    if entry.get("split_hash") and entry["split_hash"] != h:
        raise ValueError(f"{key}: split hash {h} != reference {entry['split_hash']}")

    out = staging / h[:12]
    write_json(out / "split.json", split)
    write_partitions(df, target, split, out)
    (out / "private").mkdir(parents=True, exist_ok=True)
    export_dest = out / "private" / "export.csv"
    if export_csv.resolve() != export_dest.resolve():
        export_dest.write_bytes(export_csv.read_bytes())
    classes = (
        sorted(df[target].dropna().astype(str).unique())
        if task == "classification"
        else None
    )
    y = pd.to_numeric(df[target], errors="coerce") if task == "regression" else None
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "prepared_at": utcnow(),
        "dataset_key": key,
        "task": task,
        "objective": entry["objective"],
        "target": target,
        "target_units": entry.get("target_units"),
        "target_summary": None
        if y is None
        else {"min": float(y.min()), "max": float(y.max()), "mean": float(y.mean())},
        "class_mapping": None
        if classes is None
        else {c: i for i, c in enumerate(classes)},
        "source": source,
        "export_sha256": sha256_file(export_dest),
        "schema": schema_of(df, target),
        "split_hash": h,
        "split_algorithm": split["algorithm"],
        "split_seed": split.get("seed"),
        "counts": {k: len(split[k]) for k in ("train", "validation", "test")},
        "files": {
            p.relative_to(out).as_posix(): sha256_file(p)
            for p in sorted(out.rglob("*.csv"))
        },
    }
    write_json(out / "manifest.json", manifest)
    return out


def find_prepared(artifact_root: Path, key: str) -> Path:
    """The single prepared directory for ``key``; ambiguity is an error."""
    found = sorted((artifact_root / "datasets" / key).glob("*/manifest.json"))
    if len(found) != 1:
        raise FileNotFoundError(
            f"{key}: expected one prepared split, found {len(found)}; run `prepare`"
        )
    return found[0].parent


def assert_search_isolated(search_dir: Path) -> None:
    """Only the allowed fitting/validation files may be visible to search."""
    extra = {p.name for p in search_dir.iterdir()} - SEARCH_FILES - {"README.md"}
    if extra:
        raise RuntimeError(f"unexpected files in search input: {sorted(extra)}")
