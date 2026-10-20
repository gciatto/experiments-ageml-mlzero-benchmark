import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from ageml_mlzero_benchmark import write_json
from ageml_mlzero_benchmark import config as cfgmod
from ageml_mlzero_benchmark.datasets import prepare_dataset

FIXTURES = Path(__file__).parent / "fixtures"
ENDPOINT = "https://example.invalid/v1beta/openai/"


def synthetic(task: str, n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.integers(0, 5, n).astype(float),
            "color": rng.choice(["r", "g", None], n),
        }
    )
    df.loc[::17, "x1"] = np.nan
    df["y" if task == "regression" else "class"] = (
        df.x2 * 3.5 + 100
        if task == "regression"
        else rng.choice(["neg", "pos"], n, p=[0.7, 0.3])
    )
    return df


def entry(key: str, task: str) -> dict:
    return {
        "key": key,
        "openml_id": 1,
        "task": task,
        "objective": "rmse" if task == "regression" else "balanced_accuracy",
        "target": "y" if task == "regression" else "class",
        "split_seed": 42,
        "split_hash": None,
        "dataset_version": 1,
    }


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A project root with a reference manifest, prepared data and a local-executor config."""
    entries = [entry("toyclf", "classification"), entry("toyreg", "regression")]
    write_json(
        tmp_path / "ref.json",
        {
            "reference_campaign_id": "ref-test",
            "source": {"commit": None},
            "protocol": {
                "protocol_id": "ageml-holdout-v1-test-selected",
                "comparability_status": "historical_protocol",
            },
            "llm": {"requested_model": "gemini-test"},
            "resources": None,
            "datasets": entries,
        },
    )
    for e in entries:
        csv = tmp_path / "exports" / f"{e['key']}.csv"
        csv.parent.mkdir(exist_ok=True)
        synthetic(e["task"]).to_csv(csv, index=False)
        prepare_dataset(e, tmp_path / "artifacts", export_csv=csv)
    cfg = {
        "schema_version": 1,
        "campaign_id": "test",
        "reference_manifest": "ref.json",
        "required_protocol_id": "ageml-holdout-v2",
        "system": "mlzero",
        "repetitions": [1],
        "datasets": ["toyclf", "toyreg"],
        "mlzero": {
            "commit": "f228cd7",
            "executor": "local",
            "continuous_improvement": True,
            "max_iterations": 3,
            "command": [sys.executable, str(FIXTURES / "fake_mlzero.py")],
            "python": sys.executable,
        },
        "llm": {
            "source": "explicit",
            "model": "gemini-test",
            "endpoint": ENDPOINT,
            "api_key_env": "FAKE_KEY",
            "settings": {},
        },
        "resources": {"source": "explicit", "cpus": 2, "memory_gib": 4},
        "budget": {"search_wall_seconds": 60, "stop_grace_seconds": 1},
        "evaluation": {"refit": False, "inference_timeout_seconds": 60},
        "outputs": {"artifact_root": str(tmp_path / "artifacts")},
    }
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "c.yml").write_text(yaml.safe_dump(cfg))
    return tmp_path


@pytest.fixture
def resolved(project: Path) -> dict:
    return cfgmod.resolve(cfgmod.load(project / "configs" / "c.yml"))
