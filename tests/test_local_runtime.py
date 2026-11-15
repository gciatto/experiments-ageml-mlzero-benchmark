"""Local executor boundaries; no Docker, conda download or Gemini call required."""

from pathlib import Path

import pytest

from ageml_mlzero_benchmark import local_runtime as local
from ageml_mlzero_benchmark import mlzero, write_json


def test_runtime_drift_blocks_launch(tmp_path, monkeypatch):
    manifest = tmp_path / "runtime.json"
    write_json(manifest, {"prefix": "/local", "freeze": ["original"]})
    monkeypatch.setattr(
        local, "snapshot", lambda p: {"prefix": p, "freeze": ["changed"]}
    )
    with pytest.raises(ValueError, match="runtime changed"):
        local.verify(manifest)


def test_configure_local_preserves_docker_inputs(tmp_path, monkeypatch):
    prepared = tmp_path / "input"
    prepared.mkdir()
    original = "python /bench/input/evaluate_cv.py"
    (prepared / "README.md").write_text(original)
    (prepared / "data.csv").write_text("x,y\n1,2\n")
    runtime = {"prefix": "/local", "python": "/local/env/bin/python"}
    monkeypatch.setattr(local, "verify", lambda p: runtime)
    cfg = {
        "campaign_id": "example",
        "mlzero": {},
        "resources": {"memory_gib": 50},
        "dataset_entries": [
            {
                "prepared": str(prepared),
                "input_hashes": {"README.md": "unused", "data.csv": "unused"},
            }
        ],
    }
    result = local.configure(cfg, tmp_path / "runtime.json")
    assert result["campaign_id"] == "example-local"
    assert (prepared / "README.md").read_text() == original
    copied = Path(result["dataset_entries"][0]["prepared"])
    assert str(copied / "evaluate_cv.py") in (copied / "README.md").read_text()
    assert (copied / "data.csv").read_bytes() == (prepared / "data.csv").read_bytes()
    assert result["mlzero"]["executor"] == "local"


def test_local_command_has_limits_seed_cache_and_no_docker(tmp_path):
    cfg = {
        "mlzero": {
            "executor": "local",
            "python": "/local/env/bin/python",
            "command": ["/local/env/bin/mlzero"],
            "runtime_manifest": "/local/runtime.json",
            "conda_bin": "/local/conda/bin",
            "max_iterations": 10,
            "continuous_improvement": True,
            "search_seed": 42,
        },
        "resources": {"memory_gib": 50, "cpus": 16},
        "llm": {"endpoint": "google"},
    }
    argv, env = mlzero.search_command(
        cfg, tmp_path / "run", tmp_path / "input", tmp_path / "cache", "mlzero-cv-test"
    )
    assert argv[0] == "systemd-run"
    assert not any(x == "docker" for x in argv)
    assert "MemoryHigh=40G" in argv and "MemoryMax=50G" in argv
    assert "CPUQuota=1600%" in argv and "MemorySwapMax=0" in argv
    assert "RuntimeMaxSec=3630s" in argv and "3600s" in argv
    assert env["PYTHONHASHSEED"] == "42"
    assert Path(env["PYTHONPATH"]).is_dir()
    assert env["UV_CACHE_DIR"] == str(tmp_path / "cache/uv")
    assert env["CUDA_VISIBLE_DEVICES"] == ""
    assert str(tmp_path / "input") in argv


def test_resume_refuses_unrelated_systemd_unit():
    with pytest.raises(ValueError, match="Unexpected local scope"):
        local.stop_scope("ssh.service")
