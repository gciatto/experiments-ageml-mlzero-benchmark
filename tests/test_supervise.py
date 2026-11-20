"""Exercise POSIX supervision when Linux cgroup files are unavailable."""

import importlib.util
import io
import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

SUPERVISOR = Path(__file__).resolve().parents[1] / "environments/mlzero/supervise.py"


@pytest.fixture
def supervisor():
    spec = importlib.util.spec_from_file_location("supervisor_under_test", SUPERVISOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("exit_code", [0, 7])
def test_child_runs_and_final_resources_are_written_without_proc(
    tmp_path: Path, exit_code: int
) -> None:
    # Run the actual script in a subprocess with only the /proc read unavailable.
    # This reproduces the macOS startup failure even on a Linux test host.
    bootstrap = """
import builtins
import runpy
import sys

original_open = builtins.open
def without_proc(path, *args, **kwargs):
    if str(path) == '/proc/self/cgroup':
        raise FileNotFoundError('/proc/self/cgroup')
    return original_open(path, *args, **kwargs)

builtins.open = without_proc
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
"""
    resources = tmp_path / "resources.json"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            bootstrap,
            str(SUPERVISOR),
            str(resources),
            "--",
            sys.executable,
            "-c",
            f"print('child started'); raise SystemExit({exit_code})",
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == exit_code, result.stderr
    assert "child started" in result.stdout
    sample = json.loads(resources.read_text())
    assert sample["final"] is True
    assert sample["exit_code"] == exit_code
    assert sample["cgroup_cpu_s"] is None
    assert sample["cgroup_peak_bytes"] is None
    assert sample["rusage_children_cpu_s"] >= 0
    assert sample["rusage_children_maxrss_bytes"] > 0


@pytest.mark.parametrize("error", [FileNotFoundError, PermissionError])
def test_missing_or_inaccessible_proc_has_no_cgroup(supervisor, monkeypatch, error):
    def unavailable(*args, **kwargs):
        raise error("cgroup file unavailable")

    monkeypatch.setattr("builtins.open", unavailable)
    assert supervisor.cgroup_path() is None


@pytest.mark.parametrize(
    "contents,expected",
    [
        ("0::/\n", "/sys/fs/cgroup/"),
        ("0::/system.slice/mlzero.scope\n", "/sys/fs/cgroup/system.slice/mlzero.scope"),
        ("2:cpu:/legacy\n", None),
    ],
)
def test_cgroup_v2_paths_and_unsupported_v1(
    supervisor, monkeypatch, contents, expected
):
    monkeypatch.setattr("builtins.open", lambda *a, **kw: io.StringIO(contents))
    assert supervisor.cgroup_path() == expected


@pytest.mark.parametrize("platform,multiplier", [("darwin", 1), ("linux", 1024)])
def test_rusage_fallback_uses_platform_rss_units(
    supervisor, monkeypatch, platform, multiplier
):
    monkeypatch.setattr(supervisor, "CG", None)
    monkeypatch.setattr(supervisor.sys, "platform", platform)
    monkeypatch.setattr(
        supervisor.resource,
        "getrusage",
        lambda _: SimpleNamespace(ru_utime=0.25, ru_stime=0.5, ru_maxrss=123456),
    )
    sample = supervisor.sample(time.time())
    assert sample["cgroup_cpu_s"] is None
    assert sample["cgroup_peak_bytes"] is None
    assert sample["rusage_children_cpu_s"] == 0.75
    assert sample["rusage_children_maxrss_bytes"] == 123456 * multiplier


def test_linux_cgroup_measurements_remain_available(supervisor, monkeypatch, tmp_path):
    (tmp_path / "cpu.stat").write_text("usage_usec 1250000\nuser_usec 1000000\n")
    (tmp_path / "memory.peak").write_text("4096\n")
    monkeypatch.setattr(supervisor, "CG", str(tmp_path))
    sample = supervisor.sample(time.time())
    assert sample["cgroup_cpu_s"] == 1.25
    assert sample["cgroup_peak_bytes"] == 4096
