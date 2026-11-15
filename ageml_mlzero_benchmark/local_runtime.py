"""Local MLZero runtime provenance and per-search systemd resource boundaries."""

import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

from . import canonical_hash, mlzero, read_json, sha256_file, write_json


def snapshot(prefix):
    prefix = Path(prefix).resolve()
    source = prefix / "source"
    python = prefix / "env/bin/python"
    git = lambda *args: subprocess.check_output(["git", "-C", str(source), *args])
    if (
        git("rev-parse", "HEAD").decode().strip()
        != "f228cd7028e197750391c02d94afcb7eea8da5ff"
    ):
        raise ValueError("Local MLZero checkout does not match the pinned commit")
    # Includes the staged new instrumentation files; catches subsequent edits too.
    files = git("ls-files", "-z").decode().split("\0")
    hashes = {p: sha256_file(source / p) for p in files if p and (source / p).is_file()}
    freeze = subprocess.check_output([str(python), "-m", "pip", "freeze"], text=True)
    return {
        "prefix": str(prefix),
        "source": str(source),
        "python": str(python),
        "source_hashes": hashes,
        "freeze": sorted(freeze.splitlines()),
        "patches": mlzero.patch_hashes(),
        "python_version": subprocess.check_output(
            [str(python), "--version"], text=True
        ).strip(),
    }


def verify(manifest):
    recorded = read_json(Path(manifest))
    actual = snapshot(recorded["prefix"])
    if actual != recorded:
        raise ValueError(
            "Local runtime changed; install/seal a new runtime and create a new cohort"
        )
    return recorded


def configure(cfg, manifest):
    runtime = verify(manifest)
    cfg["campaign_id"] += "-local"
    cfg["mlzero"].update(
        executor="local",
        runtime_manifest=str(Path(manifest).resolve()),
        python=runtime["python"],
        command=[str(Path(runtime["prefix"]) / "env/bin/mlzero")],
        conda_bin=str(Path(runtime["prefix"]) / "conda/bin"),
    )
    cfg["resources"]["memory_high_gib"] = min(40, cfg["resources"]["memory_gib"])
    for entry in cfg["dataset_entries"]:
        prepared = Path(entry["prepared"])
        # Separate local inputs; do not mutate an existing Docker cohort's contract.
        local = prepared.with_name(prepared.name + "-local")
        import shutil

        local.mkdir(parents=True, exist_ok=True)
        for name in entry["input_hashes"]:
            shutil.copy2(prepared / name, local / name)
        readme = local / "README.md"
        readme.write_text(
            readme.read_text().replace(
                "/bench/input/evaluate_cv.py", str(local / "evaluate_cv.py")
            )
        )
        entry["prepared"] = str(local)
        entry["input_hashes"] = {
            p.name: sha256_file(p) for p in local.iterdir() if p.is_file()
        }
    cfg["local_runtime_identity"] = canonical_hash(runtime)
    return cfg


def scoped_command(core, cfg, unit):
    res = cfg["resources"]
    return [
        "systemd-run",
        "--scope",
        "--quiet",
        "--unit=" + unit,
        "-p",
        "MemoryAccounting=yes",
        "-p",
        f"MemoryHigh={res.get('memory_high_gib', 40)}G",
        "-p",
        f"MemoryMax={res['memory_gib']}G",
        "-p",
        "MemorySwapMax=0",
        "-p",
        f"CPUQuota={res['cpus'] * 100}%",
        "-p",
        "RuntimeMaxSec=3630s",
        "--",
        "timeout",
        "--signal=TERM",
        "--kill-after=30s",
        "3600s",
        *core,
    ]


def stop_scope(unit):
    if not unit.startswith("mlzero-cv-") or not unit.endswith(".scope"):
        raise ValueError("Unexpected local scope name")
    subprocess.run(["systemctl", "stop", unit], capture_output=True, check=False)


def preflight(cfg):
    if os.geteuid() != 0:
        raise ValueError(
            "Local systemd runner requires root inside the experiment container"
        )
    subprocess.run(
        scoped_command(
            ["/bin/true"],
            cfg,
            "mlzero-cv-preflight-" + uuid.uuid4().hex[:10] + ".scope",
        ),
        check=True,
    )
    return verify(cfg["mlzero"]["runtime_manifest"])


if __name__ == "__main__":
    # Called only after the setup script has applied the pinned patches and installed deps.
    prefix = Path(sys.argv[1]).resolve()
    state = snapshot(prefix)
    write_json(prefix / "runtime.json", state)
    print(
        json.dumps(
            {
                "runtime_manifest": str(prefix / "runtime.json"),
                "identity": canonical_hash(state),
            },
            indent=2,
        )
    )
