"""MLZero adapter: native input/config rendering, launch commands, artifact collection.

The benchmark talks to MLZero only through its CLI and the files written by
``patches/0001-ageml-bench-instrumentation.patch`` (``bench_hooks``) under
``$AGEML_BENCH_DIR``.
"""

import re
from pathlib import Path
from typing import Any

import yaml

from . import OBJECTIVES, read_json, read_jsonl, sha256_file, sha256_tree

ROOT = Path(__file__).resolve().parent.parent
UPSTREAM_DEFAULT = ROOT / "configs/mlzero/upstream-default-f228cd7.yaml"
PATCHES = sorted((ROOT / "patches").glob("*.patch"))
#: every LLM-backed agent section in the upstream default config
AGENTS = (
    "python_coder",
    "bash_coder",
    "executer",
    "meta_prompting",
    "reader",
    "error_analyzer",
    "retriever",
    "reranker",
    "description_file_retriever",
    "task_descriptor",
    "tool_selector",
)
TASK_FILE = "README.md"
EXPORT_ENTRY_POINT = "predict.py"
CONTAINER = {"input": "/bench/input", "run": "/bench/run", "cache": "/bench/cache"}

TASK_TEMPLATE = """\
# Task: {task} on `{dataset}`

Predict the column `{target}`. Evaluation metric: {objective} ({direction} is better).

## Files
- `train.csv`: labeled FITTING data ({n_train} rows).
- `val.csv`: labeled VALIDATION data ({n_val} rows). This is the validation set; do not
  hold out another one from `train.csv`.
- `val_features.csv`: the rows of `val.csv` without `{target}`. Treat it as the test file:
  write `results.csv` with one prediction per row, same order, column `{target}`.

## Data-use rules (benchmark protocol, no exceptions)
1. Fit every preprocessing step and every base model on `train.csv` only.
2. Use `val.csv` labels only for selection: model/hyperparameter selection, early
   stopping, ensemble weights, thresholds or calibration. A learned stacker must be
   trained on `train.csv` (or out-of-fold predictions on it), never on `val.csv`.
3. Never refit anything on train+validation. With AutoGluon pass `tuning_data=val`,
   `refit_full=False`, `set_best_to_refit_full=False` (and `use_bag_holdout=True` if
   bagging), and never call `refit_full()`.
4. Do not drop rows or features, and keep `{target}` in its original labels/units.

## Required export (in the output folder)
- Save the fitted model and preprocessing objects inside the output folder.
- Write `{entry}` so that
  `python {entry} --input <features.csv> --output <predictions.csv>`
  loads ONLY the saved fitted objects (paths relative to `Path(__file__).parent`), does
  no training, no network access and no package installation, and writes a CSV with a
  single column `prediction`, one row per input row in the same order, with original
  class labels (classification) or original target units (regression).
- Write `data_use.json` stating which data each component was fitted on and how
  validation labels were used.
"""

INSTRUCTION = (
    "Follow README.md exactly: fit on train.csv, select with val.csv, never refit on "
    "train+validation, write results.csv for val_features.csv, and export predict.py "
    "with the saved fitted model."
)


def task_text(entry: dict[str, Any], manifest: dict[str, Any]) -> str:
    return TASK_TEMPLATE.format(
        task=entry["task"],
        dataset=entry["key"],
        target=entry["target"],
        objective=entry["objective"],
        direction="higher" if OBJECTIVES[entry["objective"]] else "lower",
        n_train=manifest["counts"]["train"],
        n_val=manifest["counts"]["validation"],
        entry=EXPORT_ENTRY_POINT,
    )


def render_config(resolved: dict[str, Any]) -> dict[str, Any]:
    """Benchmark overrides of the upstream YAML.

    Every agent section is written out in full: upstream merges ``llm`` into the
    agent sections with YAML anchors at load time, so overriding only the
    top-level ``llm`` would leave every agent on the default provider.
    """
    upstream = yaml.safe_load(UPSTREAM_DEFAULT.read_text())
    llm = resolved["llm"]
    common = {"provider": "openai", "model": llm["model"], "proxy_url": llm["endpoint"]}
    common.update(llm.get("settings") or {})
    cfg: dict[str, Any] = {
        "continuous_improvement": resolved["mlzero"]["continuous_improvement"],
        "llm": {**upstream["llm"], **common},
    }
    for agent in AGENTS:
        cfg[agent] = {**upstream[agent], **common}
    cfg.update(resolved["mlzero"].get("config_overrides") or {})
    return cfg


def verify_llm(
    effective_yaml: Path, events: list[dict[str, Any]], resolved: dict[str, Any]
) -> dict[str, Any]:
    """Check that every agent got the requested model, endpoint and settings."""
    llm = resolved["llm"]
    want = {
        "model": llm["model"],
        "proxy_url": llm["endpoint"],
        **(llm.get("settings") or {}),
    }
    problems = []
    if not effective_yaml.exists():
        return {"status": "unverifiable", "problems": ["effective_config.yaml missing"]}
    eff = yaml.safe_load(effective_yaml.read_text())
    for agent in AGENTS:
        sec = eff.get(agent) or {}
        if sec.get("provider") != "openai":
            problems.append(f"{agent}: provider {sec.get('provider')}")
        for k, v in want.items():
            if sec.get(k) != v:
                problems.append(f"{agent}.{k}={sec.get(k)!r}, wanted {v!r}")
    chats = [e for e in events if e.get("event") == "chat_created"]
    for e in chats:
        s = e.get("settings") or {}
        if s.get("model") != llm["model"]:
            problems.append(f"session {e.get('session')}: model {s.get('model')}")
        if (
            s.get("base_url") != llm["endpoint"]
            or s.get("env_base_url") != llm["endpoint"]
        ):
            problems.append(
                f"session {e.get('session')}: chat/discovery endpoint mismatch"
            )
    cli: Any = (
        next((e.get("cli") for e in events if e.get("event") == "effective_config"), {})
        or {}
    )
    if cli.get("max_iterations") != resolved["mlzero"]["max_iterations"]:
        problems.append(f"max_iterations={cli.get('max_iterations')}")
    if cli.get("continuous_improvement") is not True:
        problems.append("continuous_improvement not enabled")
    return {
        "status": "fail" if problems else "pass",
        "problems": problems,
        "chat_sessions": len(chats),
    }


#: generated-code patterns that suggest a no-refit / data-use violation
CODE_FLAGS = {
    "refit_full": r"refit_full\s*\(|set_best_to_refit_full\s*=\s*True|refit_full\s*=\s*True",
    "extra_split": r"train_test_split|holdout_frac",
    "concat_train_val": r"concat\(\s*\[[^\]]*val",
    "test_access": r"test_labels|/bench/test|private/",
}


def audit_code(node_folder: Path) -> dict[str, Any]:
    files = [p for p in node_folder.glob("*.py")] + [
        node_folder / "output" / EXPORT_ENTRY_POINT
    ]
    flags: dict[str, list[str]] = {}
    for f in files:
        if not f.is_file():
            continue
        text = f.read_text(errors="replace")
        for name, rx in CODE_FLAGS.items():
            if re.search(rx, text):
                flags.setdefault(name, []).append(f.name)
    data_use = node_folder / "output" / "data_use.json"
    return {
        # static patterns cannot prove compliance: absence of flags is "unverified"
        "status": "flagged" if flags else "no_flags_unverified",
        "flags": flags,
        "data_use_declared": read_json(data_use) if data_use.exists() else None,
    }


def to_host(path: str, attempt_dir: Path, docker: bool) -> Path:
    if docker and path.startswith(CONTAINER["run"]):
        return attempt_dir / Path(path).relative_to(CONTAINER["run"])
    return Path(path)


def search_command(
    resolved: dict[str, Any],
    attempt: Path,
    input_dir: Path,
    cache: Path,
    container_name: str,
) -> tuple[list[str], dict[str, str]]:
    """argv and extra environment for one search (secrets via env, never argv)."""
    m = resolved["mlzero"]
    docker = m["executor"] == "docker"
    run = CONTAINER["run"] if docker else str(attempt)
    inp = CONTAINER["input"] if docker else str(input_dir)
    supervise = (
        "/opt/bench/supervise.py"
        if docker
        else str(ROOT / "environments/mlzero/supervise.py")
    )
    python = "python" if docker else m.get("python", "python3")
    mlzero = ["mlzero"] if docker else list(m["command"])
    core = [
        python,
        supervise,
        f"{run}/bench/resources.json",
        "--",
        *mlzero,
        "-i",
        inp,
        "-o",
        f"{run}/mlzero",
        "-c",
        f"{run}/mlzero.yaml",
        "-n",
        str(m["max_iterations"]),
        "-t",
        INSTRUCTION,
        "-v",
        str(m.get("verbosity", 2)),
    ]
    if m["continuous_improvement"]:
        core.insert(core.index("-t"), "--continuous_improvement")
    env = {
        "AGEML_BENCH_DIR": f"{run}/bench",
        "OPENAI_BASE_URL": resolved["llm"]["endpoint"],
        "OPENAI_API_KEY": "",  # filled from llm.api_key_env by the runner
    }
    if not docker:
        return core, env
    res = resolved["resources"]
    cache_c = CONTAINER["cache"]
    env.update({"UV_CACHE_DIR": f"{cache_c}/uv", "CONDA_PKGS_DIRS": f"{cache_c}/conda"})
    argv = [
        "docker",
        "run",
        "--name",
        container_name,
        "--init",
        f"--cpus={res['cpus']}",
        f"--memory={res['memory_gib']}g",
        f"--memory-swap={res['memory_gib']}g",
        "-v",
        f"{input_dir}:{CONTAINER['input']}:ro",
        "-v",
        f"{attempt}:{CONTAINER['run']}",
        "-v",
        f"{cache}:{cache_c}",
    ]
    for k in env:
        argv += ["-e", k]  # value inherited from the launching environment
    return [*argv, m["image"], *core], env


def inference_command(
    resolved: dict[str, Any],
    attempt: Path,
    incumbent: dict[str, Any],
    features_dir: Path,
    out_dir: Path,
    container_name: str,
) -> list[str]:
    """No network, no credentials, fit-call guard on, preserved node environment."""
    m = resolved["mlzero"]
    docker = m["executor"] == "docker"
    node = incumbent["node_folder"]
    output = incumbent["output_folder"]
    env_python = f"{node}/conda_env/bin/python"
    if not to_host(env_python, attempt, docker).exists():
        env_python = "python" if docker else m.get("python", "python3")
    supervise = (
        "/opt/bench/supervise.py"
        if docker
        else str(ROOT / "environments/mlzero/supervise.py")
    )
    feats = "/bench/test" if docker else str(features_dir)
    out = "/bench/out" if docker else str(out_dir)
    core = [
        env_python,
        f"{output}/{EXPORT_ENTRY_POINT}",
        "--input",
        f"{feats}/test_features.csv",
        "--output",
        f"{out}/predictions.csv",
    ]
    if not docker:
        return ["python3", supervise, f"{out}/resources.json", "--", *core]
    res = resolved["resources"]
    guard = "/opt/bench/guard"
    script = (
        f"uv pip freeze --python {env_python} > {out}/env-freeze.txt 2>&1; "
        f"exec python /opt/bench/supervise.py {out}/resources.json -- " + " ".join(core)
    )
    return [
        "docker",
        "run",
        "--rm",
        "--name",
        container_name,
        "--network",
        "none",
        f"--cpus={res['cpus']}",
        f"--memory={res['memory_gib']}g",
        f"--memory-swap={res['memory_gib']}g",
        "-e",
        f"PYTHONPATH={guard}",
        "-e",
        f"AGEML_BENCH_FIT_LOG={out}/fit_calls.jsonl",
        "-v",
        f"{attempt}:{CONTAINER['run']}",
        "-v",
        f"{features_dir}:/bench/test:ro",
        "-v",
        f"{out_dir}:/bench/out",
        "-w",
        output,
        m["image"],
        "sh",
        "-c",
        script,
    ]


def collect(attempt: Path, docker: bool) -> dict[str, Any]:
    """Read everything the search left behind. Works after a hard kill."""
    bench = attempt / "bench"
    events = read_jsonl(bench / "events.jsonl")
    incumbent = (
        read_json(bench / "incumbent.json")
        if (bench / "incumbent.json").exists()
        else None
    )
    out: dict[str, Any] = {
        "events": events,
        "usage_rows": read_jsonl(bench / "usage.jsonl"),
        "resources": read_json(bench / "resources.json")
        if (bench / "resources.json").exists()
        else None,
        "incumbent": incumbent,
    }
    if incumbent:
        node = to_host(incumbent["node_folder"], attempt, docker)
        output = node / "output"
        out["node_folder_host"] = node
        out["export_present"] = (output / EXPORT_ENTRY_POINT).is_file()
        out["asset_hashes"] = sha256_tree(output) if output.exists() else {}
        out["code_audit"] = audit_code(node)
        out["native_results"] = (
            output / "results.csv" if (output / "results.csv").exists() else None
        )
    stops = [e["reason"] for e in events if e.get("event") == "search_stopped"]
    out["native_stop"] = stops[-1] if stops else None
    out["returned"] = any(e.get("event") == "run_agent_returned" for e in events)
    out["steps"] = [e for e in events if e.get("event") == "step"]
    out["commits"] = [e for e in events if e.get("event") == "incumbent_committed"]
    return out


def patch_hashes() -> dict[str, str]:
    return {p.name: sha256_file(p) for p in PATCHES}
