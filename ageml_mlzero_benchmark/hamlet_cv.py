"""Matched AGE-ML/MLZero campaigns: native Gemini, seed 42, one-hour ten-fold CV.

This entry point is separate from the historical holdout runner. It never
imports AGE-ML. Candidate execution uses Docker or a dedicated local systemd scope.
"""

import argparse
import csv
import fcntl
import os
import shutil
import time
import uuid
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import KFold, StratifiedKFold

from . import (
    accounting,
    canonical_hash,
    mlzero,
    read_json,
    read_jsonl,
    runner,
    sha256_file,
    write_json,
)
from .cv_export import trusted_cv

PROTOCOL = "ageml-evaluation-v3/hamlet_cv"
EVALUATOR = mlzero.ROOT / "environments/mlzero/evaluate_cv.py"
MODELS = ("gemini-2.5-flash", "gemini-3.1-flash-lite")
INSTRUCTION = "Follow the mandatory harness-managed CV contract: return a self-contained build_model module and install dependencies in conda_env. The harness executes ten prescribed fresh-fold fits and supplies the MCTS score. Never fit on a holdout or all rows."
TASK = """# Ten-fold cross-validation benchmark: {dataset}
Target column: `{target}`. Task: {task}. Metric: {metric} ({direction} is better).

`data.csv` is the exact AGE-ML exported CSV. `cv_contract.json` fixes row positions
and all ten train/validation folds. There is NO independent test set. Do not
shuffle, split, filter, reorder or sample the dataset. Never add row IDs as features.

For EVERY candidate, return a self-contained Python factory module. The harness
writes it as `model_factory.py` in the node output folder. The module must provide `build_model(X_train, y_train)` returning a NEW, unfitted model with
`fit(X,y)` and `predict(X)` methods. All preprocessing must be inside that model,
fitted on the supplied fold training rows only. AutoGluon and other tools remain
allowed through a fit/predict wrapper. Do not tune on a fold's validation labels,
refit on all data, or fit in the factory. Use random_state/seed 42 for stochastic
models, including wrappers. Do not call another LLM, service or external dataset.

Return a SELF-CONTAINED Python MODULE defining build_model(X_train, y_train)
at module level. The harness writes this generated module to model_factory.py,
then executes the supplied evaluator in your node's conda_env after installation.
Your Bash script must create conda_env with required dependencies (numpy, pandas,
scikit-learn and your selected ML library); it need not train or print scores.
Do NOT run fitting, dataset reading, holdout splitting, prediction, or the
evaluator at module import time. Only imports, definitions and literal constants
are permitted at module top level. Put fit/predict logic in the returned model.
Do NOT run a separate holdout/full-data fit. Do NOT generate your own evaluation
or claim a validation score. The harness alone computes the mean of ten fold
scores, injects it into the existing MCTS reward (negative RMSE for regression),
and rejects candidates without completed valid CV exports. Earlier tutorial
examples using holdouts/results.csv do not apply to this benchmark. No final
full-data refit is required. See the mandatory node-level CV contract in prompts.
"""


def validate_contract(contract, frame):
    target = contract["target_column"]
    task = contract["task"]
    if (
        contract.get("evaluation_protocol") != "hamlet_cv"
        or contract.get("split_seed") != 42
    ):
        raise ValueError("Expected AGE-ML hamlet_cv / seed 42")
    if target not in frame or frame[target].isna().any():
        raise ValueError("Missing target or labels")
    if len(frame) != contract["source_row_count"]:
        raise ValueError("Row count mismatch")
    splitter = (
        StratifiedKFold(10, shuffle=False)
        if task == "classification"
        else KFold(10, shuffle=False)
    )
    expected = [
        {"train": a.tolist(), "validation": b.tolist()}
        for a, b in splitter.split(frame.drop(columns=[target]), frame[target])
    ]
    if contract.get("cv_folds") != expected:
        raise ValueError("AGE-ML folds differ from unshuffled ten-fold CV")


def bootstrap(age_campaign, age_data, output, cpus=16, memory_gib=50):
    age_campaign = Path(age_campaign).resolve()
    frozen = read_json(age_campaign / "campaign.json")
    cfg = frozen["config"]
    model = frozen["model"]
    name = model["model_name"]
    if (
        name not in MODELS
        or model.get("seed") != 42
        or model.get("temperature") != 0
        or cfg.get("run_seed") != 42
        or cfg.get("time_budget_seconds") != 3600
    ):
        raise ValueError(
            "AGE-ML campaign must use the requested Gemini model, seed 42, temperature 0, 3600 s"
        )
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Use a new config path; do not overwrite a frozen campaign")
    root = mlzero.ROOT / "artifacts"
    entries = []
    for task, group in cfg["datasets"].items():
        for key, openml_id in group.items():
            evidence = frozen["source_evidence"][key]
            contract = evidence["manifest"]
            source = Path(age_data).resolve() / task / f"{key}.csv"
            if (
                sha256_file(source) != contract["source_sha256"]
                or contract["source_sha256"] != evidence["source_sha256"]
            ):
                raise ValueError(f"{key}: CSV differs from AGE-ML campaign")
            validate_contract(contract, pd.read_csv(source))
            directory = (
                root
                / "datasets-cv"
                / key
                / (
                    contract["source_sha256"][:12]
                    + "-"
                    + canonical_hash(contract["cv_folds"])[:12]
                    + "-"
                    + sha256_file(EVALUATOR)[:12]
                )
            )
            directory.mkdir(parents=True, exist_ok=True)
            for file, raw in [
                (directory / "data.csv", source.read_bytes()),
                (directory / "evaluate_cv.py", EVALUATOR.read_bytes()),
            ]:
                if file.exists() and file.read_bytes() != raw:
                    raise ValueError(f"Prepared input conflict: {file}")
                file.write_bytes(raw)
            write_json(directory / "cv_contract.json", contract)
            metric = "balanced_accuracy" if task == "classification" else "rmse"
            (directory / "README.md").write_text(
                TASK.format(
                    dataset=key,
                    target=contract["target_column"],
                    task=task,
                    metric=metric,
                    direction="higher" if task == "classification" else "lower",
                )
            )
            entries.append(
                {
                    "key": key,
                    "openml_id": openml_id,
                    "task": task,
                    "target": contract["target_column"],
                    "objective": metric,
                    "prepared": str(directory),
                    "input_hashes": {
                        p.name: sha256_file(p)
                        for p in directory.iterdir()
                        if p.is_file()
                    },
                    "split_identity": evidence["split_identity"],
                }
            )
    resolved = {
        "schema_version": 1,
        "campaign_id": "mlzero-" + name.replace(".", "") + "-hamlet-cv-s42-1h",
        "protocol": PROTOCOL,
        "age_campaign": str(age_campaign),
        "age_campaign_identity": frozen["campaign_identity"],
        "age_manifest_sha256": sha256_file(age_campaign / "campaign.json"),
        "dataset_entries": entries,
        "repetitions": cfg["repetitions"],
        "mlzero": {
            "commit": "f228cd7028e197750391c02d94afcb7eea8da5ff",
            "image": "ageml-mlzero:hamlet-cv-native-gemini",
            "executor": "docker",
            "continuous_improvement": True,
            "max_iterations": 10000,
            "search_seed": 42,
            "hard_timeout_seconds": 3600,
            "instruction": INSTRUCTION,
            "config_overrides": {"remove_current_iteration_folder": False},
        },
        "llm": {
            "source": "explicit",
            "transport": "gemini",
            "model": name,
            "endpoint": "https://generativelanguage.googleapis.com/v1beta",
            "api_key_env": "GEMINI_API_KEY",
            "settings": {"seed": 42, "temperature": 0.0},
        },
        "resources": {"cpus": cpus, "memory_gib": memory_gib, "gpus": 0},
        "budget": {"search_wall_seconds": 3600, "stop_grace_seconds": 30},
        "pricing": "configs/pricing.yml",
        "outputs": {"artifact_root": str(root)},
        "seed_support": "Requested on native Gemini and local Python/NumPy/sklearn; backend determinism is not guaranteed",
        "timing_note": "MLZero initialization, generation, repair and all ten-fold fits are inside 3600 s. AGE-ML planning may precede its exploration deadline; resource-time equivalence is not asserted.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(resolved, sort_keys=False))
    return resolved


def validate_inputs(cfg):
    if (
        cfg["protocol"] != PROTOCOL
        or cfg["budget"]["search_wall_seconds"] != 3600
        or cfg["mlzero"]["search_seed"] != 42
        or cfg["llm"]["settings"] != {"seed": 42, "temperature": 0.0}
    ):
        raise ValueError("Frozen matched settings were changed")
    for ent in cfg["dataset_entries"]:
        directory = Path(ent["prepared"])
        if {p.name for p in directory.iterdir()} != set(ent["input_hashes"]):
            raise ValueError("Unexpected or missing search input")
        for name, h in ent["input_hashes"].items():
            if sha256_file(directory / name) != h:
                raise ValueError(f"Changed input: {ent['key']}/{name}")
        if sha256_file(directory / "evaluate_cv.py") != sha256_file(EVALUATOR):
            raise ValueError("Evaluator changed; bootstrap a new cohort")


def campaign_path(cfg):
    return Path(cfg["outputs"]["artifact_root"]) / "campaigns" / cfg["campaign_id"]


def check_image(cfg):
    import subprocess

    if cfg["mlzero"]["executor"] == "local":
        from .local_runtime import verify

        runtime = verify(cfg["mlzero"]["runtime_manifest"])
        if cfg.get("local_runtime_identity") != canonical_hash(runtime):
            raise ValueError(
                "Configuration refers to a different local runtime; bootstrap a new cohort"
            )
        subprocess.run(
            [
                runtime["python"],
                str(mlzero.ROOT / "environments/mlzero/check_retrieval.py"),
            ],
            check=True,
            env={
                **os.environ,
                "HF_HOME": str(Path(runtime["prefix"]) / "hf"),
                "HF_HUB_OFFLINE": "1",
                "CUDA_VISIBLE_DEVICES": "",
            },
        )
        subprocess.run(
            [
                runtime["python"],
                str(mlzero.ROOT / "environments/mlzero/check_cv_gate.py"),
                "--source",
                runtime["source"],
            ],
            check=True,
        )
        code = "from langchain_google_genai import ChatGoogleGenerativeAI as C; from autogluon.assistant.llm.gemini_chat import AssistantChatGemini as G; import importlib.metadata; g=G(model='gemini-2.5-flash',google_api_key='constructor-check-only',session_name='doctor',seed=42,temperature=0.0); assert g.seed==42 and g.temperature==0; assert importlib.metadata.version('langchain-google-genai')=='4.2.1'; assert 'seed' in C.model_fields"
        subprocess.run([runtime["python"], "-c", code], check=True)
        return {"client": "4.2.1", "runtime_identity": canonical_hash(runtime)}

    code = "import json,hashlib,pathlib,importlib.metadata; from langchain_google_genai import ChatGoogleGenerativeAI as C; from autogluon.assistant.llm.gemini_chat import AssistantChatGemini as G; g=G(model='gemini-2.5-flash',google_api_key='constructor-check-only',session_name='doctor',seed=42,temperature=0.0); assert g.seed==42 and g.temperature==0; print(json.dumps({'seed': 'seed' in C.model_fields,'client':importlib.metadata.version('langchain-google-genai'),'patches':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in pathlib.Path('/opt/bench/patches').glob('*.patch')}}))"
    result = subprocess.check_output(
        [
            "docker",
            "run",
            "--rm",
            "--entrypoint",
            "python",
            cfg["mlzero"]["image"],
            "-c",
            code,
        ],
        text=True,
    )
    import json

    observed = json.loads(result.strip().splitlines()[-1])
    if (
        not observed["seed"]
        or observed["client"] != "4.2.1"
        or observed["patches"] != mlzero.patch_hashes()
    ):
        raise ValueError(
            "Rebuild image: Gemini client or instrumentation patches differ"
        )
    return observed


def doctor(cfg, check_llm=False):
    validate_inputs(cfg)
    observed = check_image(cfg)
    print(
        f"Image/client verified: {observed['client']}; seed supported; patch hashes match"
    )
    key = os.environ.get(cfg["llm"]["api_key_env"])
    if not key:
        raise ValueError("GEMINI_API_KEY is missing")
    if check_llm:
        import json
        import urllib.request

        request = urllib.request.Request(
            "https://generativelanguage.googleapis.com/v1beta/models/"
            + cfg["llm"]["model"]
            + ":generateContent",
            data=json.dumps(
                {
                    "contents": [{"parts": [{"text": "Reply OK."}]}],
                    "generationConfig": {
                        "seed": 42,
                        "temperature": 0.0,
                        "maxOutputTokens": 16,
                    },
                }
            ).encode(),
            headers={"x-goog-api-key": key, "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            reply = json.load(response)
        print(
            "Gemini endpoint accepted model/seed/temperature:",
            reply.get("modelVersion", cfg["llm"]["model"]),
        )


def run(cfg, dry_run=False, only=None, only_repetition=None):
    validate_inputs(cfg)
    root = campaign_path(cfg)
    root.mkdir(parents=True, exist_ok=True)
    if not dry_run and not os.environ.get(cfg["llm"]["api_key_env"]):
        raise ValueError("GEMINI_API_KEY is missing")
    docker = cfg["mlzero"]["executor"] == "docker"
    if not dry_run and not docker:
        from .local_runtime import preflight

        image = canonical_hash(preflight(cfg))
        check_image(cfg)
    elif not dry_run:
        import subprocess

        subprocess.run(["docker", "info"], check=True, stdout=subprocess.DEVNULL)
        check_image(cfg)
        image = subprocess.check_output(
            ["docker", "image", "inspect", "-f", "{{.Id}}", cfg["mlzero"]["image"]],
            text=True,
        ).strip()
    else:
        image = "dry-run-no-image-check"
    repetitions = [
        r for r in cfg["repetitions"] if only_repetition is None or r == only_repetition
    ]
    if not repetitions:
        raise ValueError("Requested repetition is not in the frozen campaign")
    selected_count = sum(
        not only or ent["key"] in only for ent in cfg["dataset_entries"]
    ) * len(repetitions)
    print(
        f"PLAN: {selected_count} invocations; {selected_count} nominal search hours; seed=42; temperature=0; folds=10; model={cfg['llm']['model']}",
        flush=True,
    )
    records = []
    with (mlzero.ROOT / ".hamlet-cv.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for ent in cfg["dataset_entries"]:
            if only and ent["key"] not in only:
                continue
            for rep in repetitions:
                identity = {
                    "config": cfg,
                    "dataset": ent["key"],
                    "repetition": rep,
                    "patches": mlzero.patch_hashes(),
                    "image_digest": image,
                    "harness_sha256": sha256_file(Path(__file__)),
                    "adapter_sha256": sha256_file(Path(mlzero.__file__)),
                }
                if not docker:
                    identity["local_executor_sources"] = {
                        p.name: sha256_file(p)
                        for p in (
                            Path(__file__).with_name("local_runtime.py"),
                            Path(__file__).with_name("cv_export.py"),
                            Path(__file__).with_name("cv_gate.py"),
                            Path(mlzero.__file__).parent.parent
                            / "environments/mlzero/supervise.py",
                            Path(mlzero.__file__).parent.parent
                            / "environments/mlzero/search_seed/sitecustomize.py",
                        )
                    }
                rid = canonical_hash(identity)[:20]
                base = root / "runs" / ent["key"] / f"rep-{rep}" / rid
                finals = sorted(base.glob("attempt-*/record.json"))
                if dry_run:
                    print(
                        f"{ent['key']} rep={rep} seed=42 temperature=0 model={cfg['llm']['model']} search=3600s folds=10"
                    )
                    continue
                final_records = [
                    read_json(p) for p in finals if read_json(p).get("final", True)
                ]
                if final_records:
                    records.append(final_records[-1])
                    print(f"SKIP {ent['key']} rep={rep}", flush=True)
                    continue
                for abandoned in base.glob("attempt-*"):
                    if (abandoned / "record.json").exists():
                        continue
                    launch = abandoned / "launch.json"
                    if launch.exists() and not docker:
                        from .local_runtime import stop_scope

                        stop_scope(read_json(launch)["scope_unit"])
                    if launch.exists() and docker:
                        import subprocess

                        container = read_json(launch)["container_name"]
                        if not container.startswith("mlzero-cv-"):
                            raise ValueError("Unexpected abandoned container name")
                        subprocess.run(
                            ["docker", "rm", "-f", container],
                            capture_output=True,
                            check=False,
                        )
                    write_json(
                        abandoned / "record.json",
                        {
                            "system": "mlzero",
                            "dataset": ent["key"],
                            "repetition": rep,
                            "run_id": rid,
                            "status": "harness_interrupted",
                            "final": False,
                            "finished_at": time.time(),
                            "usage": accounting.summarize_usage(
                                read_jsonl(abandoned / "bench/usage.jsonl")
                            ),
                        },
                    )
                if shutil.disk_usage(root).free < 25 * 1024**3:
                    raise RuntimeError("Less than 25 GiB free; new search not started")
                attempt = base / f"attempt-{len(list(base.glob('attempt-*'))) + 1}"
                attempt.mkdir(parents=True)
                (attempt / "bench").mkdir()
                write_json(attempt / "identity.json", identity)
                (attempt / "mlzero.yaml").write_text(
                    yaml.safe_dump(mlzero.render_config(cfg), sort_keys=False)
                )
                prepared = Path(ent["prepared"])
                name = "mlzero-cv-" + uuid.uuid4().hex[:10]
                cache = Path(cfg["outputs"]["artifact_root"]) / "cache"
                cache.mkdir(exist_ok=True)
                write_json(
                    attempt / "launch.json",
                    {"container_name": name}
                    if docker
                    else {"scope_unit": name + ".scope"},
                )
                argv, env = mlzero.search_command(cfg, attempt, prepared, cache, name)
                env["OPENAI_API_KEY"] = os.environ[cfg["llm"]["api_key_env"]]
                env["AGEML_CV_DEADLINE"] = str(time.time() + 3600)
                print(
                    f"START {ent['key']} rep={rep} model={cfg['llm']['model']}",
                    flush=True,
                )
                try:
                    sup = runner._supervise(
                        argv,
                        env,
                        3600,
                        attempt / "search.log",
                        name if docker else None,
                        30,
                    )
                finally:
                    if not docker:
                        from .local_runtime import stop_scope

                        stop_scope(name + ".scope")
                col = mlzero.collect(attempt, docker)
                # Freeze native selection at the deadline, never choose by trusted score.
                eligible = [
                    c for c in col["commits"] if c["ts"] <= sup["started_at"] + 3600
                ]
                incumbent = eligible[-1] if eligible else None
                rec = {
                    "system": "mlzero",
                    "campaign_id": cfg["campaign_id"],
                    "run_id": rid,
                    "dataset": ent["key"],
                    "repetition": rep,
                    "model": cfg["llm"]["model"],
                    "protocol": PROTOCOL,
                    "fit_seed": 42,
                    "llm_seed_requested": 42,
                    "temperature": 0.0,
                    "search_budget_s": 3600,
                    "search_wall_s": sup["wall_s"],
                    "termination_reason": runner._termination(sup, col, docker),
                    "status": "no_artifact",
                    "cv_score": None,
                    "test_score": None,
                    "score_role": "selection_cv_not_independent_test",
                    "split_identity": ent["split_identity"],
                    "source_sha256": ent["input_hashes"]["data.csv"],
                    "resource_limits": cfg["resources"],
                    "seed_support": cfg["seed_support"],
                    "selected": incumbent,
                    "attempt": str(attempt),
                    "usage": accounting.summarize_usage(col["usage_rows"]),
                    "resources": accounting.summarize_resources(col["resources"]),
                    "timing_note": cfg["timing_note"],
                    "protocol_compliance": "audited export; no proof against adversarial generated code",
                }
                if incumbent:
                    folder = mlzero.to_host(incumbent["output_folder"], attempt, docker)
                    try:
                        rec.update(trusted_cv(folder, prepared))
                        rec["status"] = "completed"
                    except (OSError, ValueError, KeyError) as err:
                        rec["status"] = "invalid_cv_export"
                        rec["error"] = str(err)
                events = col["events"]
                chats = [e for e in events if e.get("event") == "chat_created"]
                ok = bool(chats) and all(
                    e["settings"].get("provider") == "gemini"
                    and e["settings"].get("model") == cfg["llm"]["model"]
                    and e["settings"].get("seed") == 42
                    and e["settings"].get("temperature") == 0
                    for e in chats
                )
                rec["native_validation_score"] = (
                    incumbent.get("native_validation_score") if incumbent else None
                )
                if (
                    rec["cv_score"] is not None
                    and rec["native_validation_score"] is not None
                ):
                    native_expected = (
                        rec["cv_score"]
                        if ent["task"] == "classification"
                        else -rec["cv_score"]
                    )
                    rec["native_vs_trusted_cv_discrepancy"] = (
                        rec["native_validation_score"] - native_expected
                    )
                rec["llm_settings_verified"] = ok
                if not ok and rec["status"] == "completed":
                    rec["status"] = "llm_settings_mismatch"
                rec["final"] = rec["termination_reason"] not in runner.RETRYABLE
                if rec["status"] != "completed" and rec["cv_score"] is not None:
                    rec["unpublished_cv_score"] = rec["cv_score"]
                    rec["cv_score"] = None
                rec["finished_at"] = time.time()
                write_json(attempt / "record.json", rec)
                records.append(rec)
                print(
                    f"END {ent['key']} rep={rep} {rec['status']} CV={rec['cv_score']}",
                    flush=True,
                )
                if (
                    rec["status"] == "no_artifact"
                    and rec["termination_reason"] == "process_error"
                ):
                    raise RuntimeError(
                        f"Search process crashed before producing a selectable export: {attempt / 'search.log'}. "
                        "Campaign stopped to avoid repeating a shared setup failure. Inspect this attempt before restarting."
                    )
    return records


def report(cfg):
    if (
        sha256_file(Path(cfg["age_campaign"]) / "campaign.json")
        != cfg["age_manifest_sha256"]
    ):
        raise ValueError("AGE-ML reference changed after bootstrap")
    root = campaign_path(cfg)
    rows = [read_json(p) for p in root.glob("runs/*/rep-*/*/attempt-*/record.json")]
    all_attempts = pd.DataFrame(rows)
    latest = {}
    for row in rows:
        rid = row["run_id"]
        previous = latest.get(rid)
        if previous is None or row.get("finished_at", 0) > previous.get(
            "finished_at", 0
        ):
            latest[rid] = row
    frame = pd.DataFrame(latest.values())
    if len(frame) and frame.duplicated(["dataset", "repetition"]).any():
        raise ValueError(
            "Multiple source/image cohorts in one campaign; use a new campaign_id"
        )
    out = root / "report"
    out.mkdir(exist_ok=True)
    all_attempts.to_csv(out / "all_attempts.csv", index=False)
    frame.to_csv(out / "mlzero_runs.csv", index=False)
    age = []
    for path in Path(cfg["age_campaign"]).glob(
        "baseline/run-*/ageml-evaluation-v3/results.csv"
    ):
        rep = int(path.parent.parent.name.removeprefix("run-"))
        with path.open(newline="") as stream:
            for r in csv.DictReader(stream):
                age.append(
                    {
                        "dataset": r["Dataset"],
                        "repetition": rep,
                        "age_status": r["Status"],
                        "age_cv_score": r[
                            "Search CV Score (selection; not independent test)"
                        ],
                    }
                )
    pairs = (
        frame.merge(
            pd.DataFrame(age),
            on=["dataset", "repetition"],
            how="left",
            validate="one_to_one",
        )
        if len(frame) and age
        else frame
    )
    pairs.to_csv(out / "paired_cv_scores.csv", index=False)
    (out / "README.md").write_text(
        "Scores are selected mean ten-fold CV scores, NOT independent test scores. Failed outcomes stay in denominators. No p-values; repetitions share seeds and folds. AGE-ML Original only. Resource-time equality is not claimed. Generated-factory fold compliance requires code review; exported hashes and OOF predictions are audited.\n"
    )
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    b = sub.add_parser("bootstrap")
    b.add_argument("--age-campaign", required=True, type=Path)
    b.add_argument("--age-data", required=True, type=Path)
    b.add_argument("--output", required=True, type=Path)
    b.add_argument(
        "--local-runtime",
        type=Path,
        help="Use a sealed local runtime instead of Docker",
    )
    b.add_argument(
        "--campaign-suffix", default="", help="Separate cohort label, e.g. retrieval-v2"
    )
    b.add_argument("--cpus", type=int, default=16)
    b.add_argument("--memory-gib", type=int, default=50)
    r = sub.add_parser("run")
    r.add_argument("--config", required=True, type=Path)
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--only", nargs="*")
    r.add_argument("--only-repetition", type=int)
    q = sub.add_parser("report")
    q.add_argument("--config", required=True, type=Path)
    d = sub.add_parser("doctor")
    d.add_argument("--config", required=True, type=Path)
    d.add_argument("--check-llm", action="store_true")
    a = p.parse_args(argv)
    if a.command == "bootstrap":
        cfg = bootstrap(a.age_campaign, a.age_data, a.output, a.cpus, a.memory_gib)
        if a.local_runtime:
            from .local_runtime import configure

            cfg = configure(cfg, a.local_runtime)
        if a.campaign_suffix:
            import re

            if not re.fullmatch(r"[a-z0-9-]+", a.campaign_suffix):
                raise ValueError(
                    "Campaign suffix must contain only lower-case letters, numbers or hyphens"
                )
            cfg["campaign_id"] += "-" + a.campaign_suffix
        a.output.write_text(yaml.safe_dump(cfg, sort_keys=False))
        print(
            f"Prepared {len(cfg['dataset_entries'])} datasets; {len(cfg['repetitions'])} repetitions; model={cfg['llm']['model']}; seed=42; budget=3600 s"
        )
    else:
        cfg = yaml.safe_load(a.config.read_text())
        if a.command == "run":
            run(cfg, a.dry_run, a.only, a.only_repetition)
        elif a.command == "doctor":
            doctor(cfg, a.check_llm)
        else:
            print(report(cfg))


if __name__ == "__main__":
    main()
