"""Canonical MLZero campaign paths and explicit continuation after relocation."""

import argparse
import copy
import fcntl
import getpass
import hashlib
import json
import os
import shutil
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
N25 = "mlzero-gemini-25-flash-hamlet-cv-s42-1h-local-factory-v5"
N31 = "mlzero-gemini-31-flash-lite-hamlet-cv-s42-1h-local-factory-v5"
C25 = "gemini25-hamlet-cv-local-factory-v5.yml"
C31 = "gemini31-hamlet-cv-local-factory-v5.yml"
REL = "src/autogluon/assistant/llm/gemini_chat.py"
METHOD = """    def assistant_chat(self, message: str) -> str:
        response = BaseAssistantChat.assistant_chat(self, message)
        if isinstance(response, str):
            return response
        if isinstance(response, list):
            parts = []
            for block in response:
                if isinstance(block, str):
                    parts.append(block)
                elif (
                    isinstance(block, dict)
                    and block.get("type") == "text"
                    and isinstance(block.get("text"), str)
                ):
                    parts.append(block["text"])
            text = "".join(parts)
            if text.strip():
                return text
        raise ValueError("Gemini response has no usable text.")

"""


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".writing")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def digest(data):
    return hashlib.sha256(data).hexdigest()


@contextmanager
def checkout_idle():
    with (ROOT / ".hamlet-cv.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Campagna attiva: nessuna modifica. Riprova al termine.")
        scopes = subprocess.check_output(
            [
                "systemctl",
                "list-units",
                "--type=scope",
                "--state=running",
                "--no-legend",
                "mlzero-cv-*",
            ],
            text=True,
        )
        if "mlzero-cv-" in scopes:
            raise SystemExit("Ricerca MLZero attiva: nessuna modifica.")
        yield


def runtime_files(prefix):
    old = read(prefix / "runtime.json")
    fixed_path = prefix / "runtime-gemini-textfix.json"
    fixed = (
        read(fixed_path)
        if fixed_path.exists()
        else read(prefix / "runtime-gemini31.json")
    )
    candidates = [
        prefix / "source" / REL,
        prefix / "gemini_chat.py.before-textfix",
        prefix / "campaign-adapters/gemini25.py",
        prefix / "campaign-adapters/gemini31.py",
    ]
    blobs = [p.read_bytes() for p in candidates if p.is_file()]
    plain = next((b for b in blobs if digest(b) == old["source_hashes"][REL]), None)
    if plain is None:
        raise ValueError("Adattatore 2.5 originale non trovato; nessuna migrazione.")
    corrected = next(
        (b for b in blobs if digest(b) == fixed["source_hashes"][REL]), None
    )
    if corrected is None:
        source = plain.decode()
        anchor = "    def describe(self):\n"
        if source.count(anchor) != 1 or "    def assistant_chat(" in source:
            raise ValueError("Adattatore inatteso.")
        corrected = source.replace(anchor, METHOD + anchor, 1).encode()
    if digest(corrected) != fixed["source_hashes"][REL]:
        raise ValueError(
            "Correzione Gemini 3.1 diversa dal manifest: nessuna modifica."
        )
    return old, fixed, plain, corrected


def is_diagnostic(attempt, row, model):
    if not row.get("final", True) or row.get("status") == "harness_interrupted":
        return True
    if row.get("status") == "completed":
        return False
    log = attempt / "search.log"
    text = log.read_text(errors="replace") if log.exists() else ""
    if row.get("termination_reason") == "process_error":
        if "GenerateRequestsPerDayPerProjectPerModel-FreeTier" in text:
            return True
        if model == "31" and "expected string or bytes-like object, got 'list'" in text:
            return True
    return False


def relocate_paths(value, old_root, new_root):
    if isinstance(value, dict):
        return {k: relocate_paths(v, old_root, new_root) for k, v in value.items()}
    if isinstance(value, list):
        return [relocate_paths(v, old_root, new_root) for v in value]
    if isinstance(value, str):
        return value.replace(str(old_root), str(new_root))
    return value


def move_attempt(source, destination):
    if destination.exists():
        raise ValueError(f"Destinazione gia presente: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(destination))


def records(root):
    return sorted(root.glob("runs/*/rep-*/*/attempt-*/record.json"))


def organize():
    import yaml
    from ageml_mlzero_benchmark import canonical_hash
    from ageml_mlzero_benchmark.local_runtime import snapshot

    with checkout_idle():
        prefix = ROOT / ".mlzero-local"
        old, fixed, plain, corrected = runtime_files(prefix)
        actual = snapshot(prefix)
        normalized = copy.deepcopy(actual)
        normalized["source_hashes"][REL] = old["source_hashes"][REL]
        if normalized != old:
            raise ValueError("Altre differenze nel runtime: nessuna migrazione.")
        previous_cfg = ROOT / "configs" / C31
        corrected_cfg = previous_cfg.with_name(previous_cfg.stem + "-textfix.yml")
        cfg = yaml.safe_load(
            (corrected_cfg if corrected_cfg.exists() else previous_cfg).read_text()
        )
        if cfg["llm"]["model"] != "gemini-3.1-flash-lite":
            raise ValueError("Modello inatteso nella configurazione 3.1.")
        canonical31 = ROOT / cfg["outputs"]["artifact_root"] / "campaigns" / N31
        legacy31 = canonical31.with_name(N31 + "-textfix")
        cfg25 = yaml.safe_load((ROOT / "configs" / C25).read_text())
        canonical25 = ROOT / cfg25["outputs"]["artifact_root"] / "campaigns" / N25
        diagnostics = ROOT / "artifacts/diagnostics/campaign-layout"
        plan = []
        marker = ROOT / "artifacts/campaign-layout.json"
        already_corrected = (
            marker.exists()
            and read(marker).get("numeric_revision") == "rmse-float32-compat-v1"
        )
        corrected_validator = ROOT / "scripts/cv_export_corrected.py"
        validator = ROOT / "ageml_mlzero_benchmark/cv_export.py"
        compile(corrected_validator.read_text(), str(corrected_validator), "exec")
        if not validator.is_file():
            raise ValueError("Modulo CV originale mancante: nessuna modifica.")
        if (
            already_corrected
            and digest(validator.read_bytes()) != read(marker)["validator_sha256"]
        ):
            raise ValueError(
                "Il validatore e cambiato dopo la correzione: nessuna modifica."
            )
        if not already_corrected:
            # Every prior regression search was exposed to the same gate defect,
            # including searches that happened to return a valid winner.
            for model, folder in [
                ("25", canonical25),
                ("31", canonical31),
                ("31", legacy31),
            ]:
                for dataset in ("ames_housing", "auto_mpg", "california_housing"):
                    directory = folder / "runs" / dataset
                    if directory.exists():
                        group = "original" if folder != legacy31 else "corrected"
                        plan.append(
                            (
                                directory,
                                diagnostics
                                / f"gemini{model}/regression-before-numeric-fix"
                                / group
                                / dataset,
                            )
                        )
        # First reject ambiguous duplicates; never select by score.
        known = {}
        for path in records(legacy31):
            row = read(path)
            if not is_diagnostic(path.parent, row, "31"):
                key = row["dataset"], row["repetition"]
                if key in known:
                    raise ValueError(f"Due esiti 3.1 per {key}: migrazione annullata.")
                known[key] = row
        for path in records(canonical31):
            row = read(path)
            key = row["dataset"], row["repetition"]
            if key in known:
                # The earlier manual copy is the SAME export, not another run.
                if row.get("status") == "completed" and row.get("selected") != known[
                    key
                ].get("selected"):
                    raise ValueError(f"Duplicati diversi per {key}: nessuna modifica.")
                plan.append(
                    (
                        path.parent,
                        diagnostics
                        / "gemini31/replaced"
                        / path.parent.relative_to(canonical31),
                    )
                )
            elif is_diagnostic(path.parent, row, "31"):
                plan.append(
                    (
                        path.parent,
                        diagnostics
                        / "gemini31/errors"
                        / path.parent.relative_to(canonical31),
                    )
                )
        for model, folder in [("25", canonical25), ("31", legacy31)]:
            for path in records(folder):
                if is_diagnostic(path.parent, read(path), model):
                    plan.append(
                        (
                            path.parent,
                            diagnostics
                            / f"gemini{model}/errors"
                            / path.parent.relative_to(folder),
                        )
                    )
        for src, dst in plan:
            if dst.exists():
                raise ValueError(f"Archivio diagnostico gia presente: {dst}")
        # With all scopes stopped, a 3.1 attempt lacking its final record is
        # abandoned. Preserve it as diagnostics rather than a campaign outcome.
        for folder in (canonical31, legacy31):
            for attempt in sorted(folder.glob("runs/*/rep-*/*/attempt-*")):
                if not (attempt / "record.json").exists():
                    plan.append(
                        (
                            attempt,
                            diagnostics
                            / "gemini31/interrupted"
                            / attempt.relative_to(folder),
                        )
                    )
        for label, folder in [("previous", canonical31), ("corrected", legacy31)]:
            if (folder / "report").exists():
                plan.append(
                    (folder / "report", diagnostics / f"gemini31/reports-{label}")
                )
        # A dataset-level archival move subsumes nested attempt-level moves.
        plan = [
            (src, dst)
            for src, dst in plan
            if not any(other != src and other in src.parents for other, _ in plan)
        ]
        destinations = set()
        for src, dst in plan:
            if dst.exists() or dst in destinations:
                raise ValueError(f"Conflitto archivio: {dst}; nessuna modifica.")
            destinations.add(dst)
        # Validate merge conflicts BEFORE any mutation.
        moving = {p for p, _ in plan}
        if legacy31.exists():
            for path in records(legacy31):
                if path.parent in moving:
                    continue
                target = canonical31 / path.parent.relative_to(legacy31)
                if target.exists() and target not in moving:
                    raise ValueError(f"Conflitto directory: {target}")
        if legacy31.exists():
            validate_merge(legacy31, canonical31, moving)
        configuration_history = diagnostics / "configuration-history"
        for name in (C25, C31):
            unused = ROOT / "configs" / name.replace(".yml", "-openai-3.55.0.yml")
            if unused.exists() and (configuration_history / unused.name).exists():
                raise ValueError(f"Conflitto configurazione archiviata: {unused.name}")
        unused_runtime = prefix / "runtime-openai-3.55.0.json"
        if (
            unused_runtime.exists()
            and (configuration_history / unused_runtime.name).exists()
        ):
            raise ValueError("Conflitto runtime 3.55 archiviato.")
        log_moves = []
        for logfile in (ROOT / "logs").glob("*-textfix*"):
            target = logfile.with_name(logfile.name.replace("-textfix", ""))
            if target.exists():
                target = logfile.with_name(
                    logfile.name.replace("-textfix", "-corrected")
                )
            if target.exists():
                raise ValueError(f"Conflitto nome log: {target}")
            log_moves.append((logfile, target))
        # Prove every selected output is actually present before moving folders.
        for folder in (canonical31, legacy31):
            for path in records(folder):
                row = read(path)
                output = (row.get("selected") or {}).get("output_folder")
                if (
                    row.get("status") == "completed"
                    and output
                    and not Path(output).is_dir()
                ):
                    raise ValueError(f"Artefatti mancanti: {output}; nessuna modifica.")
        cfg["campaign_id"] = N31
        cfg["mlzero"]["runtime_manifest"] = str(prefix / "runtime-gemini31.json")
        cfg["local_runtime_identity"] = canonical_hash(fixed)
        # Preserve the original verifier before installing the bounded correction.
        if not already_corrected:
            validator_backup = diagnostics / "validator-before-numeric-fix.py"
            if validator_backup.exists():
                raise ValueError(
                    "Backup validatore gia presente senza marker: controllare la precedente migrazione."
                )
            validator_backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(validator, validator_backup)
        for src, dst in plan:
            move_attempt(src, dst)
        canonical31.mkdir(parents=True, exist_ok=True)
        if legacy31.exists():
            # Merge every surviving artifact, not just CSV/record summaries.
            merge_tree(legacy31, canonical31)
        for path in records(canonical31):
            row = relocate_paths(read(path), legacy31, canonical31)
            row["campaign_id"] = N31
            if row.get("status") == "completed":
                selected = row.get("selected") or {}
                output = selected.get("output_folder")
                if output and not Path(output).is_dir():
                    raise ValueError(f"Output selezionato mancante: {output}")
            write(path, row)
        adapters = prefix / "campaign-adapters"
        adapters.mkdir(exist_ok=True)
        (adapters / "gemini25.py").write_bytes(plain)
        (adapters / "gemini31.py").write_bytes(corrected)
        write(prefix / "runtime-gemini31.json", fixed)
        config_archive = diagnostics / "configuration-history"
        config_archive.mkdir(parents=True, exist_ok=True)
        # Preserve old configuration and provenance, without an active suffix directory.
        if corrected_cfg.exists():
            if not (config_archive / "gemini31-before-layout.yml").exists():
                shutil.copy2(
                    previous_cfg, config_archive / "gemini31-before-layout.yml"
                )
            corrected_cfg.unlink()
        previous_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False))
        if (prefix / "runtime-gemini-textfix.json").exists():
            (prefix / "runtime-gemini-textfix.json").unlink()
        for logfile, target in log_moves:
            logfile.rename(target)
        for name in (C25, C31):
            unused = ROOT / "configs" / name.replace(".yml", "-openai-3.55.0.yml")
            target = config_archive / unused.name
            if unused.exists():
                if target.exists():
                    raise ValueError(f"Archivio configurazione gia presente: {target}")
                unused.rename(target)
        unused_runtime = prefix / "runtime-openai-3.55.0.json"
        if unused_runtime.exists():
            target = config_archive / unused_runtime.name
            if target.exists():
                raise ValueError(f"Archivio runtime gia presente: {target}")
            unused_runtime.rename(target)
        # Reports are derivatives. Regenerate on next launch after path relocation.
        if (canonical31 / "report").exists():
            shutil.rmtree(canonical31 / "report")
        for folder in (canonical25, canonical31):
            parents = set()
            for path in folder.glob("runs/*/rep-*/*"):
                parents.update((path, path.parent, path.parent.parent))
            for p in sorted(parents, key=lambda p: len(p.parts), reverse=True):
                try:
                    p.rmdir()
                except OSError:
                    pass
        validator.write_bytes(corrected_validator.read_bytes())
        write(
            ROOT / "artifacts/campaign-layout.json",
            {
                "campaigns": [N25, N31],
                "numeric_revision": "rmse-float32-compat-v1",
                "validator_sha256": digest(corrected_validator.read_bytes()),
                "note": "Canonical paths; recorded run IDs, source identities and historical logs preserved. Only manifest paths and selected-output paths relocated; no model code or scores changed.",
            },
        )
        for folder in (canonical25, canonical31):
            print(f"{folder.name}: {len(records(folder))} esiti finali conservati")
        print("Migrazione completata. Usa i nomi originali delle configurazioni.")


def validate_merge(source, destination, excluded):
    for src in source.iterdir():
        if any(src == p or p in src.parents for p in excluded):
            continue
        dst = destination / src.name
        if any(dst == p or p in dst.parents for p in excluded):
            continue
        if not dst.exists():
            continue
        if src.is_dir() and dst.is_dir():
            validate_merge(src, dst, excluded)
        elif src.is_file() and dst.is_file():
            if (
                src.name not in ("campaign.json", "README.md")
                and src.read_bytes() != dst.read_bytes()
            ):
                raise ValueError(f"Conflitto file: {dst}")
        else:
            raise ValueError(f"Conflitto di tipo: {dst}")


def merge_tree(source, destination):
    destination.mkdir(parents=True, exist_ok=True)
    for src in list(source.iterdir()):
        dst = destination / src.name
        if src.is_dir() and dst.exists():
            merge_tree(src, dst)
        elif dst.exists():
            if src.is_file() and dst.is_file() and src.read_bytes() == dst.read_bytes():
                src.unlink()
            elif src.name in ("campaign.json", "README.md"):
                # Campaign metadata must follow the actual corrected cohort.
                src.replace(dst)
            else:
                raise ValueError(f"Conflitto file: {dst}")
        else:
            shutil.move(str(src), str(dst))
    source.rmdir()


def launch(configs):
    import yaml
    from ageml_mlzero_benchmark import canonical_hash
    from ageml_mlzero_benchmark.hamlet_cv import doctor, run, report, validate_inputs
    from ageml_mlzero_benchmark.local_runtime import snapshot, verify

    marker = read(ROOT / "artifacts/campaign-layout.json")
    validator = ROOT / "ageml_mlzero_benchmark/cv_export.py"
    if (
        marker.get("validator_sha256")
        and digest(validator.read_bytes()) != marker["validator_sha256"]
    ):
        raise ValueError("Validatore modificato dopo la migrazione.")
    for config in configs:
        cfg = yaml.safe_load(Path(config).read_text())
        model = cfg["llm"]["model"]
        version = {"gemini-2.5-flash": "25", "gemini-3.1-flash-lite": "31"}.get(model)
        if version is None:
            raise ValueError("Launcher canonico limitato ai due modelli del benchmark.")
        with checkout_idle():
            manifest = Path(cfg["mlzero"]["runtime_manifest"])
            expected = read(manifest)
            prefix = Path(expected["prefix"])
            actual = snapshot(prefix)
            patched = copy.deepcopy(actual)
            patched["source_hashes"][REL] = expected["source_hashes"][REL]
            if patched != expected:
                raise ValueError("Altre differenze nel runtime: avvio annullato.")
            adapter = prefix / "campaign-adapters" / f"gemini{version}.py"
            if digest(adapter.read_bytes()) != expected["source_hashes"][REL]:
                raise ValueError("Adattatore registrato non corrisponde al manifest.")
            (prefix / "source" / REL).write_bytes(adapter.read_bytes())
            verify(manifest)
            if cfg.get("local_runtime_identity") != canonical_hash(expected):
                raise ValueError("Identita della configurazione diversa dal runtime.")
            validate_inputs(cfg)
        if not os.environ.get("GEMINI_API_KEY"):
            os.environ["GEMINI_API_KEY"] = getpass.getpass("Gemini API key: ")
        doctor(cfg, True)
        folder = (
            ROOT / cfg["outputs"]["artifact_root"] / "campaigns" / cfg["campaign_id"]
        )
        done = {}
        for path in records(folder):
            row = read(path)
            if not row.get("final", True):
                continue
            key = row["dataset"], row["repetition"]
            if key in done:
                raise ValueError(
                    f"Due esiti finali per {key}; nessuna selezione automatica."
                )
            done[key] = row
        planned = [
            (entry["key"], rep)
            for entry in cfg["dataset_entries"]
            for rep in cfg["repetitions"]
        ]
        if os.environ.get("AGEML_BENCH_PILOT") == "1":
            planned = [("adult", 1)]
        for dataset, rep in planned:
            if (dataset, rep) in done:
                print(f"SKIP {dataset} rep={rep}", flush=True)
                continue
            # Native runner retains its identity/provenance. Frozen result lookup
            # above recognizes pre-migration completions without forging IDs.
            run(cfg, False, [dataset], rep)
        print(report(cfg))


if __name__ == "__main__":
    os.chdir(ROOT)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["organize", "run"])
    parser.add_argument("configs", nargs="*")
    args = parser.parse_args()
    if args.command == "organize":
        organize()
    elif args.configs:
        launch(args.configs)
    else:
        parser.error("Indica almeno una configurazione.")
