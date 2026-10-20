"""ageml-bench command line."""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from . import append_jsonl, read_json, write_json
from . import config as cfgmod
from . import datasets, evaluation, importers, reference, reporting, runner


def _resolved(path: str) -> dict:
    return cfgmod.resolve(cfgmod.load(Path(path)))


def cmd_reference(a: argparse.Namespace) -> int:
    ref = reference.build_reference(
        Path(a.age_source),
        a.campaign_id,
        Path(a.protocol_manifest) if a.protocol_manifest else None,
        Path(a.resources_file) if a.resources_file else None,
        fetch_openml=not a.no_openml,
    )
    write_json(Path(a.output), ref)
    p = ref["protocol"]
    print(
        f"wrote {a.output}: AGE-ML {ref['source']['commit'][:12]}, protocol "
        f"{p['protocol_id']} ({p['comparability_status']})"
    )
    if p["comparability_status"] != "corrected_protocol":
        print(
            "PREREQUISITE: AGE-ML still selects the global winner on the test set; "
            "results are historical until ageml-holdout-v2 lands upstream.",
            file=sys.stderr,
        )
    return 0


def cmd_import_age(a: argparse.Namespace) -> int:
    ref = read_json(Path(a.reference))
    recs = importers.import_results(Path(a.results), a.campaign, ref)
    out = Path(a.artifact_root) / "campaigns" / a.campaign / "age" / "records.jsonl"
    out.unlink(missing_ok=True)  # deterministic re-import, never duplicated rows
    for r in recs:
        append_jsonl(out, r)
    statuses = sorted({r["comparability_status"] for r in recs})
    print(f"imported {len(recs)} AGE-ML rows -> {out} ({', '.join(statuses)})")
    return 0


def cmd_prepare(a: argparse.Namespace) -> int:
    r = _resolved(a.config)
    root = Path(r["outputs"]["artifact_root"])
    for entry in r["dataset_entries"]:
        if a.only and entry["key"] not in a.only:
            continue
        split = Path(a.split_dir) / f"{entry['key']}.json" if a.split_dir else None
        export = None
        if a.export_dir:
            export = next(Path(a.export_dir).rglob(f"{entry['key']}.csv"), None)
            if export is None:
                raise FileNotFoundError(f"{entry['key']}.csv not under {a.export_dir}")
        out = datasets.prepare_dataset(
            entry, root, split if split and split.exists() else None, export
        )
        m = read_json(out / "manifest.json")
        print(f"{entry['key']}: {m['counts']} split={m['split_hash'][:12]} -> {out}")
    return 0


def cmd_doctor(a: argparse.Namespace) -> int:
    checks: list[tuple[str, bool, str]] = []
    try:
        r = _resolved(a.config)
        checks.append(("config resolves", True, r["campaign_id"]))
    except (cfgmod.ConfigError, FileNotFoundError) as e:
        print(f"FAIL config: {e}")
        return 1
    corrected = cfgmod.is_corrected(r)
    checks.append(
        (
            "reference protocol is " + r["required_protocol_id"],
            corrected,
            r["reference"]["protocol"].get("protocol_id") or "unknown",
        )
    )
    root = Path(r["outputs"]["artifact_root"])
    for ent in r["dataset_entries"]:
        try:
            d = datasets.find_prepared(root, ent["key"])
            m = read_json(d / "manifest.json")
            ok = m["target"] == ent["target"] and (
                not ent.get("split_hash") or ent["split_hash"] == m["split_hash"]
            )
            datasets.assert_search_isolated(d / "search")
            checks.append((f"prepared {ent['key']}", ok, m["split_hash"][:12]))
        except (FileNotFoundError, RuntimeError) as err:
            checks.append((f"prepared {ent['key']}", False, str(err)))
    m = r["mlzero"]
    if m["executor"] == "docker":
        has_docker = shutil.which("docker") is not None
        img = (
            has_docker
            and subprocess.run(
                ["docker", "image", "inspect", m["image"]], capture_output=True
            ).returncode
            == 0
        )
        checks.append(("docker image present", bool(img), m["image"]))
    checks.append(
        (
            f"${r['llm']['api_key_env']} set",
            bool(os.environ.get(r["llm"]["api_key_env"])),
            "presence only",
        )
    )
    if a.check_llm:
        from .providers import check_llm

        res = check_llm(r["llm"])
        checks.append((f"live model {r['llm']['model']}", res["ok"], str(res)))
    for name, ok, detail in checks:
        print(f"{'ok  ' if ok else 'FAIL'} {name}: {detail}")
    # an uncorrected reference only blocks comparative publication, not baseline runs
    blocking = [
        c for c in checks if not c[1] and not c[0].startswith("reference protocol")
    ]
    return 1 if blocking else 0


def cmd_run(a: argparse.Namespace) -> int:
    r = _resolved(a.config)
    if a.dry_run:
        df = runner.dry_run(r)
        if a.only:
            df = df[df.dataset.isin(a.only)]
        print(df.to_string(index=False))
        hours = (
            len(df)
            * (
                r["budget"]["search_wall_seconds"]
                + r["evaluation"]["inference_timeout_seconds"]
            )
            / 3600
        )
        print(
            f"\n{len(df)} invocations, upper allocation {hours:.1f} h "
            f"(search {r['budget']['search_wall_seconds']} s + inference "
            f"{r['evaluation']['inference_timeout_seconds']} s each)"
        )
        if not cfgmod.is_corrected(r):
            print(
                "note: reference protocol is not "
                + r["required_protocol_id"]
                + "; MLZero records will be marked provisional_protocol"
            )
        return 0
    recs = runner.run_campaign(r, resume=a.resume, only=a.only)
    for rec in recs:
        print(
            f"{rec['dataset_key']} rep {rec['repetition_id']}: {rec['artifact_status']} "
            f"({rec['termination_reason']}) test={rec['test_score']}"
        )
    return 0


def cmd_evaluate(a: argparse.Namespace) -> int:
    """Re-score saved normalized predictions; no model execution."""
    root = Path(a.artifact_root)
    bad = 0
    for rec in runner.all_attempt_records(root / "campaigns" / a.campaign):
        pred_rel = (rec.get("evidence") or {}).get("predictions")
        if not pred_rel or rec["artifact_status"] != "valid":
            continue
        prepared = datasets.find_prepared(root, rec["dataset_key"])
        m = read_json(prepared / "manifest.json")
        if m["split_hash"] != rec["split_hash"]:
            print(f"SKIP {rec['attempt_id']}: prepared split differs from the record's")
            continue
        import pandas as pd

        pred = pd.read_csv(
            root / pred_rel, dtype={"prediction": str}, keep_default_na=False
        )
        truth = datasets.load_labels(prepared / "private" / "test_labels.csv")
        ev = evaluation.evaluate(
            pred,
            truth,
            rec["task"],
            rec["objective"],
            list(m["class_mapping"]) if m["class_mapping"] else None,
        )
        same = ev["valid"] and abs(ev["score"] - rec["test_score"]) < 1e-12
        bad += not same
        print(
            f"{'ok  ' if same else 'DIFF'} {rec['attempt_id']}: {ev['score']} (record {rec['test_score']})"
        )
    return 1 if bad else 0


def cmd_report(a: argparse.Namespace) -> int:
    out = reporting.write_report(
        Path(a.artifact_root) / "campaigns" / a.campaign, a.required_protocol
    )
    print(f"report written to {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="ageml-bench", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser(
        "reference", help="capture an AGE-ML reference manifest (never runs AGE-ML)"
    )
    s.add_argument("--age-source", required=True)
    s.add_argument("--output", required=True)
    s.add_argument("--campaign-id", default="ageml-general")
    s.add_argument(
        "--protocol-manifest",
        help="AGE-ML-exported protocol manifest (ageml-holdout-v2)",
    )
    s.add_argument(
        "--resources-file", help="YAML with the AGE-ML campaign's measured resources"
    )
    s.add_argument(
        "--no-openml",
        action="store_true",
        help="skip OpenML metadata (targets stay null)",
    )
    s.set_defaults(fn=cmd_reference)

    s = sub.add_parser("import-age", help="import AGE-ML results.csv files")
    s.add_argument("--reference", required=True)
    s.add_argument("--results", required=True)
    s.add_argument("--campaign", default="ageml-mlzero-main")
    s.add_argument("--artifact-root", default="artifacts")
    s.set_defaults(fn=cmd_import_age)

    s = sub.add_parser("prepare", help="prepare datasets and immutable split manifests")
    s.add_argument("--config", required=True)
    s.add_argument("--only", nargs="*")
    s.add_argument(
        "--split-dir", help="directory of AGE-ML-exported <key>.json split manifests"
    )
    s.add_argument(
        "--export-dir",
        help="reuse AGE-ML exported <key>.csv files instead of downloading",
    )
    s.set_defaults(fn=cmd_prepare)

    s = sub.add_parser("doctor", help="check prerequisites")
    s.add_argument("--config", required=True)
    g = s.add_mutually_exclusive_group()
    g.add_argument("--offline", action="store_true", help="no network calls (default)")
    g.add_argument(
        "--check-llm", action="store_true", help="live model availability check"
    )
    s.set_defaults(fn=cmd_doctor)

    s = sub.add_parser("run", help="run MLZero searches")
    s.add_argument("--config", required=True)
    s.add_argument("--resume", action="store_true")
    s.add_argument("--dry-run", action="store_true")
    s.add_argument("--only", nargs="*", help="dataset keys")
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser("evaluate", help="re-score saved predictions offline")
    s.add_argument("--campaign", required=True)
    s.add_argument("--artifact-root", default="artifacts")
    s.set_defaults(fn=cmd_evaluate)

    s = sub.add_parser("report", help="tables, plots and provenance from saved records")
    s.add_argument("--campaign", required=True)
    s.add_argument("--artifact-root", default="artifacts")
    s.add_argument("--required-protocol", default="ageml-holdout-v2")
    s.set_defaults(fn=cmd_report)

    a = p.parse_args(argv)
    try:
        return a.fn(a)
    except (cfgmod.ConfigError, FileNotFoundError, RuntimeError) as e:
        print(f"ageml-bench {a.cmd}: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
