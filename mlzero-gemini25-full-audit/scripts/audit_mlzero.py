"""Read-only audit of saved MLZero outputs. Never imports generated model code."""
import argparse
import ast
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

REGRESSION = ("ames_housing", "california_housing", "auto_mpg")
DATASETS = ("adult", "har", "madelon", "mnist_784", "texture", "wilt", *REGRESSION)

FAILURE_NOTES = {
    ("adult", 1): "Only node_0 is saved; generated Python and installation script exist, but no exported factory. The archive cannot distinguish installation delay, initialization delay or another pre-export interruption.",
    ("adult", 4): "Repair sequence alternates Tabular fit(seed=...), constructor random_state, MultiModal problem_type='classification', and set_seed import changes. Evidence of API integration difficulties; the final blocking exception is not archived.",
    ("ames_housing", 3): "18 factory modules, no CV export, native exit after 1888 s. Repeated Tabular constructor random_state / fit(seed=...), set_seed import-path changes and MultiModal seed settings. Strong evidence of an unsuccessful repair sequence, rather than a completed fit with a bad score.",
    ("ames_housing", 4): "9 factory modules, no CV export. Code revisions remove fit(random_seed=...), use medium_quality_faster_inference_only_refit and mention segmentation faults. The latter are generated-code comments, not retained runtime traces. Training/setup/repair consumed the deadline; exact final cause is unresolved.",
    ("auto_mpg", 5): "6 factory modules, no CV export. Revisions replace TabularPredictor.set_seed, change set_random_seed imports and remove fit(seed=...). Integration/repair and fitting are plausible contributors; the final exception is not archived.",
    ("har", 4): "18 factory modules, no CV export, native exit after 1805 s. Repeated constructor random_state, fit(seed=...), MultiModal problem_type='classification' and env.seed changes. Strong evidence of an unsuccessful repair sequence.",
    ("madelon", 3): "2 factory modules, no CV export. Tabular constructor seed and MultiModal constructor/fit settings occur. Substantial CPU time was consumed; the retained files cannot establish whether installation, fit, or dependency failure ultimately prevented export.",
    ("mnist_784", 4): "3 factory modules, no CV export. Tabular fit(random_state=...), constructor seed and MultiModal problem_type='classification' occur. Considerable CPU and about 26 GiB sampled peak memory; neither OOM nor exact final exception is established.",
    ("mnist_784", 5): "5 factory modules, no CV export. MultiModal requests env.num_gpus=1 despite a CPU-only campaign; Tabular random_state settings and set_seed imports are revised. Sampled memory approaches MemoryHigh, but this does not prove OOM or throttling as the cause.",
    ("texture", 1): "6 factory modules, no CV export. Revisions change constructor random_state and set_seed import paths; MultiModal problem_type='classification' also occurs. Integration/repair difficulty is suggested, but the final blocking exception is absent.",
}


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def summarize(values):
    a = np.asarray(values, dtype=float)
    if not len(a):
        return math.nan, math.nan
    half = float(t.ppf(.975, len(a) - 1) * a.std(ddof=1) / math.sqrt(len(a))) if len(a) > 1 else math.nan
    return float(a.mean()), half


def inspect_source(path):
    """AST inventory and review leads, not a proof of runtime compliance."""
    text = path.read_text()
    tree = ast.parse(text)
    constants = {}
    for n in tree.body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            try:
                constants[n.targets[0].id] = ast.literal_eval(n.value)
            except (ValueError, TypeError):
                pass
    def literal(n):
        if isinstance(n, ast.Name) and n.id in constants:
            return constants[n.id]
        try:
            return ast.literal_eval(n)
        except (ValueError, TypeError):
            return ast.unparse(n)
    constructors, fits, imports, risks, io_calls = [], [], [], [], []
    predictor_classes, presets = [], []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imports.append(ast.unparse(node))
        if not isinstance(node, ast.Call):
            continue
        function = ast.unparse(node.func)
        kws = {kw.arg: ast.unparse(kw.value) for kw in node.keywords}
        statement = f"L{node.lineno}: {ast.unparse(node)}"
        if function.endswith(("TabularPredictor", "MultiModalPredictor")):
            predictor_classes.append(function)
            constructors.append(statement)
            if "random_state" in kws or "seed" in kws:
                risks.append(f"L{node.lineno}: predictor-constructor seed/random_state setting; check installed API")
            if kws.get("problem_type") == "'classification'":
                risks.append(f"L{node.lineno}: generic classification problem_type; check binary/multiclass API")
        if function.endswith(".fit"):
            fits.append(statement)
            presets.extend(str(literal(kw.value)) for kw in node.keywords if kw.arg == "presets")
            if "Tabular" in text and any(k in kws for k in ("seed", "random_seed", "random_state")):
                risks.append(f"L{node.lineno}: direct fit seed keyword; check concrete predictor API")
        if function.endswith(("read_csv", "read_parquet", ".load", "load_model", "fetch_openml", "open")):
            io_calls.append(statement)
    return dict(constructor_calls=" | ".join(constructors), fit_calls=" | ".join(fits),
                imports=" | ".join(imports), api_review_leads=" | ".join(risks),
                external_data_or_load_calls=" | ".join(io_calls),
                predictor_classes=" | ".join(predictor_classes), presets=" | ".join(presets),
                log_target_transform_present="log1p" in text and "expm1" in text)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign", required=True, type=Path)
    p.add_argument("--reference-dir", required=True, type=Path)
    p.add_argument("--paper-reference", required=True, type=Path)
    p.add_argument("--outdir", required=True, type=Path)
    args = p.parse_args()
    root, out = args.campaign.resolve(), args.outdir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    references = {ds: load(args.reference_dir / ds / "age_protocol_compact.json") for ds in DATASETS}
    paper = load(args.paper_reference)["performance_references"]
    paired = pd.read_csv(root / "report/paired_cv_scores.csv")
    paired = {(r["dataset"], int(r["repetition"])): r for r in paired.to_dict("records")}
    runs, nodes, winners, checks, failures, regression_winners = [], [], [], [], [], []
    keys = set()
    for record_path in sorted(root.glob("runs/*/rep-*/*/attempt-*/record.json")):
        r = load(record_path)
        if not r.get("final", True):
            continue
        ds, rep = r["dataset"], int(r["repetition"])
        assert (ds, rep) not in keys, f"Duplicate final outcome: {(ds, rep)}"
        keys.add((ds, rep))
        attempt = record_path.parent
        identity = load(attempt / "identity.json")
        entry = next(e for e in identity["config"]["dataset_entries"] if e["key"] == ds)
        reference = references[ds]
        node_paths = sorted((attempt / "mlzero").glob("node_[0-9]*"), key=lambda f: int(f.name[5:]))
        for node in node_paths:
            f = node / "generated_code.py"
            if not f.is_file():
                continue
            info = inspect_source(f)
            nodes.append(dict(dataset=ds, repetition=rep, node=node.name,
                              run_status=r["status"], selected=(r.get("selected") or {}).get("node_id") == int(node.name[5:]),
                              factory_present=(node / "output/model_factory.py").is_file(),
                              cv_export_present=(node / "output/cv_evaluation.json").is_file(),
                              generated_code_sha256=sha(f), code_path=str(f.relative_to(root)), **info))
        n_factories = sum((n / "output/model_factory.py").is_file() for n in node_paths)
        n_exports = sum((n / "output/cv_evaluation.json").is_file() for n in node_paths)
        resource = r.get("resources") or {}
        summary = dict(dataset=ds, repetition=rep, status=r["status"], cv_score=r.get("cv_score"),
                       termination_reason=r.get("termination_reason"), search_wall_seconds=r.get("search_wall_s"),
                       generated_nodes=len(node_paths), exported_factories=n_factories, cv_exports=n_exports,
                       selected_node=(r.get("selected") or {}).get("node_id"),
                       selected_tool_label=(r.get("selected") or {}).get("tool"),
                       sampled_cpu_seconds=resource.get("cpu_s"), sampled_peak_gib=(resource.get("peak_rss_bytes") or 0) / 2**30,
                       resource_sample_final=resource.get("resources_final"),
                       llm_calls=(r.get("usage") or {}).get("llm_calls"),
                       source_sha256=r["source_sha256"], split_identity=r["split_identity"],
                       same_source_as_age=r["source_sha256"] == reference["source_sha256"],
                       same_split_identity_as_age=r["split_identity"] == reference["split_identity"],
                       age_cv_score=paired[(ds, rep)]["age_cv_score"], record_path=str(record_path.relative_to(root)))
        runs.append(summary)
        if r["status"] != "completed":
            failures.append({**summary, "cause_assessment": FAILURE_NOTES[(ds, rep)],
                             "causal_certainty": "Termination recorded; detailed cause inferred or unresolved without runtime logs",
                             "search_log_in_archive": (attempt / "search.log").is_file()})
            continue
        node_id = r["selected"]["node_id"]
        folder = attempt / "mlzero" / f"node_{node_id}" / "output"
        ev = load(folder / "cv_evaluation.json")
        factory = folder / "model_factory.py"
        pred = pd.read_csv(folder / "cv_predictions.csv")
        cf = reference["cv_folds"]
        expected_fold = {int(row): fold for fold, part in enumerate(cf) for row in part["validation"]}
        row_ids = pred["row_id"].to_numpy()
        fold_ids = pred["fold"].to_numpy()
        row_coverage = (len(pred) == reference["source_row_count"] and not pred["row_id"].duplicated().any()
                        and set(row_ids) == set(expected_fold) and all(float(x).is_integer() for x in row_ids))
        fold_match = row_coverage and all(expected_fold[int(row)] == int(fold) and float(fold).is_integer() for row, fold in zip(row_ids, fold_ids))
        values = pd.to_numeric(pred["prediction"], errors="coerce")
        declared = np.asarray(ev["fold_scores"], dtype=float)
        trusted = np.asarray(r["cv_fold_scores"], dtype=float)
        mean_check = len(declared) == len(trusted) == 10 and np.isclose(declared.mean(), ev["cv_score"], rtol=1e-10, atol=1e-10) and np.isclose(trusted.mean(), r["cv_score"], rtol=1e-10, atol=1e-10)
        tol = 1e-6 if ds in REGRESSION else 1e-10
        numeric = bool(np.allclose(declared, trusted, rtol=tol, atol=1e-8 if ds in REGRESSION else 1e-10))
        check = dict(dataset=ds, repetition=rep, factory_sha256_matches=sha(factory) == ev["factory_sha256"],
                     source_sha256_matches=ev["source_sha256"] == r["source_sha256"] == entry["input_hashes"]["data.csv"],
                     evaluator_sha256_matches=ev["evaluator_sha256"] == entry["input_hashes"]["evaluate_cv.py"],
                     contract_sha256_matches=ev["contract_sha256"] == entry["input_hashes"]["cv_contract.json"],
                     oof_rows=len(pred), complete_unique_oof_rows=row_coverage,
                     oof_folds_match_age=fold_match,
                     predictions_nonmissing_or_finite=(bool(np.isfinite(values).all()) if ds in REGRESSION else bool(pred["prediction"].notna().all())),
                     prediction_unique_values=int(pred["prediction"].nunique()),
                     exactly_ten_folds_and_means_consistent=bool(mean_check), saved_scores_numerically_consistent=numeric,
                     max_relative_fold_difference=float(np.max(np.abs(declared-trusted)/np.maximum(np.abs(trusted), 1e-30))),
                     declared_vs_record_mean_difference=float(ev["cv_score"] - r["cv_score"]),
                     target_values_available=False, independent_metric_recomputation=False)
        checks.append(check)
        info = inspect_source(factory)
        winner = {**summary, **check, **info, "factory_path": str(factory.relative_to(root)),
                  "cv_wall_seconds": ev.get("wall_seconds"), "effective_random_states": json.dumps(ev.get("effective_random_states")),
                  "leakage_review": "No obvious external-data reads/model loads in reviewed source; static evidence is not proof of absence",
                  "metric_role": "Selection CV; not an independent test"}
        winners.append(winner)
        if ds in REGRESSION:
            regression_winners.append(winner)

    assert len(runs) == 45 and len(winners) == 35 and len(failures) == 10
    for c in checks:
        failed = [k for k,v in c.items() if k.endswith(("_matches", "_rows", "_age", "_finite", "_consistent")) and isinstance(v, (bool, np.bool_)) and not v]
        assert not failed, (c["dataset"], c["repetition"], failed)

    summaries, comparison = [], []
    for ds in DATASETS:
        rs = [r for r in runs if r["dataset"] == ds]
        success = [r for r in rs if r["status"] == "completed"]
        m, ci = summarize([r["cv_score"] for r in success])
        am, aci = summarize([r["age_cv_score"] for r in rs])
        ap, _ = summarize([r["age_cv_score"] for r in success])
        s = dict(dataset=ds, metric="RMSE" if ds in REGRESSION else "balanced_accuracy", total_runs=len(rs),
                 completed=len(success), no_artifact=len(rs)-len(success), success_fraction=len(success)/len(rs),
                 mlzero_success_only_mean=m, mlzero_success_only_ci95_halfwidth=ci,
                 age_all_five_mean=am, age_all_five_ci95_halfwidth=aci,
                 age_mean_on_mlzero_success_repetitions=ap,
                 generated_nodes=sum(r["generated_nodes"] for r in rs), cv_exports=sum(r["cv_exports"] for r in rs))
        summaries.append(s)
        if ds in REGRESSION:
            comparison.append({**s, "rmse_reduction_percent_vs_age_all_five": 100*(1-m/am),
                               "rmse_reduction_percent_vs_age_matched_success_repetitions": 100*(1-m/ap),
                               "historical_paper_reference": paper[ds]["score"],
                               "historical_reference_source": paper[ds]["source"],
                               "rmse_reduction_percent_vs_historical_reference": 100*(1-m/paper[ds]["score"]),
                               "historical_reference_is_hamlet": False,
                               "historical_protocol_equivalence_verified": False})
    for name, table in (("run_audit.csv", runs), ("node_inventory.csv", nodes), ("winner_audit.csv", winners),
                        ("failure_audit.csv", failures), ("export_checks.csv", checks),
                        ("dataset_summary.csv", summaries), ("regression_comparison.csv", comparison),
                        ("regression_winners.csv", regression_winners)):
        save_csv(out / name, table)
    result = dict(final_runs=len(runs), completed=len(winners), no_artifact=len(failures),
                  failure_termination_counts=dict(Counter(r["termination_reason"] for r in failures)),
                  generated_nodes=len(nodes), cv_exports=sum(r["cv_exports"] for r in runs),
                  failed_run_nodes=sum(r["generated_nodes"] for r in failures),
                  failed_run_cv_exports=sum(r["cv_exports"] for r in failures),
                  checked_winner_exports=len(checks), regressor_winners_reviewed=len(regression_winners),
                  all_winner_oof_folds_match_age=all(c["oof_folds_match_age"] for c in checks),
                  all_recorded_dataset_hashes_match_age=all(r["same_source_as_age"] for r in runs),
                  all_recorded_split_identities_match_age=all(r["same_split_identity_as_age"] for r in runs),
                  maximum_regression_relative_fold_rounding_difference=max(c["max_relative_fold_difference"] for c in checks if c["dataset"] in REGRESSION),
                  all_regression_winners_use_autogluon=all("autogluon" in r["imports"] for r in regression_winners),
                  raw_targets_in_archive=False, runtime_failure_logs_in_archive=False,
                  experiment_reruns_performed=0)
    (out / "verification_summary.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))
    print(pd.DataFrame(comparison)[["dataset", "completed", "mlzero_success_only_mean", "age_all_five_mean", "age_mean_on_mlzero_success_repetitions", "rmse_reduction_percent_vs_age_matched_success_repetitions"]].to_string(index=False))


if __name__ == "__main__":
    main()
