"""Tables, plots, provenance and discrepancy reports from saved records only."""

import json
from pathlib import Path
from typing import Any

import pandas as pd

from . import read_jsonl
from .evaluation import better
from .reference import pair_checks

FAILURE_KINDS = (
    "no_artifact",
    "export_failed",
    "inference_failed",
    "invalid_predictions",
    "protocol_violation",
)


def load_records(campaign: Path) -> list[dict[str, Any]]:
    from .runner import all_attempt_records

    return all_attempt_records(campaign) + read_jsonl(
        campaign / "age" / "records.jsonl"
    )


def terminal(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One record per logical run: the final attempt, else the latest attempt."""
    by_run: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        by_run.setdefault(r["run_id"], []).append(r)
    out = []
    for rs in by_run.values():
        finals = [r for r in rs if (r.get("extra") or {}).get("final", True)]
        out.append((finals or rs)[-1])
    return out


def runs_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    for col in (
        "agent_settings",
        "resource_limits",
        "evidence",
        "protocol_checks",
        "candidates",
        "extra",
    ):
        if col in df:
            df[col] = df[col].map(
                lambda v: (
                    json.dumps(v, sort_keys=True, default=str)
                    if v is not None
                    else None
                )
            )
    return df


def outcome(a: dict[str, Any] | None, b: dict[str, Any] | None) -> str:
    """Preregistered rule: valid beats invalid, two invalid tie, else the metric decides."""
    va = a is not None and a["artifact_status"] == "valid"
    vb = b is not None and b["artifact_status"] == "valid"
    if va and vb:
        assert a is not None and b is not None
        if a["test_score"] == b["test_score"]:
            return "tie"
        return (
            "ageml"
            if better(a["objective"], a["test_score"], b["test_score"])
            else "mlzero"
        )
    if va != vb:
        return "ageml" if va else "mlzero"
    return "tie"


def comparison(
    records: list[dict[str, Any]], required_protocol: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-dataset summary and per-pair eligibility table."""
    recs = terminal(records)
    age = [r for r in recs if r["system"] == "ageml"]
    mlz = [r for r in recs if r["system"] == "mlzero"]
    pairs = []
    for m in mlz:
        a = next(
            (
                r
                for r in age
                if r["dataset_key"] == m["dataset_key"]
                and r["repetition_id"] == m["repetition_id"]
            ),
            None,
        )
        chk: dict[str, Any] = (
            pair_checks(a, m, required_protocol)
            if a
            else {
                "checks": {},
                "predictive_eligible": False,
                "resource_eligible": False,
            }
        )
        pairs.append(
            {
                "dataset": m["dataset_key"],
                "repetition": m["repetition_id"],
                "ageml_run": a and a["run_id"],
                "mlzero_run": m["run_id"],
                "ageml_status": a and a["artifact_status"],
                "mlzero_status": m["artifact_status"],
                "ageml_test": a and a["test_score"],
                "mlzero_test": m["test_score"],
                "paired_diff": (m["test_score"] - a["test_score"])
                if a and a["test_score"] is not None and m["test_score"] is not None
                else None,
                "winner": outcome(a, m),
                "predictive_eligible": chk["predictive_eligible"],
                "resource_eligible": chk["resource_eligible"],
                "failed_checks": ",".join(
                    k for k, v in chk["checks"].items() if v != "pass"
                )
                or None,
                "ageml_comparability": a and a["comparability_status"],
            }
        )
    pairs_df = pd.DataFrame(pairs)

    rows = []
    for key in sorted({r["dataset_key"] for r in recs}):
        for system, group in (("ageml", age), ("mlzero", mlz)):
            g = [r for r in group if r["dataset_key"] == key]
            if not g:
                continue
            valid = [r["test_score"] for r in g if r["artifact_status"] == "valid"]
            s = pd.Series(valid, dtype=float)
            row = {
                "dataset": key,
                "system": system,
                "objective": g[0]["objective"],
                "planned_runs": len(g),
                "valid_runs": len(valid),
                "success_rate": len(valid) / len(g),
                "test_mean": s.mean() if len(s) else None,
                "test_std": s.std(ddof=1) if len(s) > 1 else None,
                "search_wall_s_mean": pd.Series(
                    [r["search_wall_s"] for r in g], dtype=float
                ).mean(),
                "first_valid_s_mean": pd.Series(
                    [r["first_valid_artifact_s"] for r in g], dtype=float
                ).mean(),
                "inference_s_mean": pd.Series(
                    [r["inference_s"] for r in g], dtype=float
                ).mean(),
                "cpu_s_mean": pd.Series([r["cpu_s"] for r in g], dtype=float).mean(),
                "peak_rss_gib_max": pd.Series(
                    [r["peak_rss_bytes"] for r in g], dtype=float
                ).max()
                / 2**30,
                "llm_calls_mean": pd.Series(
                    [r["llm_calls"] for r in g], dtype=float
                ).mean(),
                "input_tokens_mean": pd.Series(
                    [r["input_tokens"] for r in g], dtype=float
                ).mean(),
                "output_tokens_mean": pd.Series(
                    [r["output_tokens"] for r in g], dtype=float
                ).mean(),
                "cost_mean": pd.Series([r["cost"] for r in g], dtype=float).mean(),
                "usage_coverage": ",".join(
                    sorted({str(r["usage_coverage"]) for r in g})
                ),
                "comparability": ",".join(
                    sorted({str(r["comparability_status"]) for r in g})
                ),
            }
            for k in FAILURE_KINDS:
                row[f"n_{k}"] = sum(1 for r in g if r["artifact_status"] == k)
            row["n_timeout"] = sum(1 for r in g if r["termination_reason"] == "timeout")
            if not pairs_df.empty and system == "mlzero":
                p = pairs_df[(pairs_df.dataset == key) & pairs_df.predictive_eligible]
                row["eligible_pairs"] = len(p)
                row["paired_diff_mean"] = p.paired_diff.mean() if len(p) else None
                row["wins_mlzero"] = int((p.winner == "mlzero").sum())
                row["wins_ageml"] = int((p.winner == "ageml").sum())
                row["ties"] = int((p.winner == "tie").sum())
            rows.append(row)
    return pd.DataFrame(rows), pairs_df


def _fmt(v: Any, nd: int = 4) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "--"
    return f"{v:.{nd}g}" if isinstance(v, float) else str(v)


def latex_table(summary: pd.DataFrame, objective: str) -> str:
    s = summary[summary.objective == objective]
    lines = [
        r"\begin{tabular}{llrrl}",
        r"\toprule",
        r"Dataset & System & Valid/Planned & "
        + objective.replace("_", r"\_")
        + r" & Comparability \\",
        r"\midrule",
    ]
    for _, r in s.iterrows():
        score = (
            "--"
            if pd.isna(r.test_mean)
            else (
                f"{r.test_mean:.4g}"
                + ("" if pd.isna(r.test_std) else f" $\\pm$ {r.test_std:.2g}")
            )
        )
        lines.append(
            f"{r.dataset.replace('_', chr(92) + '_')} & {r.system} & "
            f"{r.valid_runs}/{r.planned_runs} & {score} & "
            f"{r.comparability.replace('_', chr(92) + '_')} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


def plots(summary: pd.DataFrame, records: list[dict[str, Any]], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    recs = terminal(records)
    for objective in sorted(summary.objective.dropna().unique()):
        keys = sorted(summary[summary.objective == objective].dataset.unique())
        fig, axes = plt.subplots(
            1, len(keys), figsize=(2.6 * len(keys), 2.8), squeeze=False
        )
        for ax, key in zip(axes[0], keys):
            for x, system in enumerate(("ageml", "mlzero")):
                ys = [
                    r["test_score"]
                    for r in recs
                    if r["dataset_key"] == key
                    and r["system"] == system
                    and r["artifact_status"] == "valid"
                ]
                ax.scatter([x] * len(ys), ys, s=18, alpha=0.8)
            ax.set_xticks([0, 1], ["AGE-ML", "MLZero"])
            ax.set_xlim(-0.6, 1.6)
            ax.set_title(key, fontsize=9)
        axes[0][0].set_ylabel(objective)
        fig.tight_layout()
        for ext in ("pdf", "svg"):
            fig.savefig(out / f"test_{objective}.{ext}")
        plt.close(fig)

    traj = [
        r
        for r in recs
        if r["system"] == "mlzero" and (r.get("extra") or {}).get("trajectory")
    ]
    if traj:
        keys = sorted({r["dataset_key"] for r in traj})
        fig, axes = plt.subplots(
            1, len(keys), figsize=(2.8 * len(keys), 2.6), squeeze=False
        )
        for ax, key in zip(axes[0], keys):
            for r in traj:
                if r["dataset_key"] == key:
                    t = r["extra"]["trajectory"]
                    ax.step(
                        [p[0] / 60 for p in t],
                        [p[1] for p in t],
                        where="post",
                        alpha=0.7,
                    )
            ax.set_title(key, fontsize=9)
            ax.set_xlabel("elapsed (min)")
        axes[0][0].set_ylabel("native incumbent score")
        fig.tight_layout()
        for ext in ("pdf", "svg"):
            fig.savefig(out / f"trajectories.{ext}")
        plt.close(fig)


def write_report(campaign: Path, required_protocol: str) -> Path:
    out = campaign / "report"
    out.mkdir(parents=True, exist_ok=True)
    records = load_records(campaign)
    if not records:
        raise FileNotFoundError(f"no records under {campaign}")
    runs_frame(records).to_csv(out / "runs.csv", index=False)
    summary, pairs = comparison(records, required_protocol)
    summary.to_csv(out / "comparison.csv", index=False)
    pairs.to_csv(out / "pairs.csv", index=False)
    for objective in sorted(summary.objective.dropna().unique()):
        (out / f"table_{objective}.tex").write_text(latex_table(summary, objective))
    plots(summary, records, out)
    lines = [
        "# Provenance report",
        "",
        f"Required protocol: `{required_protocol}`. Rows below are NOT in the "
        "primary matched comparison unless `predictive_eligible` is true.",
        "",
    ]
    if not pairs.empty:
        lines += [
            "| dataset | rep | eligible | failed checks | AGE comparability |",
            "|---|---|---|---|---|",
        ]
        for _, p in pairs.iterrows():
            lines.append(
                f"| {p.dataset} | {p.repetition} | {p.predictive_eligible} | "
                f"{p.failed_checks} | {p.ageml_comparability} |"
            )
    (out / "provenance.md").write_text("\n".join(lines) + "\n")
    return out
