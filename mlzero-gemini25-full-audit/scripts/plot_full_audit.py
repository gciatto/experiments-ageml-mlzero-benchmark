"""Publication/export plots from immutable result tables; no model execution."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

p = argparse.ArgumentParser()
p.add_argument("--outdir", type=Path, required=True)
a = p.parse_args()
out = a.outdir
runs = pd.read_csv(out / "run_audit.csv")
diagnostics = pd.read_csv(out / "run_diagnostics.csv")
nodes = pd.read_csv(out / "node_diagnostics.csv")
summary = pd.read_csv(out / "dataset_summary.csv")
names = dict(adult="Adult", har="HAR", madelon="Madelon", mnist_784="MNIST", texture="Texture", wilt="Wilt",
             ames_housing="Ames Housing", california_housing="California Housing", auto_mpg="Auto MPG")
datasets = list(names)
teal, blue, orange, purple, red = "#258477", "#326a9e", "#d88728", "#8255a0", "#bd4f56"
grey = "#dce1e7"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.titleweight": "bold",
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 190})


def save(fig, stem):
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{stem}.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)


fig, ax = plt.subplots(figsize=(11.5, 7.7))
color_for = {"Valid CV export": "#e2f0eb", "Timeout with evaluation unfinished": "#f8e4c5",
             "Repair branches exhausted": "#eaddf0", "Timeout during setup": "#f2d7da"}
for i, ds in enumerate(datasets):
    for j, rep in enumerate(range(1, 6)):
        r = runs[(runs.dataset == ds) & (runs.repetition == rep)].iloc[0]
        d = diagnostics[(diagnostics.dataset == ds) & (diagnostics.repetition == rep)].iloc[0]
        ax.add_patch(plt.Rectangle((j-.5, i-.5), 1, 1, color=color_for[d.terminal_phase], ec="white", lw=3))
        if r.status == "completed":
            val = f"{r.cv_score / 1000:.2f}k" if ds.endswith("housing") else f"{r.cv_score:.3f}"
            label = val
        else:
            label = {"Timeout during setup": "No artifact\nsetup timeout",
                     "Timeout with evaluation unfinished": "No artifact\nCV unfinished",
                     "Repair branches exhausted": "No artifact\nrepair limit"}[d.terminal_phase]
        ax.text(j, i, label, ha="center", va="center", fontsize=10)
    count = int(summary.loc[summary.dataset.eq(ds), "completed"].iloc[0])
    ax.text(4.7, i, f"{count}/5", va="center", fontsize=11, fontweight="bold")
ax.text(4.7, -.82, "Valid", fontweight="bold", va="center")
ax.set_xlim(-.5, 5.3)
ax.set_ylim(8.5, -1.1)
ax.set_xticks(range(5), [f"Rep. {n}" for n in range(1, 6)])
ax.xaxis.tick_top()
ax.set_yticks(range(9), [names[d] + ("  [RMSE]" if d in ("ames_housing", "california_housing", "auto_mpg") else "  [bal. acc.]") for d in datasets])
ax.tick_params(length=0, pad=9)
for s in ax.spines.values():
    s.set_visible(False)
ax.axhline(5.5, color="#78818b", lw=1)
fig.suptitle("MLZero + Gemini 2.5 Flash: all 45 final outcomes", x=.53, y=.98, fontsize=16, fontweight="bold")
legend = [Patch(facecolor=color_for[k], label=v) for k, v in (
    ("Valid CV export", "Valid export (35)"), ("Timeout with evaluation unfinished", "CV unfinished at deadline (7)"),
    ("Repair branches exhausted", "Repair branches exhausted (2)"), ("Timeout during setup", "Setup timeout (1)"))]
fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(.54, .025), ncol=2, frameon=False, fontsize=10)
fig.text(.54, .006, "Values are selection-CV scores. A valid export does not imply a strong model. k = 1,000.", ha="center", fontsize=10)
fig.tight_layout(rect=(0, .11, 1, .94))
save(fig, "01_outcomes_and_scores")

fig, axes = plt.subplots(1, 3, figsize=(13.7, 4.8))
for ax, ds in zip(axes, ("ames_housing", "california_housing", "auto_mpg")):
    d = runs[runs.dataset.eq(ds)].sort_values("repetition")
    valid = d[d.status.eq("completed")]
    offsets = (np.arange(5)-2)*.045
    for k, (_, row) in enumerate(d.iterrows()):
        x0, x1 = offsets[k], 1+offsets[k]
        if row.status == "completed":
            ax.plot((x0, x1), (row.age_cv_score, row.cv_score), color="#c3cbd2", lw=1, zorder=1)
        ax.scatter(x0, row.age_cv_score, color=blue, s=46, zorder=3)
        if row.status == "completed":
            ax.scatter(x1, row.cv_score, color=teal, marker="s", s=46, zorder=3)
            ax.annotate(str(int(row.repetition)), (x1, row.cv_score), xytext=(6, 0), textcoords="offset points", fontsize=8, va="center")
    for x, values, color in ((0, d.age_cv_score, blue), (1, valid.cv_score, teal)):
        ax.hlines(values.mean(), x-.19, x+.19, color=color, lw=3)
    ax.set_xlim(-.35, 1.42)
    ax.set_xticks((0, 1), ("AGE-ML\n5/5 exports", f"MLZero\n{len(valid)}/5 exports"))
    ax.set_ylabel("Selected mean 10-fold RMSE ↓")
    ax.set_title(names[ds], pad=12)
    ax.grid(axis="y", color="#e7ebef")
    ax.set_axisbelow(True)
    ax.text(.5, -.21, f"No MLZero artifact: {5-len(valid)}/5", ha="center", transform=ax.transAxes,
            color=orange if len(valid)<5 else "#535a63", fontsize=10)
    if ds != "auto_mpg":
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x/1000:g}k"))
fig.suptitle("Regression: absolute performance and missing results", fontsize=16, fontweight="bold", y=.995)
fig.text(.5, .035, "Each point is one repetition; grey lines join available pairs. Thick bars show success-only means.", ha="center", fontsize=10)
fig.text(.5, .002, "All 35 selected scores were recomputed from original targets and saved OOF predictions; no experiment was rerun.", ha="center", fontsize=10)
fig.tight_layout(rect=(0, .08, 1, .94))
save(fig, "02_regression_comparison")

failed = diagnostics[diagnostics.status.ne("completed")].copy()
failed["order"] = failed.dataset.map({d:i for i,d in enumerate(datasets)})
failed = failed.sort_values(["order", "repetition"])
fig, (ax, bx) = plt.subplots(1, 2, figsize=(14, 6.5), gridspec_kw={"width_ratios": [1.7, 1]})
for i, (_, row) in enumerate(failed.iterrows()):
    if row.terminal_phase == "Repair branches exhausted":
        ax.barh(i, row.search_wall_minutes, color=purple)
        ax.text(row.search_wall_minutes+1, i, "18 candidates", va="center", fontsize=9)
    else:
        before = row.last_llm_response_offset_minutes
        remainder = row.interval_after_last_llm_response_minutes
        ax.barh(i, before, color=grey)
        ax.barh(i, remainder, left=before, color=red if row.terminal_phase == "Timeout during setup" else orange)
        ax.text(before+remainder/2, i, f"{remainder:.1f} min", ha="center", va="center", fontsize=9)
ax.set_yticks(range(len(failed)), [f"{names[r.dataset]} · rep. {int(r.repetition)}" for _, r in failed.iterrows()])
ax.invert_yaxis()
ax.set_xlim(0, 63)
ax.axvline(60, ls="--", color="#434a51", lw=1)
ax.set_xticks(range(0, 61, 10))
ax.set_xlabel("Elapsed minutes (60-minute search deadline)")
ax.set_title("Where the ten unsuccessful runs stopped", pad=15)
failed_nodes = nodes[nodes.run_status.ne("completed")]
groups = [("Library API / imports", "API / import errors", blue),
          ("Environment / dependencies", "Environment / dependencies", blue),
          ("Native runtime backtrace", "Ray runtime backtrace", blue),
          ("No returned diagnostic", "Unfinished at deadline", orange)]
for i, (key, label, color) in enumerate(groups):
    count = int(failed_nodes.error_category.eq(key).sum())
    bx.barh(i, count, color=color)
    bx.text(count+.7, i, str(count), va="center", fontweight="bold")
bx.set_yticks(range(len(groups)), [g[1] for g in groups])
bx.invert_yaxis()
bx.set_xlim(0, 49)
bx.set_xlabel("Candidate nodes within the ten failed runs")
bx.set_title("73 candidates: one category per node", pad=15)
fig.legend(handles=[Patch(facecolor=grey, label="Before last LLM response"),
                    Patch(facecolor=orange, label="Evaluation unfinished"),
                    Patch(facecolor=red, label="Setup unfinished"),
                    Patch(facecolor=purple, label="Repair tree exhausted")],
           loc="lower center", bbox_to_anchor=(.48, .045), ncol=2, frameon=False, fontsize=10)
fig.suptitle("Failure diagnosis from preserved logs and events", fontsize=16, fontweight="bold", y=.99)
fig.text(.5, .005, "Orange/red intervals follow the last recorded LLM response; they are not a measured breakdown of training, import and waiting time.", ha="center", fontsize=9)
fig.tight_layout(rect=(0, .17, 1, .94), w_pad=3)
save(fig, "03_failure_diagnosis")
