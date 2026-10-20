"""Stand-in for the `mlzero` CLI, writing what the bench_hooks patch writes.

Behaviour is chosen with FAKE_MLZERO_MODE:
  ok                two successful nodes, the second better; stops at -n
  hang              commits node 1, starts node 2 without committing, never exits
  no_export         node 1 succeeds without predict.py
  native_vs_trusted node 1 has the best NATIVE score but poor predictions
  fit_in_predict    predict.py calls sklearn fit
  crash             exits 3 before any node
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import yaml

PREDICT = """
import argparse, json, csv
from pathlib import Path
p = argparse.ArgumentParser(); p.add_argument("--input"); p.add_argument("--output")
a = p.parse_args()
model = json.loads((Path(__file__).parent / "model.json").read_text())
with open(a.input) as f:
    n = sum(1 for _ in csv.reader(f)) - 1
{extra}
with open(a.output, "w") as f:
    f.write("prediction\\n" + "".join(f"{{model['value']}}\\n" for _ in range(n)))
"""
FIT = "from sklearn.linear_model import LinearRegression; LinearRegression().fit([[0],[1]],[0,1])"


def append(bench, name, obj):
    with open(bench / name, "a") as f:
        f.write(json.dumps({"ts": time.time(), **obj}) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-i")
    p.add_argument("-o")
    p.add_argument("-c")
    p.add_argument("-n", type=int)
    p.add_argument("-t")
    p.add_argument("-v")
    p.add_argument("--continuous_improvement", action="store_true")
    a = p.parse_args()
    mode = os.environ.get("FAKE_MLZERO_MODE", "ok")
    bench = Path(os.environ["AGEML_BENCH_DIR"])
    if mode == "crash":
        sys.exit(3)
    cfg = yaml.safe_load(Path(a.c).read_text())
    (bench / "effective_config.yaml").write_text(yaml.safe_dump(cfg))
    append(
        bench,
        "events.jsonl",
        {
            "event": "effective_config",
            "cli": {
                "max_iterations": a.n,
                "continuous_improvement": a.continuous_improvement,
            },
        },
    )
    for agent in ("python_coder", "executer"):
        s = cfg[agent]
        append(
            bench,
            "events.jsonl",
            {
                "event": "chat_created",
                "session": agent,
                "settings": {
                    "model": s["model"],
                    "base_url": s["proxy_url"],
                    "env_base_url": os.environ.get("OPENAI_BASE_URL"),
                },
            },
        )

    readme = (Path(a.i) / "README.md").read_text()
    target = readme.split("Predict the column `")[1].split("`")[0]
    import pandas as pd

    train = pd.read_csv(Path(a.i) / "train.csv")
    regression = "regression" in readme.splitlines()[0]
    good = train[target].mean() if regression else train[target].mode()[0]
    bad = train[target].max() * 10 if regression else sorted(train[target].unique())[-1]
    if not regression and bad == good:
        bad = sorted(train[target].unique())[0]
    n_val = len(pd.read_csv(Path(a.i) / "val_features.csv"))

    nodes = {"native_vs_trusted": [(1, 0.9, bad), (2, 0.5, good)]}.get(
        mode, [(1, 0.5, good), (2, 0.7, good)]
    )
    best = None
    for nid, native, value in nodes:
        out = Path(a.o) / f"node_{nid}" / "output"
        out.mkdir(parents=True, exist_ok=True)
        (out.parent / "generated_code.py").write_text("# fake\n")
        (out / "model.json").write_text(
            json.dumps({"value": value if regression else str(value)})
        )
        (out / "results.csv").write_text("prediction\n" + f"{value}\n" * n_val)
        if mode != "no_export":
            (out / "predict.py").write_text(
                PREDICT.format(extra=FIT if mode == "fit_in_predict" else "")
            )
        append(
            bench,
            "usage.jsonl",
            {
                "session": "python_coder",
                "reported_model": cfg["llm"]["model"],
                "usage_metadata": {
                    "input_tokens": 100,
                    "output_tokens": 10,
                    "output_token_details": {"reasoning": 4},
                },
                "token_usage": {"total_tokens": 110},
            },
        )
        if mode == "hang" and nid == 2:
            time.sleep(3600)
        append(
            bench,
            "events.jsonl",
            {
                "event": "step",
                "node_id": nid,
                "success": True,
                "validation_score": native,
            },
        )
        if best is None or native > best[1]:
            best = (nid, native)
        rec = {
            "node_id": best[0],
            "native_validation_score": best[1],
            "node_folder": str(Path(a.o) / f"node_{best[0]}"),
            "output_folder": str(Path(a.o) / f"node_{best[0]}" / "output"),
            "export_status": "missing" if mode == "no_export" else "present",
            "selected_by": "best_validation_score",
            "committed_at": time.time(),
        }
        tmp = bench / ".incumbent.tmp"
        tmp.write_text(json.dumps(rec))
        os.replace(tmp, bench / "incumbent.json")
        append(bench, "events.jsonl", {"event": "incumbent_committed", **rec})
    append(
        bench, "events.jsonl", {"event": "search_stopped", "reason": "max_iterations"}
    )
    append(bench, "events.jsonl", {"event": "run_agent_returned"})


if __name__ == "__main__":
    main()
