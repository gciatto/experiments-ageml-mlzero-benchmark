# experiments-ageml-mlzero-benchmark

This repository holds the MLZero (AutoGluon Assistant) baseline experiments for the
AGE-ML paper revision. It runs MLZero on AGE-ML's datasets, splits and resource
settings, imports AGE-ML's results, and produces reproducible comparisons.
AGE-ML runs and ablations live in the AGE-ML repository.

- Package: `ageml_mlzero_benchmark`
- Command: `ageml-bench` (`reference`, `import-age`, `prepare`, `doctor`, `run`,
  `evaluate`, `report`)
- [docs/protocol.md](docs/protocol.md): splits, the no-refit rule, timing,
  native selection, outcome definitions and metrics
- [docs/reproduction.md](docs/reproduction.md): exact procedure and the rationale
  for the MLZero patch
- [docs/manuscript-discrepancies.md](docs/manuscript-discrepancies.md): facts the
  paper must update
- [upstreams.lock.yml](upstreams.lock.yml): pinned AGE-ML and MLZero commits,
  patch hash and container inputs

## Status

**Upstream prerequisite.** AGE-ML `86e4af6` (5.8.0) still chooses the global
pipeline winner by test score. Its results import as `historical_protocol`, and
MLZero runs are labelled `ageml-holdout-v2-provisional` until a corrected AGE-ML
campaign, with its manifest and measured resources, is captured with
`ageml-bench reference`. Baseline development and pilots do not need to wait for
it.

## Running the benchmark

You need Linux with Docker (cgroup v2), Python 3.12+ with Poetry, and a Gemini
API key in `$GEMINI_API_KEY`. Reporting and offline evaluation need neither
Docker nor the key. Full details are in
[docs/reproduction.md](docs/reproduction.md).

```bash
# 1. orchestration environment
pip install -r requirements.txt && poetry install

# 2. MLZero image and warm caches (once; outside every timed region)
docker build -f environments/mlzero/Dockerfile -t ageml-mlzero:f228cd7 .
mkdir -p artifacts/cache
docker run --rm -v "$PWD/artifacts/cache:/bench/cache" \
  -e UV_CACHE_DIR=/bench/cache/uv -e CONDA_PKGS_DIRS=/bench/cache/conda \
  ageml-mlzero:f228cd7 bash /opt/bench/warm_cache.sh

# 3. datasets and immutable splits (downloads from OpenML)
poetry run ageml-bench prepare --config configs/pilot.yml

# 4. prerequisites, then the plan (no credentials or execution needed)
poetry run ageml-bench doctor --config configs/pilot.yml --check-llm
poetry run ageml-bench run --config configs/pilot.yml --dry-run

# 5. pilot: Wilt, Auto MPG and Adult, one repetition, 10-minute search each
poetry run ageml-bench run --config configs/pilot.yml

# 6. AGE-ML results, offline re-scoring, report
poetry run ageml-bench import-age --reference references/age-ml/reference.json \
  --results <age-ml>/experiments/results/<model>/specification/general \
  --campaign ageml-mlzero-pilot
poetry run ageml-bench evaluate --campaign ageml-mlzero-pilot
poetry run ageml-bench report   --campaign ageml-mlzero-pilot
```

**Full campaign** (`configs/comparison.yml`): 9 datasets x 5 repetitions = 45
MLZero invocations, each with a 2 h search deadline and a 600 s inference
allowance. That is at most about 98 h of wall time, run sequentially. It resolves
only once `references/age-ml/reference.json` has been regenerated from a
corrected AGE-ML campaign with measured resources (see Status). Then:

```bash
poetry run ageml-bench prepare --config configs/comparison.yml
poetry run ageml-bench run --config configs/comparison.yml --resume
```

`--resume` makes reruns safe:

- Completed runs are skipped.
- An attempt the harness never finished is closed with its partial usage and
  then restarted.
- Changing the model, budget, data, split or MLZero commit produces new run
  identities instead of overwriting old ones.

## What to expect

Each search runs for up to its deadline, which is usually the full budget,
because continuous improvement is on and the 10,000-iteration ceiling is only a
safety net. A pilot invocation therefore takes about 10 minutes plus inference.

**Per run**, under
`artifacts/campaigns/<campaign>/runs/<dataset>/rep-<r>/<run_id>/attempt-<n>/`:

- `record.json`: the authoritative result. It holds:
  - `artifact_status`: `valid`, `no_artifact`, `export_failed`,
    `inference_failed`, `invalid_predictions` or `protocol_violation`
  - `termination_reason`: `timeout` is the usual one; the others are
    `native_completed`, `safety_ceiling`, `resource_exhaustion`,
    `process_error`, `infrastructure_error` and `harness_interrupted`
  - `test_score`: balanced accuracy, or RMSE in the target's original units,
    filled only when `valid`
  - MLZero's native validation score next to the independently recomputed one
  - search and inference time, CPU time, peak memory, LLM calls, tokens and
    estimated cost
  - protocol checks: per-agent model and endpoint, generated-code audit flags,
    and fit calls during inference
- `bench/`: the incrementally written `events.jsonl` and `usage.jsonl`, the
  `effective_config.yaml` and the committed `incumbent.json`. These survive a
  hard kill.
- `mlzero/`: every MCTS node, with its generated code, conda environment and
  outputs, kept so the selected model can be re-run.
- `inference/out/`: the native `predictions.csv` and its normalized
  `row_id,prediction` copy.

**Report**, under `artifacts/campaigns/<campaign>/report/`:

| File | Content |
|---|---|
| `runs.csv` | every attempt, every record field |
| `comparison.csv` | per dataset and system: valid/planned runs, success rate, test mean ± std over valid runs, failure counts by kind, time, CPU, memory, tokens, cost; for MLZero, the eligible paired difference and win/loss/tie counts |
| `pairs.csv` | each MLZero run matched with the AGE-ML run of the same dataset and repetition, with the failed compatibility checks |
| `table_balanced_accuracy.tex`, `table_rmse.tex` | LaTeX tables for the paper |
| `test_*.pdf/svg`, `trajectories.pdf/svg` | per-dataset test scores; best native incumbent score against elapsed time |
| `provenance.md` | which pairs are in the primary matched comparison and why the others are not |

Failed runs stay in the denominators. A valid result beats an invalid one and
two invalid results tie. Classification and regression are summarized
separately.

**With the current AGE-ML results**, the 45 historical rows at `86e4af6` import
with `historical_protocol`, and MLZero runs are `provisional_protocol`. Every
pair therefore fails the protocol check, `provenance.md` lists them as excluded,
and there are no eligible paired differences or wins. The per-system columns of
`comparison.csv` are still filled. This is intended: matched numbers appear only
after the corrected AGE-ML campaign has been captured with `ageml-bench
reference`.

## Layout

| Path | Responsibility |
|---|---|
| `ageml_mlzero_benchmark/reference.py` | AGE-ML reference capture (AST/YAML/git only) and pair eligibility |
| `ageml_mlzero_benchmark/datasets.py` | OpenML export as AGE-ML does it, immutable splits, isolated search input |
| `ageml_mlzero_benchmark/mlzero.py` | MLZero config/task rendering, CLI and inference commands, artifact collection |
| `ageml_mlzero_benchmark/runner.py` | Deadline supervision, final inference, records, resume/retries |
| `ageml_mlzero_benchmark/evaluation.py` | Trusted prediction checks and metrics |
| `ageml_mlzero_benchmark/accounting.py` | Usage, cost and resources |
| `ageml_mlzero_benchmark/importers.py` | AGE-ML `results.csv` and manifest import |
| `ageml_mlzero_benchmark/reporting.py` | CSV, LaTeX, plots and provenance |
| `ageml_mlzero_benchmark/providers.py` | Live Gemini endpoint check (`doctor --check-llm`) |
| `environments/mlzero/` | Pinned container, in-container resource supervisor, inference fit guard |
| `patches/` | MLZero instrumentation and Gemini-compatibility patch |
| `configs/` | Campaign, pilot, pricing, and the upstream MLZero default config |
| `references/age-ml/reference.json` | Reference captured from AGE-ML `86e4af6` |

## Development

```bash
poetry install
poetry run poe test           # offline; live tests are marked `live`
poetry run poe static-checks  # ruff + mypy
```
