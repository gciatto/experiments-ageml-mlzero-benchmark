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
