# Protocol

This repository runs **MLZero (AutoGluon Assistant, NodeManager + MCTS, commit
`f228cd7`)** as a baseline against AGE-ML, and imports AGE-ML results. It never
runs, patches or ablates AGE-ML.

## Reference contract

- **Target protocol: `ageml-holdout-v2`.** This is a proposal, not yet upstream. It
  specifies one explicit target and one immutable train/validation/test partition
  (about 64/16/20, seed 42). Fitting uses the training set only. Hyperparameter and
  global selection use the validation set only. The selected fitted model is
  frozen, then scored once on the test set. There is no train+validation refit.
- **Inspected AGE-ML (`86e4af6`, 5.8.0) is `ageml-holdout-v1-test-selected`.**
  `ageml/evaluation/agent.py::get_best_run_across_pipelines` scores the finalist of
  each pipeline on the test set and picks the overall winner from those scores. That
  creates a selection bias. Results produced this way are imported with
  `comparability_status: historical_protocol` and kept out of the primary matched
  table. `ageml-bench reference` detects this automatically.
- MLZero records carry `ageml-holdout-v2` once the reference manifest comes from a
  corrected AGE-ML campaign. Until then they carry `ageml-holdout-v2-provisional`.

## Data

- Datasets are exported exactly as AGE-ML's `experiments/download_datasets.py`
  does: `get_data(dataset_format="dataframe")` with no target, every column in
  OpenML order, written with `to_csv(index=False)`. The export is then read back
  with `pd.read_csv`, as AGE-ML reads it. Row ids are positions in that export.
- **Target:** AGE-ML v1 has an LLM pick the target. The benchmark uses OpenML's
  `default_target_attribute`, or the AGE-ML v2 manifest's target once it exists,
  and never guesses.
- **Split:** the preferred source is an AGE-ML-exported split (`prepare
  --split-dir`). Otherwise the AGE-ML v1 two-stage split is regenerated:
  `train_test_split(test_size=0.2, stratify=y, random_state=42)` for outer and
  inner, with an unstratified fallback on `ValueError`. Calling it on row ids
  reproduces the DataFrame calls exactly (tested). If the reference records a
  split hash, the regenerated split must match it.
- **Search sees only** `train.csv`, `val.csv`, `val_features.csv` and `README.md`,
  which holds the task and data-use rules. The test set, the full export and the
  labels stay in `private/`, which is never mounted into search.

## No-refit and data use

MLZero's README and its short `-t` instruction impose these rules:

- Fit only on `train.csv`. Use `val.csv` labels only for selection: model and
  hyperparameter choice, early stopping, ensemble weights, thresholds and
  calibration.
- Never refit on train+validation. Under AutoGluon this means
  `refit_full=False` and `set_best_to_refit_full=False`.
- Export `predict.py` and record data use in `data_use.json`.

Compliance is **audited, not assumed**:

- A static scan of the selected node's code flags `refit_full`, extra splits,
  train+validation concatenation and test-path access.
  `no_flags_unverified` means nothing was found, not that compliance was proven.
- The declared `data_use.json` is kept.
- During inference, a fit-call guard (`environments/mlzero/guard/sitecustomize.py`)
  records `fit`, `partial_fit`, `fit_transform`, `train` and `refit_full` calls
  from sklearn, AutoGluon, LightGBM, XGBoost, CatBoost and imblearn. Asset hashes
  are compared before and after inference.

## Timing

- The search clock starts **before** the container is launched. It covers
  initialization, retrieval, LLM latency and retries, per-node conda/uv
  environment creation, training, repair, selection and export.
- The deadline (`budget.search_wall_seconds`) applies to the whole container
  through `docker stop`: SIGTERM, then SIGKILL after `stop_grace_seconds`.
  MLZero's `per_execution_timeout` only limits each generated script and is kept
  at its native 28,800 s.
- **Outside the timed region:** the image build, retrieval-model download, tutorial
  index build, and cache warming (`warm_cache.sh`). Caches are shared across runs
  (warm-cache policy). Each environment is built at run time inside the clock and
  its cost is retained.
- Final inference has a separate timeout (`evaluation.inference_timeout_seconds`,
  600 s while piloting). It runs with `--network none`, no credentials, and the
  selected node's preserved `conda_env`.
- AGE-ML's nominal 7,200 s wraps only the pipeline jobs in the inspected code, so
  imported AGE-ML timing has `timing_boundary_verified: false` unless an AGE-ML
  manifest says otherwise. Resource comparisons require that check to pass.

## Native selection

- MLZero's executer LLM extracts the validation score and negates loss metrics.
  MCTS maximizes that value.
- The benchmark patch publishes `incumbent.json` atomically after each
  `create_best_run_copy()`, using the same rule: `_best_node`, otherwise the last
  successful node.
- A run killed mid-copy keeps the previous committed incumbent.
- If a committed selection lacks `predict.py`, the result is `export_failed`. The
  benchmark never falls back to an earlier candidate, never reselects using
  trusted or test scores, and never repairs an artifact.
- Every record stores the native score, the independently recomputed validation
  score, and the discrepancy between them (`native_vs_trusted_validation`).

## Outcomes

| `termination_reason` | meaning |
|---|---|
| `native_completed` | `run_agent` returned (all nodes terminal) |
| `safety_ceiling` | the 10,000-iteration ceiling was hit |
| `timeout` | overall deadline |
| `resource_exhaustion` | container OOM-killed |
| `process_error` | non-zero exit or no `run_agent` return; an outcome, never retried |
| `infrastructure_error` | the container/process failed to launch; retried once |
| `harness_interrupted` | the harness died mid-attempt; partial accounting kept, retried once |

| `artifact_status` | meaning |
|---|---|
| `valid` | test predictions passed the schema checks and were scored |
| `no_artifact` | no committed incumbent |
| `export_failed` | the committed selection has no `predict.py` |
| `inference_failed` | non-zero exit, timeout, or no output |
| `invalid_predictions` | wrong row count, missing or duplicate ids, invalid labels, or non-finite values |
| `protocol_violation` | fit calls detected during inference, or fitted assets modified |

Retries never depend on a score. Every attempt and its charges are kept.

## Metrics and reporting

- Classification uses balanced accuracy with labels compared as exported strings.
  Regression uses RMSE in the target's original units.
- Predictions are normalized to `row_id,prediction` and validated before scoring.
- Reported success rate is valid runs over all planned runs. Quality is reported
  only for valid runs.
- **Preregistered win rule:** a valid result beats an invalid one, two invalid
  results tie, and otherwise the metric decides. Wins count only pairs that pass
  every predictive compatibility check: data hash, split hash, protocol, target,
  objective and model.
- Repetitions are complete search invocations on one fixed split. They describe
  search variability, not sensitivity to the partition.
- Token accounting:
  - `cache_read` and `reasoning` are subsets and are never added to totals.
  - Reasoning counts toward billable output only when the provider's raw
    `total_tokens` shows it was excluded from completion tokens.
  - Missing usage is `null`, never zero.
  - Costs come from the dated `configs/pricing.yml`.
