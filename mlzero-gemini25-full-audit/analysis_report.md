# MLZero + Gemini 2.5 Flash — definitive campaign audit with original data and logs

20 November 2026. This report supersedes the diagnostic uncertainty in the earlier, smaller-archive audit. No experiment was rerun, no generated model code was imported, and no final outcome was removed or changed.

## Main findings

- All nine original datasets, CV contracts, evaluator scripts and task READMEs match the hashes recorded in the frozen invocation identities. Hashes were computed on decompressed original bytes, not gzip container bytes.
- All 89 saved candidate CV scores were independently recomputed from the original targets and saved OOF predictions. Their numerical values agree with the evaluator exports within the original float32/float64 compatibility tolerance. The recomputed values for all 35 selected winners match the final trusted record scores exactly in this analysis environment.
- Every selected winner is also the best-scoring saved complete candidate within its run. There is no evidence here of a score-sign or incumbent-selection bug explaining the results.
- The final outcomes remain 35 valid exports and ten no-artifact results. The terminal phases are more informative than the generic `no_artifact` status: seven time out with the last evaluation unfinished, two stop because all repair branches are terminal, and one times out during setup.
- Regression gains on Ames and California remain plausible. The regression winners use AutoGluon Tabular/MultiModal; the corresponding AGE-ML winners are simpler Ridge, linear-regression, KNN or SVR pipelines. The data do not isolate the contribution of the LLM from the AutoGluon tool stack.
- Some technically valid classification exports have genuinely weak predictions. The low HAR/MNIST scores are not an aggregation mistake: several selected models predict an almost constant label within each fold.

## What was added to the evidence

The new `artifacts(1).zip` contains the canonical Gemini 2.5 campaign plus original dataset inputs under `artifacts/datasets-cv`. It includes search logs, native events, token accounting, resources, generated Bash/Python, node states, raw setup output, error-analyzer prompts and candidate OOF exports. Compressed files were read transparently without changing original experiment records.

The audit uses the earlier uploaded AGE-ML compliance archive as an independent source of its prescribed folds and baseline winning implementations, and the current `paired_cv_scores.csv` for baseline scores. All 45 dataset/repetition pairs remain represented. The inventory contains 335 generated nodes and 89 complete candidate CV exports.

The new archive resolves two major earlier limitations: labels are now available for independent metric recomputation, and runtime evidence is available for the failed searches. It still does not contain a complete per-fold progress trace, all generated environments, AutoGluon predictor binaries or leaderboards for selected winners.

## Figure 1 — outcomes and scores

`01_outcomes_and_scores.png/pdf` shows all 45 outcomes as a dataset-by-repetition matrix. Successful cells show balanced accuracy for classification or RMSE for regression. Green means that a complete, valid export exists; it is deliberately not a color scale for predictive quality. Missing outcomes have separate labels for unfinished evaluation, repair exhaustion and unfinished setup.

This distinction matters on HAR and MNIST. Their weak scores must not be merged into the no-artifact category, nor should technical export success be described as reliable predictive success.

Selected weak-classification examples:

| Dataset | Rep. | Balanced accuracy | Observed prediction pattern |
|---|---:|---:|---|
| HAR | 1 | 0.166760 | One label accounts for 99.9223% of all OOF predictions; 8/10 folds predict one class exclusively. |
| HAR | 5 | 0.182211 | Only three labels are predicted; one accounts for 81.3963% of all OOF predictions. |
| MNIST | 1 | 0.098361 | Four folds predict a single class exclusively; several other folds are almost constant. |
| MNIST | 2 | 0.100929 | Seven folds predict a single class exclusively. |

The independent target-based metric recomputation confirms these scores. They are compatible with collapsed/poorly learned models, not evidence that a decimal point was misplaced. The saved evidence does not identify the exact optimization dynamics that caused collapse. The final winners are nevertheless the best of the candidates with complete exports in those runs.

## Figure 2 — regression performance

`02_regression_comparison.png/pdf` plots each available repetition rather than hiding variability inside a mean. Grey lines join matching AGE-ML/MLZero repetitions. Thick horizontal bars show each method's mean over its own successful exports. Failed repetitions are stated explicitly and are not imputed as scores.

| Dataset | MLZero success | MLZero mean RMSE | AGE-ML all-five mean | AGE-ML mean on matching successful MLZero repetitions | Reduction vs matching AGE-ML |
|---|---:|---:|---:|---:|---:|
| Ames Housing | 3/5 | 26,743.813 | 33,040.923 | 31,687.846 | 15.60% |
| California Housing | 5/5 | 60,168.034 | 70,644.079 | 70,644.079 | 14.83% |
| Auto MPG | 4/5 | 2.656808 | 2.614751 | 2.614751 | -1.61% |

Ames is the case where the success denominator materially changes the apparent gain. Comparing three successful MLZero scores to all five AGE-ML scores gives a 19.06% reduction; matching repetitions gives 15.60%. Neither removes the two MLZero failures from the experimental outcome. California has five successes, so its gain cannot be explained by dropping failed runs on that dataset. Auto MPG is slightly worse on average despite four successful exports.

The descriptive Student-t 95% half-widths across successful repetitions are approximately 4,654.79 for Ames, 4,980.91 for California and 0.13044 for Auto MPG. These are small-sample summaries of repeated searches on shared data/folds and requested seeds. They are not independent-test confidence intervals or proof of a significant generalization advantage.

### Why the regression gains are credible

All twelve regression-winning factories delegate learning to AutoGluon, including several MLZero candidates labelled `machine learning` rather than `autogluon.tabular`. The actual source is more informative than the tool name.

For Ames, the selected models are AutoGluon Tabular with `best_quality` (rep. 1), MultiModal (rep. 2), and Tabular with `extreme` (rep. 5). The first applies `log1p` to training targets and `expm1` to predictions, so scoring still uses original price units. The corresponding AGE-ML winners are all Ridge-based pipelines.

For California, the selected models are MultiModal (reps. 1 and 3), Tabular with `extreme` (2 and 4), and Tabular with `best` (5). AGE-ML's selected implementations use KNN with PCA, linear regression or Ridge. A full AutoML tool can train/select/combine several internal models inside a single outer candidate; that is a concrete capacity/search-space difference. A requested preset alone does not prove which internal models actually ran or survived the CPU/time limits.

This supports a result about the complete MLZero-plus-AutoGluon stack. It does not isolate whether the advantage arose from the LLM's reasoning, model-family access, internal ensembling, target transformation, different training allocation or their combination.

The historical regression references in the supplied paper are Kaggle notebooks, not HAMLET measurements: Ames 29,660.363 (Gus Martins), California 69,838.844 (Ahmed Mahmoud), and Auto MPG 2.508 (Ertugrul Demir). Conditional improvements against the first two are 9.83% and 13.85%. Exact historical protocol equivalence is not established. The actual shared input here has 1,460 Ames rows and nine California feature columns, whereas the paper's descriptive table lists 2,930 Ames rows and eight California features. Current AGE-ML/MLZero source hashes match; matching a historical dataset name alone is weaker evidence.

## Figure 3 — why ten runs produced no selectable artifact

`03_failure_diagnosis.png/pdf` separates run-level terminal mechanisms from candidate-level errors. These are different counting units.

| Terminal phase | Runs | Which runs |
|---|---:|---|
| Timeout with final evaluation unfinished | 7 | Adult 4; Madelon 3; MNIST 4 and 5; Texture 1; Ames 4; Auto MPG 5 |
| Repair branches exhausted before one hour | 2 | HAR 4; Ames 3 |
| Timeout during initial setup | 1 | Adult 1 |

### Two repair-exhaustion cases are now confirmed

For HAR repetition 4 and Ames repetition 3, `bench/events.jsonl` records `search_stopped` with `reason=all_nodes_terminal`. The search logs also state that all nodes are terminal and include maximum-debug-depth messages. Both searches explored 18 nodes without an incumbent. HAR stops after about 30.09 minutes; Ames after 31.47 minutes. Unused wall-time remains because the search tree has no non-terminal branch to expand, not because the final score table lost a selected artifact.

Returned candidate diagnostics classify as follows:

- HAR 4: eleven library API/import errors and seven environment/dependency errors.
- Ames 3: twelve library API/import errors and six environment/dependency errors.

Examples are explicit exceptions, rather than inferred comments: invalid `random_state` constructor arguments, unsupported direct `.fit(seed=...)`, imports of nonexistent `set_seed` functions, invalid preset names, unknown `env.seed` configuration keys, and missing `numpy` in the interpreter executing the evaluator.

### Seven final evaluations are interrupted before producing a complete export

For seven timeout runs, the final factory exists, setup output has returned, the final setup-review LLM response is recorded, but there is no completed node step or CV export before the search deadline. The interval after that last reply is about 46–54 minutes in six runs and 17 minutes in Ames repetition 4. Earlier failed candidates consumed much more of Ames 4's budget.

The trusted evaluator writes its complete prediction/score export only after all ten folds finish. A run that trains on several folds but fails to finish the full CV therefore remains `no_artifact`. The evidence does not justify treating these as ten cases where no model was ever trained.

The orange intervals in the chart measure time after the last recorded LLM response. They include evaluation startup, imports, training, prediction and any waiting; they are not an instrumented measurement of pure fit time. Logs do not reveal exactly which fold or operation was active at the deadline. Substantial aggregate CPU use supports computation occurring during several of these intervals, but cannot rule out a partial hang.

### One setup-stage timeout

Adult repetition 1 saves only the initial node and its generated installation/model code. The last log message starts code execution; there is no returned setup output, exported factory or CV result. Around 57 minutes follow the last code-generation LLM reply. This localizes the interruption to setup/pre-export execution, but does not identify the exact installation or subprocess operation still active.

### Candidate-level counts within the ten failed runs

| Primary observed category | Candidate nodes |
|---|---:|
| Library API / import errors | 43 |
| Environment / dependency failures | 21 |
| Native Ray runtime backtrace | 1 |
| No returned diagnostic before the deadline | 8 |
| Total | 73 |

These are mutually exclusive primary categories per candidate, not counts of independent exception lines, specification violations or failed full experiments. The eight unreturned nodes are the final node of each timeout run. The Ray trace is preserved in a truncated error-analyzer input for Ames 4 node 3; the retained fragment does not establish its precise triggering native exception or signal.

Concrete environment failures include hard-coded `/opt/conda` paths although this local runtime uses a repository-local Conda installation, failed activation, and evaluators running in environments without NumPy or AutoGluon. These are execution failures of this configured local stack; they should not be generalized to an intrinsic universal failure rate of every MLZero installation.

The failed-run search logs do not show Gemini HTTP 4xx/5xx responses. The diagnoses above are supported by execution state and returned errors, not by assuming a key/quota failure.

## Limits that still apply

- Recomputing an OOF metric verifies the saved predictions and labels, not the entire training process. The inspected regression-winning wrappers show fold-local fits and no obvious full-data loading or prior fitted-model reuse. This is still not a dynamic proof against every possible leakage path inside dependencies or pretrained weights.
- These are selected CV scores, not independent-test scores. Adaptive reuse of folds can create selection optimism even with correct fold-local fitting.
- Wrapper `effective_random_states` entries are empty dictionaries. This does not certify that every internal AutoGluon estimator received seed 42; some wrappers use global seeds or library-specific settings/defaults.
- Exact selected ensemble composition and complete historical generated-environment package states cannot be reconstructed from the wrapper and partial package logs alone.
- LLM-written `error_summary` text is not an authoritative runtime diagnosis. Some summaries describe ordinary dependency versions as suspicious or mistake setup-only logs for failed training. The audit uses explicit tracebacks/events when available and does not relabel those LLM opinions as confirmed root causes.
- No OOM kill is established by these records. High sampled memory in MNIST 5 is not sufficient evidence to replace its recorded timeout with an OOM diagnosis.

## Reporting policy for the final experiments

Keep all 45 outcomes: 35 valid selected exports (77.78%) and ten no-artifact outcomes (22.22%). Report predictive performance conditional on success together with artifact success counts. Do not discard failures, retry until success or promote an unselected candidate retrospectively. No metric was imputed for a missing result in this audit.

The final interpretation is a strong conditional regression result on Ames/California, alongside execution fragility and occasional near-chance classification predictions. Artifact success, predictive quality and runtime reliability are distinct findings, all visible in the supplied figures.

## Deliverables and reproduction

- `verified_input_files.csv`: original-byte hash verification for nine datasets and their frozen CV files.
- `recomputed_candidate_metrics.csv`: all 89 independently recomputed candidate metrics, selected flags and classification-collapse indicators.
- `run_diagnostics.csv`: all 45 final phases, timings, candidate counts and winner-selection checks.
- `node_diagnostics.csv`: all 335 candidates, observed errors and evidence paths.
- `failure_audit.csv`: ten final failures with observed error counts/excerpts and interpretation limits.
- `dataset_summary.csv`, `regression_comparison.csv`, `regression_winners.csv`: performance and matching-repetition comparisons.
- `verification_summary.json`, `full_verification_summary.json`: integrity and recomputation results.
- `01_outcomes_and_scores`, `02_regression_comparison`, `03_failure_diagnosis`: PNG and vector PDF figures.
- `scripts/`: read-only audit and plotting code. No generated estimator module is imported.

The input archive and raw dataset contents are not duplicated in this analysis bundle. Extract the uploaded archive separately, then use the commands in `README.md`. The compact AGE-ML protocol references are included for the original-vs-MLZero fold comparison. Paths in diagnostic CSVs are relative to `artifacts/`; paths in the earlier static audit tables are relative to the canonical campaign folder.
