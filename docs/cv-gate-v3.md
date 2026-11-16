# Mandatory CV scoring before native MLZero selection

## What the Adult records prove

Both recorded searches reached their 3600-second deadline. The selected folders
contained saved native models and results.csv, but no model_factory.py,
cv_evaluation.json or cv_predictions.csv. Native 0.8493 is therefore not a verified
ten-fold metric. The other native selection had no score at all. Neither attempt
can be presented as a matched CV result. Do not re-fit them after the deadline
and silently replace their original outcome with a new score.

The previous adapter requested a factory/evaluator in README/user instructions
and verified it only AFTER the native search selected its winner. That was
insufficient: native tool/tutorial prompts and the execution judge still allowed
ordinary native results to become incumbents. This update changes the integration
with the evaluation protocol, not just the end-of-run file check.

## New integration boundary

- Python/Bash coder prompts receive a mandatory, last-position CV contract.
- The Python coder returns a self-contained module defining build_model(X_train,
  y_train). Import-time fitting/data loading is rejected before script execution.
- The Bash coder prepares each node's conda_env. No separate holdout/full-data
  training is requested. Model choice, wrappers and dependencies remain free.
- After dependency preparation, NodeManager.simulate invokes the trusted evaluator
  in that node's environment for ten new fold fits, inside the remaining shared
  exploration deadline. The harness writes the generated module as model_factory.py.
- The shared verifier checks export hashes, OOF row/fold identities, all fold
  metrics and their arithmetic mean. Classification supplies balanced accuracy;
  regression supplies negative mean-fold RMSE to native maximization.
- This happens BEFORE updating _best_node, last_successful_node and MCTS rewards.
  Missing/invalid factories or failed evaluations return failure with concrete
  feedback through the native debug mechanism; an LLM-claimed score is ignored.
- The incumbent export flag now checks the three CV files, not predict.py.
- Post-search verification reuses exactly the same verifier and does not reselect.

MCTS and its model/tool choice are retained, but MLZero is adapted to the matched
factory/CV interface. This adaptation must be disclosed when describing the
competitor. Do not call these independent test scores, or claim the unmodified
upstream evaluation protocol was used. Generated Python is still trusted code
inside the dedicated experiment container, not an adversarial security sandbox.

## Update the existing local installation

Stop the foreground launcher with Ctrl+C inside tmux. Apply the ZIP in the
benchmark root (or commit/push locally and pull on the experiment container).
Preserve the earlier attempts, models and their usage logs for diagnosis.

Inside the existing tmux terminal:

```bash
cd /root/experiments-ageml-mlzero-benchmark
set -o pipefail
bash scripts/repair-local-cv-gate.sh 2>&1 | tee repair-local-cv-gate.log
```

This refuses to run with a campaign lock or active local scope. It verifies the
previously recorded source/package freeze, preserves the old runtime manifest,
applies patch 0003 to the dedicated MLZero checkout and updates patch provenance.
It does NOT reinstall packages or touch AGE-ML. It reruns BGE retrieval and a
functional native-selection smoke test, then seals runtime.json only on success.

The CV smoke test extracts the hash-verified pinned NodeManager.simulate method
and executes it with real subprocess CV and minimal synthetic candidates, without
LLM calls: classifier score 0.5, regressor MCTS score -5.0, a legacy script rejected
before execution, and a broken factory rejected without updating the winner.
This validates the selection boundary, not the ability of a real LLM to generate
all required factories. Doctor reruns both smoke checks before campaigns.

## Prepare a separate cohort and first run only Adult/repetition 1

After the repair prints CV gate repair complete:

```bash
bash scripts/prepare-cv-gate-v3.sh

AGEML_BENCH_PILOT=1 bash scripts/run-local-hamlet-cv.sh \
  configs/gemini25-hamlet-cv-local-cv-gate-v3.yml
```

This prompts for GEMINI_API_KEY if missing and runs ONLY Adult, repetition 1,
with the full unchanged one-hour/ten-fold budget. It is the FIRST planned run
of the new cohort, not an extra selectively discarded pilot. Check its recorded
outcome and search.log. Do not start all 90 runs if the integration still fails.
A genuine no-valid-candidate result remains an outcome; do not repeat trials
until success. If an infrastructure/interface defect requires another change,
record it and start a separately identified cohort.

On a successful end-to-end result (completed, finite CV score and all three
export files), launch the full predetermined campaigns:

```bash
bash scripts/run-local-hamlet-cv.sh \
  configs/gemini25-hamlet-cv-local-cv-gate-v3.yml \
  configs/gemini31-hamlet-cv-local-cv-gate-v3.yml
```

The completed first run is skipped, so each model still has exactly 9 datasets x
5 repetitions. The old retrieval-v2 attempts remain in their own campaign folders.
Local runtime/model references, seed 42, temperature 0, unchanged CSV/folds,
3600 s, 16 CPU, MemoryHigh 40 GiB, MemoryMax 50 GiB, and no swap are retained.
Detaching uses Ctrl+B then D. Do not start another launcher concurrently.

## Validation completed

55 offline tests pass; Python lint/type checks and Bash syntax pass. All three
patches apply cleanly in order to pinned upstream f228cd7. The actual patched
native simulate method has been exercised with real ten-fold toy fits and its
best-node updates checked for both valid and invalid candidates. No live Gemini
factory-generation search has been executed in the authoring environment; the
single predetermined Adult run is required before the full rollout.
