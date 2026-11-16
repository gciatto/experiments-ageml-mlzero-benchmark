# Factory interface correction after the first cv-gate-v3 run

The supplied node_0 code is a generator: its build_model function is embedded in
MODEL_FACTORY_CONTENT, not defined at module scope in generated_code.py. V3
rejected the outer script before installation/evaluation. The native error
analyzer incorrectly attributed this to whitespace. Leading blank lines,
comments, imports and helper classes/functions have always been permitted.

The generator has an additional unresolved outer-f-string reference: unique_id
belongs to the inner fit method but is interpolated while constructing the outer
string. That is a genuine code-generation error. It must receive debugging
feedback, not be silently repaired or evaluated as an experimental success.

## Changes

- Static MODEL_FACTORY_CONTENT strings/f-strings can be extracted using literal
  bindings only. Arbitrary Python calls are not executed. Unsupported/dynamic
  expressions report the specific cause. Model choices and hyperparameters are
  not rewritten. Normalization is logged with original/factory source hashes.
- Extraction happens before generated_code.py is saved and before the Bash coder
  runs, so Bash sees the factory module and never executes the outer generator.
- Matched Python/Bash templates replace the original standalone train/predict
  templates. Native library advice, selected tool, parent/debug context and
  tutorials remain available, under the matched-CV contract. Bash prepares the
  node environment; the harness performs fold evaluation.
- The exact final prompt, including the contract, is saved after augmentation.
  Previously the saved *_coder_prompt.txt could precede the contract appended by
  BasePrompt.build, which made the diagnostic log misleading.
- V3's mandatory ten-fold scoring BEFORE native winner/MCTS updates is retained.
  No post-deadline fitting or post-hoc re-selection recovers the earlier runs.

## Apply on the experiment container

Stop the foreground launcher with Ctrl+C in its existing tmux terminal. Apply
this modified-files ZIP in the repository, or commit/push locally and pull the
changes on the container. No package reinstall or Docker is required.

```bash
cd /root/experiments-ageml-mlzero-benchmark
set -o pipefail
bash scripts/repair-local-factory-v4.sh 2>&1 | tee repair-local-factory-v4.log
```

The script checks the campaign lock/scopes and previous source/package freeze,
preserves runtime-before-factory-v4.json, applies patch 0004, verifies BGE retrieval,
then runs the real native prompt-building and simulate methods with synthetic
candidates. Smoke cases include a wrapped static factory accepted after extraction
and an unresolved wrapper rejected. Runtime.json is resealed only on success.

After Factory interface repair complete:

```bash
bash scripts/prepare-factory-v4.sh
```

This creates NEW configurations/campaign IDs ending in factory-v4. It preserves
both models, actual AGE-ML data/folds, seed 42, temperature 0, 3600 seconds and
16 CPU/40 GiB memory.high/50 GiB memory.max/no swap. Earlier attempts remain in
their original campaign folders, with their API usage and generated code.

Recommended first check, using the first planned Adult repetition:

```bash
AGEML_BENCH_PILOT=1 bash scripts/run-local-hamlet-cv.sh \
  configs/gemini25-hamlet-cv-local-factory-v4.yml
```

For the full predetermined pair of campaigns:

```bash
unset AGEML_BENCH_PILOT
bash scripts/run-local-hamlet-cv.sh \
  configs/gemini25-hamlet-cv-local-factory-v4.yml \
  configs/gemini31-hamlet-cv-local-factory-v4.yml
```

Terminal outcomes in this new cohort are skipped on continuation. A pilot is
part of that cohort, not an extra selectively discarded run. Do not repeat
legitimate method failures until they succeed. If the first run has another
integration error, stop the rollout and inspect its actual artifacts.

## Evidence and limitations

62 offline tests pass, including a minimized regression fixture based on the
reported generator, preservation of parameters during static extraction, rejection
of unresolved/call-based strings, and acceptance of whitespace/comments/helpers.
All four patches apply in order to pinned upstream f228cd7. Native prompt smoke
checks that Python/Bash no longer receive the legacy template and that the saved
prompt equals the sent prompt. Native selection smoke verifies six cases with
real toy ten-fold fits. Lint, typing and Bash syntax pass.

The supplied AutoGluon factory has not been fitted here and its library arguments
have not been silently corrected. No live Gemini generation has been executed
in the authoring environment. The container-side checks validate integration,
not that every generated model can finish all ten fits within the hour.
