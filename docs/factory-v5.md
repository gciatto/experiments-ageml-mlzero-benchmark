# Factory v5: remove the adapter-only Python style restriction

## Cause and bounded correction

V4's adapter required literal-only module-level assignments. It incorrectly
rejected ordinary Python initialization, including metric scorer construction
and logging. These rejections are harness errors, not MLZero method failures.
V5 removes this filter entirely; it does not replace it with a whitelist.

Direct factory modules retain normal Python import semantics, as in AGE-ML.
Parsing checks Python syntax; import and callable/factory failures are reported
by real execution. Runtime factory bindings and repeated definitions are not
rejected by a stylistic AST rule. The existing static MODEL_FACTORY_CONTENT
extraction remains unchanged and logged; it neither executes arbitrary outer
scripts nor repairs unresolved expressions. Direct module source is unchanged.

The unpublished draft that added Conda/CPU installation advice was discarded.
V5 retains the v4 Bash template unchanged. It does not repair generated Conda
commands, dependency choices, model code, hyperparameters, or failed training.
Failures are handled by MLZero's existing debug mechanism inside the same hour.

## Comparison boundaries

| Aspect | AGE-ML | MLZero matched-CV adapter v5 |
| --- | --- | --- |
| Dataset and folds | Frozen campaign data and ten folds | Those same exported data and folds |
| Selection metric | Mean of complete fold scores | Same; recomputed from OOF predictions |
| Randomness | Seed 42, exposed nested random_state values | Seed 42, exposed nested random_state values; wrappers must seed internal libraries |
| Shared search wall budget | 3600 seconds | 3600 seconds, including setup and evaluation |
| Factory import | Normal Python module execution | Normal Python module execution, no module-level whitelist |
| Candidate fitting | Fresh factory/clone per fold | Fresh factory call per fold; non-sklearn wrappers remain supported |
| Semantic conformity judge | AGE-ML component, ablated independently | No extra semantic judge imposed on MLZero |
| Search and code repair | AGE-ML planner, grid and retry policy | Native MLZero search/debug; no manual candidate repair |

This aligns the evaluation contract, not the two methods' search algorithms.
Import semantics are not a security sandbox or proof of no data leakage. A
factory can violate the requested fresh/unfitted contract without static checks
proving it; AGE-ML's judge is likewise not a proof. Neither model receives a
higher score without an actual complete, verified CV export. CPU limits remain
16 CPUs, memory.high 40 GiB, memory.max 50 GiB, no swap for local search scopes;
temperature 0 and both Gemini backends are unchanged.

## Apply in the existing repository, with no environment reinstall

Stop the current launcher in its existing tmux terminal before replacing files.
Do not delete results, manually unlock an active campaign, or change environments
while a search is running. Extract this modified-files archive at repository root.

```bash
cd /root/experiments-ageml-mlzero-benchmark
bash scripts/check-factory-v5.sh
bash scripts/prepare-factory-v5.sh
```

The first command checks the existing pinned runtime and native selection with
synthetic ten-fold fits. No Gemini request is made. The second bootstraps both
new campaign configs from the actual AGE-ML manifests/data and prints dry runs.
No runtime resealing or upstream source patch is needed beyond the existing
v4 patches. The old configurations contain the old prompt: do not reuse them.

Inside tmux, launch the newly prepared configs:

```bash
unset AGEML_BENCH_PILOT
bash scripts/run-local-hamlet-cv.sh \
  configs/gemini25-hamlet-cv-local-factory-v5.yml \
  configs/gemini31-hamlet-cv-local-factory-v5.yml
```

The launcher requests GEMINI_API_KEY if absent. Existing output is not deleted.
The new suffix and source/config hashes separate v5 from v4. For a clean
comparison, use the complete predeclared v5 cohort, including legitimate failed
runs; do not combine the successful v4 runs with selectively retried failures.
V4 remains diagnostic evidence of a harness-affected cohort. No historical result
has been reclassified as a valid v5 result or recovered after its deadline.

## Regression policy

Tests exercise scorer calls, logging, computed constants, helper initialization,
conditional and alias-bound factories, normal import failures, unchanged source,
and static-extraction failures. Native selection smoke includes scorer/logging
initialization with actual toy folds. Existing tests cover invalid predictions,
OOF identities, score provenance and shared deadlines. Any future adapter
restriction must have a protocol rationale and an AGE-ML comparison; arbitrary
Python style restrictions are not part of this benchmark contract.

Offline tests cannot guarantee a newly generated AutoGluon model fits within
one hour. No paid Gemini campaign was executed in the authoring environment.

## Authoring validation

- Full offline suite: 73 passed (one environment-level pytest plugin deprecation warning).
- Patched native selection: seven cases passed, including ordinary scorer/logging
  initialization and actual classification/regression toy ten-fold evaluation.
- Ruff lint/format on the four changed Python files: passed.
- Mypy on cv_gate.py, hamlet_cv.py and test_cv_gate.py: passed.
- Both new Bash scripts: bash -n passed.
- The Bash coding template is byte-for-byte identical to the supplied v4 bundle.
- Repository-wide lint also reports 18 findings in other files; repository-wide
  mypy reports an existing tests/conftest.py NumPy choice typing error. Those
  files are outside this correction and were not modified to mask the findings.
