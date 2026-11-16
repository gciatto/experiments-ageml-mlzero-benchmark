# Retrieval dependency repair: start a clean, separately identified cohort

Observed: BGE loading raises `BertModel.__init__() got an unexpected keyword
argument 'dtype'`. The retriever consequently supplies no tutorials, the reranker
constructs an empty human prompt, and native Gemini raises `contents are required`.
The reported MNIST attempt exited with process_error after 633 seconds, without
a committed candidate. This is not evidence of a completed one-hour ML search.
The secondary data-reader `applymap` failure is handled by the pandas pin.

The original setup failed to catch this: upstream build_indices logs model-loading
exceptions and continues. The setup must not seal a runtime on that basis alone.

## Apply and repair

Stop the foreground campaign launcher with Ctrl+C inside its tmux terminal.
Apply this modified-files ZIP in the repository (or commit/push locally, then
pull on the experiment container). It is a delta on the local-executor update.
Do not remove old artifacts, their usage logs or their configurations.

In the existing tmux terminal, in /root/experiments-ageml-mlzero-benchmark:

```bash
set -o pipefail
bash scripts/repair-local-retrieval.sh 2>&1 | tee repair-local-retrieval.log
```

The script refuses to run with an active campaign lock or local search scope.
It preserves the old runtime manifest as runtime-before-retrieval-v2.json,
then installs compatible retrieval versions into **.mlzero-local/env only**:
FlagEmbedding 1.3.5, transformers 4.48.3, sentence-transformers 3.4.1, peft 0.15.2,
pandas 2.2.3. Native Gemini remains 4.2.1. Neither the orchestration .venv nor
AGE-ML's .venv is changed. It runs pip check and rebuilds the tutorial indices.

Before resealing runtime.json it performs six real offline retrieval queries
on the cached BGE model: normal/condensed tutorials for autogluon.tabular,
machine learning, autogluon.multimodal. Empty results, non-finite scores, blank
content and model-load errors stop the repair. No Gemini calls occur here.
The same retrieval check runs in doctor before new searches.

If repair stops partway, inspect the error and rerun after addressing it; with
runtime.json absent the partially modified runtime is not accepted as sealed.
A successful repair must not be repeated over the preserved previous manifest.

## Prepare a new cohort and launch

After repair succeeds:

```bash
bash scripts/prepare-retrieval-v2.sh
bash scripts/run-local-hamlet-cv.sh \
  configs/gemini25-hamlet-cv-local-retrieval-v2.yml \
  configs/gemini31-hamlet-cv-local-retrieval-v2.yml
```

The preparation script reads the original completed AGE-ML references under
/root/age-ml, uses 16 CPU/50 GiB, seed 42, temperature 0, ten unchanged folds and
3600 seconds. It creates NEW configurations and adds -retrieval-v2 to both
campaign IDs. Older invalid attempts are retained separately and their API usage
is not erased; they must not be treated as completed competitor trials. Old
configurations are deliberately rejected after the runtime identity changes.
AGE_ML_ROOT and AGEML_LOCAL_RUNTIME can override those two paths if necessary.

The launcher requests GEMINI_API_KEY if missing, validates the environment and
backends, then runs Gemini 2.5 followed by 3.1. Detach with Ctrl+B then D.
A new unhandled process error before any selectable export now stops the campaign
immediately, after saving the attempt, instead of repeating the same setup failure
across all datasets. A genuine no-candidate outcome on normal completion/deadline
is still recorded as an outcome; it is not automatically converted to a success.

## Checks completed here

45 offline tests pass, including propagation of model-load failures and rejection
of empty/invalid retrieval. Python lint/type checks and Bash syntax pass. An actual
FlagAutoModel with the pinned libraries loads and encodes a tiny local BertModel
using the exact upstream options without the dtype error. The real upstream
TutorialIndexer also rebuilt/saved/reloaded FAISS indices and passed six queries
with that tiny model. This is a compatibility test, not a BGE-quality test. No live Gemini search
or full CT 123 repair has been executed in this authoring environment. The
container-side BGE smoke check is therefore mandatory before a timed campaign.
